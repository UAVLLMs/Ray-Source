"""RAGV6 Chunk Studio 管理 API。

路由由 ``api_server.py`` 以现有 Bearer 鉴权挂载。所有修改操作均复用
``chunk_pipeline`` 的预览、质量检查、备份与原子发布逻辑；向量索引重建在
后台线程运行，不阻塞普通问答请求。
"""

from __future__ import annotations

import os
import shutil
import threading
import traceback
from typing import Any
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chunk_pipeline import (
    ChunkPipelineError,
    DATA_DIR,
    MANUAL_SOURCE_DIR,
    build_artifacts,
    decode_uploaded_source,
    get_repository_status,
    list_backups,
    list_manuals,
    load_manual_detail,
    mark_index_built,
    publish_artifacts,
    rollback_backup,
    safe_manual_name,
    utc_now,
)


class SourceRequest(BaseModel):
    manual: str = Field(..., min_length=1, max_length=96)
    filename: str = Field(default="manual.md", max_length=255)
    text: str | None = Field(default=None, max_length=8_000_000)
    content_base64: str | None = Field(default=None, max_length=36_000_000)
    options: dict[str, Any] = Field(default_factory=dict)


class PublishRequest(SourceRequest):
    replace_existing: bool = False
    rebuild_index: bool = False


class RebuildRequest(BaseModel):
    batch_size: int = Field(default=32, ge=1, le=256)


class RollbackRequest(BaseModel):
    backup_id: str = Field(..., min_length=8, max_length=120)
    rebuild_index: bool = False


class SearchTestRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    manual: str | None = Field(default=None, max_length=96)
    top_k: int = Field(default=6, ge=1, le=20)


_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.RLock()
_REBUILD_LOCK = threading.Lock()
_INDEX_DIR = DATA_DIR / "index"
_INDEX_STAGING_DIR = DATA_DIR / "chunk-admin" / "index-staging"
_INDEX_BACKUP_DIR = DATA_DIR / "chunk-admin" / "index-backups"
_INDEX_FILENAMES = ("dense.faiss", "retrieval_index.pkl")


def _ok(data: Any, msg: str = "success") -> dict[str, Any]:
    return {"code": 0, "msg": msg, "data": data}


def _job_snapshot(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise ChunkPipelineError("任务不存在")
        return dict(job)


def _create_job(kind: str, detail: dict[str, Any] | None = None) -> str:
    job_id = f"chunk_{uuid.uuid4().hex[:16]}"
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "progress": 0,
            "message": "等待执行",
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "detail": detail or {},
            "error": None,
        }
        if len(_JOBS) > 100:
            completed = [
                key
                for key, value in _JOBS.items()
                if value.get("status") in {"succeeded", "failed"}
            ]
            for key in completed[: max(0, len(_JOBS) - 100)]:
                _JOBS.pop(key, None)
    return job_id


def _update_job(job_id: str, **values: Any) -> None:
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(values)


def _start_index_job(batch_size: int, reason: str) -> str:
    with _JOBS_LOCK:
        active = next(
            (
                job
                for job in _JOBS.values()
                if job.get("kind") == "rebuild_index"
                and job.get("status") in {"queued", "running"}
            ),
            None,
        )
        if active:
            return str(active["job_id"])
    job_id = _create_job("rebuild_index", {"batch_size": batch_size, "reason": reason})
    thread = threading.Thread(
        target=_run_index_job,
        args=(job_id, batch_size),
        name=f"ragv6-{job_id}",
        daemon=True,
    )
    thread.start()
    return job_id


def _run_index_job(job_id: str, batch_size: int) -> None:
    if not _REBUILD_LOCK.acquire(blocking=False):
        _update_job(
            job_id,
            status="failed",
            message="已有索引重建任务正在运行",
            finished_at=utc_now(),
            error="index rebuild already running",
        )
        return
    try:
        _update_job(
            job_id,
            status="running",
            progress=5,
            message="正在加载 Chunk 数据",
            started_at=utc_now(),
        )
        from retrieval_engine import RetrievalEngine

        staging_dir = _INDEX_STAGING_DIR / job_id
        backup_dir = _INDEX_BACKUP_DIR / job_id
        staging_dir.mkdir(parents=True, exist_ok=False)
        engine = RetrievalEngine(index_dir=staging_dir)
        _update_job(job_id, progress=15, message="正在暂存区生成 BM25 与向量索引")
        engine.build_index(batch_size=batch_size)
        for filename in _INDEX_FILENAMES:
            if not (staging_dir / filename).is_file():
                raise RuntimeError(f"暂存索引缺少文件: {filename}")

        _update_job(job_id, progress=88, message="正在备份旧索引")
        _INDEX_DIR.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir(parents=True, exist_ok=False)
        previous_files: set[str] = set()
        for filename in _INDEX_FILENAMES:
            current = _INDEX_DIR / filename
            if current.is_file():
                shutil.copy2(current, backup_dir / filename)
                previous_files.add(filename)

        _update_job(job_id, progress=92, message="正在原子切换线上索引")
        switched: list[str] = []
        try:
            for filename in _INDEX_FILENAMES:
                os.replace(staging_dir / filename, _INDEX_DIR / filename)
                switched.append(filename)
            # 从正式目录重新加载一次，确认两个索引文件彼此匹配；通过后才热切换。
            live_engine = RetrievalEngine()
            live_engine.load_index()
        except Exception:
            for filename in switched:
                target = _INDEX_DIR / filename
                previous = backup_dir / filename
                if filename in previous_files and previous.is_file():
                    shutil.copy2(previous, target)
                elif target.exists():
                    target.unlink()
            raise

        # api_server 模块已经完成初始化；替换对象引用不会破坏正在执行的请求，
        # 新请求会自动使用新索引。
        try:
            import api_server

            api_server._engine = live_engine
        except Exception:
            # 命令行或独立测试环境没有 api_server 时，索引文件仍已成功发布。
            pass
        mark_index_built()
        _update_job(
            job_id,
            status="succeeded",
            progress=100,
            message="索引重建并切换完成",
            finished_at=utc_now(),
            result=get_repository_status(),
        )
    except Exception as exc:  # noqa: BLE001
        _update_job(
            job_id,
            status="failed",
            message="索引重建失败",
            finished_at=utc_now(),
            error=str(exc),
            traceback=traceback.format_exc(limit=12),
        )
    finally:
        try:
            staging_dir = _INDEX_STAGING_DIR / job_id
            if staging_dir.is_dir():
                shutil.rmtree(staging_dir)
        except OSError:
            pass
        _REBUILD_LOCK.release()


def _decode_request(req: SourceRequest) -> str:
    return decode_uploaded_source(
        filename=req.filename,
        content_base64=req.content_base64,
        text=req.text,
    )


def _preview_payload(artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "manual": artifacts["manual"],
        "document_title": artifacts["document_title"],
        "source_sha256": artifacts["source_sha256"],
        "options": artifacts["options"],
        "quality": artifacts["quality"],
        "sections": artifacts["section_chunks"],
        "retrieval_chunks": artifacts["retrieval_chunks"],
    }


def create_chunk_admin_router() -> APIRouter:
    router = APIRouter(prefix="/admin/chunks", tags=["chunk-admin"])

    @router.get("/status")
    def status() -> dict[str, Any]:
        with _JOBS_LOCK:
            active_jobs = [
                dict(job)
                for job in _JOBS.values()
                if job.get("status") in {"queued", "running"}
            ]
        return _ok({**get_repository_status(), "active_jobs": active_jobs})

    @router.get("/manuals")
    def manuals() -> dict[str, Any]:
        return _ok({"manuals": list_manuals(), "repository": get_repository_status()})

    @router.get("/manual/{manual}")
    def manual_detail(manual: str) -> dict[str, Any]:
        try:
            return _ok(load_manual_detail(manual))
        except ChunkPipelineError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/preview")
    def preview(req: SourceRequest) -> dict[str, Any]:
        try:
            source = _decode_request(req)
            artifacts = build_artifacts(req.manual, source, req.options)
            return _ok(_preview_payload(artifacts))
        except ChunkPipelineError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/publish")
    def publish(req: PublishRequest) -> dict[str, Any]:
        try:
            source = _decode_request(req)
            artifacts = build_artifacts(req.manual, source, req.options)
            result = publish_artifacts(
                artifacts,
                data_dir=DATA_DIR,
                source_dir=MANUAL_SOURCE_DIR,
                replace_existing=req.replace_existing,
            )
            if req.rebuild_index:
                result["job_id"] = _start_index_job(
                    batch_size=32,
                    reason=f"publish:{artifacts['manual']}",
                )
                result["index_status"] = "rebuilding"
            return _ok(result, "manual published")
        except ChunkPipelineError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/rebuild")
    def rebuild(req: RebuildRequest) -> dict[str, Any]:
        job_id = _start_index_job(req.batch_size, "manual_admin")
        return _ok(_job_snapshot(job_id), "index rebuild queued")

    @router.get("/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        try:
            return _ok(_job_snapshot(job_id))
        except ChunkPipelineError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/backups")
    def backups() -> dict[str, Any]:
        return _ok({"backups": list_backups()})

    @router.post("/rollback")
    def rollback(req: RollbackRequest) -> dict[str, Any]:
        try:
            result = rollback_backup(req.backup_id)
            if req.rebuild_index:
                result["job_id"] = _start_index_job(
                    batch_size=32,
                    reason=f"rollback:{req.backup_id}",
                )
                result["index_status"] = "rebuilding"
            return _ok(result, "backup restored")
        except ChunkPipelineError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/search-test")
    def search_test(req: SearchTestRequest) -> dict[str, Any]:
        try:
            from retrieval_engine import RetrievalEngine

            try:
                import api_server

                engine = api_server._engine
            except Exception:
                engine = None
            if engine is None:
                engine = RetrievalEngine()
                engine.ensure_index()
            products = [safe_manual_name(req.manual)] if req.manual else None
            results, filtered = engine.search_manual(
                keywords=[],
                semantic_query=req.question,
                original_query=req.question,
                top_k=req.top_k,
                products=products,
            )
            rows = [
                {
                    "product": item.product,
                    "heading": item.heading,
                    "text": item.text,
                    "pics": item.pics,
                    "score": item.score,
                    "parent_section_id": item.source.get(
                        "parent_section_id",
                        item.source.get("section_id"),
                    ),
                    "chunk_id": item.chunk_id,
                }
                for item in results
            ]
            return _ok({"results": rows, "filtered_count": filtered})
        except (ChunkPipelineError, RuntimeError, OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router

"""NL2SQL chunk 管理层。

该模块把当前 JSON chunk 库同步到本地 SQLite 镜像，并提供标准的自然语言到 SQL
流程。LLM 负责生成 SQL，程序负责 schema 注入、SQL 安全校验、参数化执行和变更
审批。chunk 内容仍以现有 JSON/发布流水线为权威源，LLM 不能直接改线上检索资产。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chunk_pipeline import DATA_DIR, utc_now


SQL_DB_PATH = DATA_DIR / "chunk-admin" / "chunks.sqlite"
RETRIEVAL_CHUNKS_PATH = DATA_DIR / "retrieval_chunks.json"
SECTION_CHUNKS_PATH = DATA_DIR / "section_chunks.json"
CATALOG_PATH = DATA_DIR / "catalog.json"
MAX_PREVIEW_ROWS = 50
MAX_TEXT_PREVIEW = 600
NL2SQL_MAX_SQL_LENGTH = 8000
NL2SQL_ALLOWED_TABLES = {
    "manuals", "sections", "chunks", "chunk_tags", "chunk_pics", "sync_meta",
    "chunk_change_requests",
}

_STORE: "ChunkSqlStore | None" = None
_STORE_LOCK = threading.Lock()


class ChunkSqlRequest(BaseModel):
    instruction: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=20, ge=1, le=MAX_PREVIEW_ROWS)


class ChunkSqlSyncRequest(BaseModel):
    force: bool = False


class Nl2SqlRequest(BaseModel):
    instruction: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field(default="query", pattern="^(query|change_plan)$")
    limit: int = Field(default=50, ge=1, le=MAX_PREVIEW_ROWS)


class ChunkChangeApproval(BaseModel):
    request_id: int = Field(..., ge=1)
    approve: bool = False


def _json_rows(path: Path, key: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if key and isinstance(data, dict) and isinstance(data.get(key), list):
        return [row for row in data[key] if isinstance(row, dict)]
    return []


def _catalog_rows() -> list[tuple[str, dict[str, Any]]]:
    if not CATALOG_PATH.is_file():
        return []
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    return [(str(manual), value) for manual, value in data.items() if isinstance(value, dict)]


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False, separators=(",", ":"))


def _signature() -> str:
    digest = hashlib.sha256()
    for path in (RETRIEVAL_CHUNKS_PATH, SECTION_CHUNKS_PATH, CATALOG_PATH):
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


class ChunkSqlStore:
    def __init__(self, db_path: Path = SQL_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self.lock, self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS manuals (
                    manual_id TEXT PRIMARY KEY,
                    lang TEXT NOT NULL DEFAULT '',
                    source_sha256 TEXT NOT NULL DEFAULT '',
                    total_chars INTEGER NOT NULL DEFAULT 0,
                    total_pics INTEGER NOT NULL DEFAULT 0,
                    section_count INTEGER NOT NULL DEFAULT 0,
                    retrieval_chunk_count INTEGER NOT NULL DEFAULT 0,
                    synced_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS sections (
                    manual_id TEXT NOT NULL,
                    section_id INTEGER NOT NULL,
                    heading TEXT NOT NULL DEFAULT '',
                    heading_path_json TEXT NOT NULL DEFAULT '[]',
                    chapter TEXT NOT NULL DEFAULT '',
                    subheading TEXT NOT NULL DEFAULT '',
                    subsubheading TEXT NOT NULL DEFAULT '',
                    heading_level INTEGER NOT NULL DEFAULT 0,
                    text TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    char_len INTEGER NOT NULL DEFAULT 0,
                    pic_count INTEGER NOT NULL DEFAULT 0,
                    is_special INTEGER NOT NULL DEFAULT 0,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    pics_json TEXT NOT NULL DEFAULT '[]',
                    PRIMARY KEY (manual_id, section_id)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    manual_id TEXT NOT NULL,
                    lang TEXT NOT NULL DEFAULT '',
                    parent_section_id INTEGER,
                    source_section_ids_json TEXT NOT NULL DEFAULT '[]',
                    subchunk_id INTEGER NOT NULL DEFAULT 0,
                    heading TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    char_start INTEGER NOT NULL DEFAULT 0,
                    char_end INTEGER NOT NULL DEFAULT 0,
                    char_len INTEGER NOT NULL DEFAULT 0,
                    pic_count INTEGER NOT NULL DEFAULT 0,
                    is_special INTEGER NOT NULL DEFAULT 0,
                    split_kind TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    pics_json TEXT NOT NULL DEFAULT '[]',
                    enabled INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS chunk_tags (
                    chunk_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, tag)
                );
                CREATE TABLE IF NOT EXISTS chunk_pics (
                    chunk_id TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    position INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (chunk_id, image_id, position)
                );
                CREATE TABLE IF NOT EXISTS sync_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS chunk_change_requests (
                    request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instruction TEXT NOT NULL,
                    sql_text TEXT NOT NULL,
                    params_json TEXT NOT NULL DEFAULT '[]',
                    explanation TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL DEFAULT '',
                    reviewed_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_manual ON chunks(manual_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(manual_id, parent_section_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_heading ON chunks(heading);
                CREATE INDEX IF NOT EXISTS idx_sections_heading ON sections(heading);
                """
            )

    def status(self) -> dict[str, Any]:
        self.sync_if_stale()
        with self.lock:
            manual_count = self.conn.execute("SELECT COUNT(*) FROM manuals").fetchone()[0]
            section_count = self.conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
            chunk_count = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            signature = self.conn.execute(
                "SELECT value FROM sync_meta WHERE key = 'source_signature'"
            ).fetchone()
            synced_at = self.conn.execute(
                "SELECT value FROM sync_meta WHERE key = 'synced_at'"
            ).fetchone()
        return {
            "db_path": str(self.db_path),
            "manual_count": int(manual_count),
            "section_count": int(section_count),
            "chunk_count": int(chunk_count),
            "source_signature": signature[0] if signature else "",
            "synced_at": synced_at[0] if synced_at else "",
            "nl2sql_mode": "standard_sql_with_guardrails",
            "read_only_source_mirror": True,
            "source_of_truth": "retrieval_chunks.json + section_chunks.json + catalog.json",
        }

    def sync_if_stale(self, force: bool = False) -> dict[str, Any]:
        source_signature = _signature()
        with self.lock:
            current = self.conn.execute(
                "SELECT value FROM sync_meta WHERE key = 'source_signature'"
            ).fetchone()
        if not force and current and current[0] == source_signature:
            return self.status_without_sync()

        chunks = _json_rows(RETRIEVAL_CHUNKS_PATH)
        sections = _json_rows(SECTION_CHUNKS_PATH)
        catalog = _catalog_rows()
        now = utc_now()
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM chunk_tags")
            self.conn.execute("DELETE FROM chunk_pics")
            self.conn.execute("DELETE FROM chunks")
            self.conn.execute("DELETE FROM sections")
            self.conn.execute("DELETE FROM manuals")

            chunk_counts: dict[str, int] = {}
            section_counts: dict[str, int] = {}
            for row in chunks:
                manual = str(row.get("product") or "")
                chunk_counts[manual] = chunk_counts.get(manual, 0) + 1
            for row in sections:
                manual = str(row.get("product") or "")
                section_counts[manual] = section_counts.get(manual, 0) + 1

            for manual, meta in catalog:
                self.conn.execute(
                    """INSERT INTO manuals
                    (manual_id, lang, source_sha256, total_chars, total_pics,
                     section_count, retrieval_chunk_count, synced_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        manual,
                        str(meta.get("lang") or ""),
                        str(meta.get("source_sha256") or ""),
                        int(meta.get("total_chars") or 0),
                        int(meta.get("total_pics") or 0),
                        int(meta.get("section_count") or section_counts.get(manual, 0)),
                        int(meta.get("retrieval_chunk_count") or chunk_counts.get(manual, 0)),
                        now,
                    ),
                )

            for row in sections:
                manual = str(row.get("product") or "")
                section_id = int(row.get("section_id") or 0)
                tags = row.get("tags") if isinstance(row.get("tags"), list) else []
                pics = row.get("pics") or row.get("linked_pics") or []
                self.conn.execute(
                    """INSERT INTO sections
                    (manual_id, section_id, heading, heading_path_json, chapter,
                     subheading, subsubheading, heading_level, text, summary,
                     char_len, pic_count, is_special, tags_json, pics_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        manual, section_id, str(row.get("heading") or ""),
                        _json(row.get("heading_path") or []), str(row.get("chapter") or ""),
                        str(row.get("subheading") or ""), str(row.get("subsubheading") or ""),
                        int(row.get("heading_level") or 0), str(row.get("text") or ""),
                        str(row.get("summary") or ""), int(row.get("char_len") or 0),
                        int(row.get("pic_count") or len(pics)), int(bool(row.get("is_special"))),
                        _json(tags), _json(pics),
                    ),
                )

            for row in chunks:
                chunk_id = str(row.get("chunk_id"))
                manual = str(row.get("product") or "")
                tags = row.get("tags") if isinstance(row.get("tags"), list) else []
                pics = row.get("pics") or row.get("linked_pics") or []
                self.conn.execute(
                    """INSERT INTO chunks
                    (chunk_id, manual_id, lang, parent_section_id,
                     source_section_ids_json, subchunk_id, heading, text, summary,
                     char_start, char_end, char_len, pic_count, is_special,
                     split_kind, tags_json, pics_json, enabled)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                    (
                        chunk_id, manual, str(row.get("lang") or ""),
                        row.get("parent_section_id"), _json(row.get("source_section_ids") or []),
                        int(row.get("subchunk_id") or 0), str(row.get("heading") or ""),
                        str(row.get("text") or ""), str(row.get("summary") or ""),
                        int(row.get("char_start") or 0), int(row.get("char_end") or 0),
                        int(row.get("char_len") or 0), int(row.get("pic_count") or len(pics)),
                        int(bool(row.get("is_special"))), str(row.get("split_kind") or ""),
                        _json(tags), _json(pics),
                    ),
                )
                for tag in tags:
                    self.conn.execute("INSERT OR IGNORE INTO chunk_tags(chunk_id, tag) VALUES (?, ?)", (chunk_id, str(tag)))
                for position, image_id in enumerate(pics):
                    self.conn.execute(
                        "INSERT OR IGNORE INTO chunk_pics(chunk_id, image_id, position) VALUES (?, ?, ?)",
                        (chunk_id, str(image_id), position),
                    )

            self.conn.execute(
                "INSERT INTO sync_meta(key, value) VALUES('source_signature', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (source_signature,),
            )
            self.conn.execute(
                "INSERT INTO sync_meta(key, value) VALUES('synced_at', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (now,),
            )
        return self.status_without_sync()

    def status_without_sync(self) -> dict[str, Any]:
        with self.lock:
            manual_count = self.conn.execute("SELECT COUNT(*) FROM manuals").fetchone()[0]
            section_count = self.conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
            chunk_count = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            synced_at = self.conn.execute("SELECT value FROM sync_meta WHERE key='synced_at'").fetchone()
        return {
            "db_path": str(self.db_path),
            "manual_count": int(manual_count),
            "section_count": int(section_count),
            "chunk_count": int(chunk_count),
            "synced_at": synced_at[0] if synced_at else "",
            "read_only_nl2sql": True,
            "source_of_truth": "retrieval_chunks.json + section_chunks.json + catalog.json",
        }

    def query(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(sql, params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for key in ("tags_json", "pics_json", "source_section_ids_json", "heading_path_json"):
                if key in item:
                    try:
                        item[key[:-5] if key.endswith("_json") else key] = json.loads(item.pop(key))
                    except (TypeError, json.JSONDecodeError):
                        pass
            if "text" in item:
                item["text_preview"] = str(item["text"])[:MAX_TEXT_PREVIEW]
                item.pop("text", None)
            result.append(item)
        return result

    def execute_sql(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        """Execute a validated read query and normalize rows for the API."""
        return self.query(sql, params)

    def save_change_request(
        self, instruction: str, sql: str, params: list[Any], explanation: str
    ) -> int:
        with self.lock, self.conn:
            cursor = self.conn.execute(
                """INSERT INTO chunk_change_requests
                (instruction, sql_text, params_json, explanation, status, created_at)
                VALUES (?, ?, ?, ?, 'pending', ?)""",
                (instruction, sql, _json(params), explanation, utc_now()),
            )
            return int(cursor.lastrowid)


def _store() -> ChunkSqlStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = ChunkSqlStore()
        return _STORE


def _quoted_terms(instruction: str) -> list[str]:
    values = re.findall(r"[“\"「『](.+?)[”\"」』]", instruction)
    if values:
        return [item.strip() for item in values if item.strip()]
    match = re.search(r"(?:包含|含有|关键词|关键字|关于|搜索|查找)\s*[:：]?\s*([^，。；;\n]+)", instruction)
    if match:
        value = re.split(r"(?:的chunk|的片段|的章节|的内容|并|且|，|。)", match.group(1), maxsplit=1)[0]
        value = value.strip(" \t:：")
        if value and len(value) <= 80:
            return [value]
    return []


def _manual_match(store: ChunkSqlStore, instruction: str) -> str | None:
    with store.lock:
        names = [row[0] for row in store.conn.execute("SELECT manual_id FROM manuals ORDER BY length(manual_id) DESC")]
    folded = instruction.casefold()
    for name in names:
        if name.casefold() in folded:
            return name
    return None


def _free_terms(instruction: str, manual: str | None) -> list[str]:
    """Extract explicit nouns without treating the whole sentence as a keyword."""
    text = instruction
    if manual:
        text = re.sub(re.escape(manual), " ", text, flags=re.IGNORECASE)
    for phrase in (
        "查询", "查找", "搜索", "列出", "显示", "介绍", "相关内容", "内容", "中的", "里面的",
        "手册", "章节", "chunk", "片段", "带图片", "带图", "有图片", "图片", "图像",
        "警告", "安全警告", "故障排除", "故障", "排除", "步骤", "操作", "部件", "统计",
        "多少", "数量", "有哪些", "所有", "全部", "标题包含", "正文包含", "关键词",
    ):
        text = text.replace(phrase, " ")
    english = re.findall(r"[A-Za-z][A-Za-z0-9+_-]{2,}", text)
    chinese = re.findall(r"[\u3400-\u9fff]{2,}", text)
    stop = {"the", "and", "for", "with", "from", "this", "that", "what", "how", "does", "manual"}
    result: list[str] = []
    for term in english + chinese:
        if term.casefold() not in stop and term not in result:
            result.append(term)
    return result[:4]


def _compile_instruction(instruction: str, limit: int, store: ChunkSqlStore) -> dict[str, Any]:
    text = instruction.strip()
    folded = text.casefold()
    if any(token in text for token in ("删除数据库", "删库", "drop table", "truncate")):
        raise ValueError("拒绝危险数据库操作；当前 NL2SQL 只允许白名单只读查询")

    manual = _manual_match(store, text)
    terms = _quoted_terms(text)
    terms = [
        term for term in terms
        if term != manual
        and not any(marker in term for marker in ("手册", "章节", "故障排除", "相关内容", "中的"))
    ]
    if not terms:
        terms = _free_terms(text, manual)
    ids = [str(value) for value in re.findall(r"(?:chunk|块|片段|编号|id)\s*#?\s*(\d+)", folded)]
    if not ids and re.search(r"\bchunk\s*\d+", folded):
        ids = re.findall(r"\bchunk\s*(\d+)", folded)

    wants_manuals = ("手册" in text or "manual" in folded) and any(token in text for token in ("列表", "列出", "有哪些", "多少", "清单", "全部"))
    wants_sections = any(token in text for token in ("章节", "父章节", "section"))
    wants_count = any(token in text for token in ("多少", "数量", "统计", "count"))
    has_pic = any(token in text for token in ("图片", "图像", "配图", "带图", "有图", "pic"))
    tag_map = {
        "警告": "warning", "安全": "warning", "故障": "troubleshooting",
        "排除": "troubleshooting", "步骤": "procedure", "操作": "procedure",
        "部件": "parts", "组件": "parts",
    }
    tags = sorted({tag for token, tag in tag_map.items() if token in text})

    params: list[Any] = []
    filters: list[str] = []
    facts: list[str] = []
    if manual:
        filters.append("c.manual_id = ?")
        params.append(manual)
        facts.append(f"手册={manual}")
    if ids:
        placeholders = ", ".join("?" for _ in ids)
        filters.append(f"c.chunk_id IN ({placeholders})")
        params.extend(ids)
        facts.append(f"chunk_id={','.join(ids)}")
    for term in terms:
        filters.append("(c.heading LIKE ? OR c.text LIKE ? OR c.summary LIKE ?)")
        value = f"%{term}%"
        params.extend([value, value, value])
        facts.append(f"关键词={term}")
    if has_pic:
        filters.append("c.pic_count > 0")
        facts.append("pic_count>0")
    for tag in tags:
        filters.append("EXISTS (SELECT 1 FROM chunk_tags ct WHERE ct.chunk_id = c.chunk_id AND ct.tag = ?)")
        params.append(tag)
        facts.append(f"tag={tag}")

    if wants_manuals and not ids and not terms and not has_pic and not tags:
        sql = "SELECT manual_id, lang, total_chars, total_pics, section_count, retrieval_chunk_count, synced_at FROM manuals ORDER BY manual_id LIMIT ?"
        params.append(limit)
        return {"operation": "select", "target": "manuals", "sql": sql, "params": params, "facts": facts or ["手册列表"]}

    if wants_sections and not ids:
        section_filters: list[str] = []
        section_params: list[Any] = []
        if manual:
            section_filters.append("s.manual_id = ?")
            section_params.append(manual)
        for tag in tags:
            section_filters.append("s.tags_json LIKE ?")
            section_params.append(f'%"{tag}"%')
        for term in terms:
            section_filters.append("(s.heading LIKE ? OR s.text LIKE ? OR s.summary LIKE ?)")
            value = f"%{term}%"
            section_params.extend([value, value, value])
        where = " AND ".join(section_filters) or "1=1"
        sql = f"SELECT s.manual_id, s.section_id, s.heading, s.text, s.char_len, s.pic_count, s.tags_json, s.pics_json FROM sections s WHERE {where} ORDER BY s.manual_id, s.section_id LIMIT ?"
        section_params.append(limit)
        return {"operation": "select", "target": "sections", "sql": sql, "params": section_params, "facts": facts + ["目标=章节"]}

    if not filters:
        raise ValueError("无法安全识别查询条件。请指定手册、chunk 编号、关键词、图片、警告或故障等条件")

    where = " AND ".join(filters)
    if wants_count:
        sql = f"SELECT c.manual_id, COUNT(*) AS chunk_count FROM chunks c WHERE {where} GROUP BY c.manual_id ORDER BY c.manual_id LIMIT ?"
        params.append(limit)
        target = "chunk_count"
    else:
        sql = f"""SELECT c.chunk_id, c.manual_id, c.lang, c.parent_section_id,
            c.source_section_ids_json, c.subchunk_id, c.heading, c.text,
            c.summary, c.char_start, c.char_end, c.char_len, c.pic_count,
            c.is_special, c.split_kind, c.tags_json, c.pics_json, c.enabled
            FROM chunks c WHERE {where}
            ORDER BY c.manual_id, c.parent_section_id, c.subchunk_id, c.chunk_id
            LIMIT ?"""
        params.append(limit)
        target = "chunks"
    return {"operation": "select", "target": target, "sql": sql, "params": params, "facts": facts}


NL2SQL_SYSTEM = """You are a SQL generation component for a product-manual Chunk database.
Generate one SQLite SQL statement from the user's Chinese or English request.
Return JSON only with exactly: sql, params, explanation, operation.
params must be a JSON array and every user value must use ? placeholders.
For mode=query, generate SELECT or WITH ... SELECT only.
For mode=change_plan, do not modify manuals, sections, chunks, chunk_tags, or chunk_pics.
Instead insert one pending change request into chunk_change_requests with the original
instruction, a short explanation, and a JSON patch in params_json. The request will be
reviewed and applied by a separate release pipeline.
Never use DROP, DELETE, ALTER, ATTACH, PRAGMA, VACUUM, multiple statements, or system tables.

Schema:
manuals(manual_id,lang,source_sha256,total_chars,total_pics,section_count,retrieval_chunk_count,synced_at)
sections(manual_id,section_id,heading,heading_path_json,chapter,subheading,subsubheading,
heading_level,text,summary,char_len,pic_count,is_special,tags_json,pics_json)
chunks(chunk_id,manual_id,lang,parent_section_id,source_section_ids_json,subchunk_id,heading,
text,summary,char_start,char_end,char_len,pic_count,is_special,split_kind,tags_json,pics_json,enabled)
chunk_tags(chunk_id,tag), chunk_pics(chunk_id,image_id,position)
chunk_change_requests(request_id,instruction,sql_text,params_json,explanation,status,created_at,reviewed_at)
"""


def _extract_llm_text(response: Any) -> str:
    blocks = getattr(response, "content", []) or []
    return "".join(str(getattr(block, "text", "") or "") for block in blocks).strip()


def _parse_nl2sql_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("NL2SQL 模型没有返回 JSON 对象")
    sql = str(payload.get("sql") or "").strip()
    params = payload.get("params")
    if not sql or not isinstance(params, list):
        raise ValueError("NL2SQL 返回缺少 sql 或 params")
    return {
        "sql": sql,
        "params": params,
        "explanation": str(payload.get("explanation") or ""),
        "operation": str(payload.get("operation") or "select"),
    }


def _validate_generated_sql(sql: str, mode: str) -> str:
    normalized = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.DOTALL).strip()
    if len(normalized) > NL2SQL_MAX_SQL_LENGTH or normalized.count(";") > 1:
        raise ValueError("SQL 过长或包含多条语句")
    if re.search(r"\b(DROP|DELETE|ALTER|ATTACH|PRAGMA|VACUUM|REINDEX|DETACH)\b", normalized, re.IGNORECASE):
        raise ValueError("拒绝危险 SQL 操作")
    tables = {
        name.casefold()
        for name in re.findall(r"\b(?:FROM|JOIN|UPDATE|INTO|TABLE)\s+([A-Za-z_][A-Za-z0-9_]*)", normalized, re.IGNORECASE)
    }
    unknown = tables - {name.casefold() for name in NL2SQL_ALLOWED_TABLES}
    if unknown:
        raise ValueError(f"SQL 引用了未授权表: {sorted(unknown)}")
    if mode == "query" and not re.match(r"^(?:SELECT|WITH)\b", normalized, re.IGNORECASE):
        raise ValueError("query 模式只允许 SELECT")
    if mode == "change_plan" and not re.match(r"^INSERT\s+INTO\s+chunk_change_requests\b", normalized, re.IGNORECASE):
        raise ValueError("change_plan 只能写入待审批变更表")
    return normalized.rstrip(";").strip()


def create_chunk_sql_router() -> APIRouter:
    router = APIRouter(prefix="/admin/chunk-sql", tags=["chunk-sql-admin"])

    @router.get("/status")
    def status() -> dict[str, Any]:
        return {"code": 0, "msg": "success", "data": _store().status()}

    @router.post("/sync")
    def sync(req: ChunkSqlSyncRequest) -> dict[str, Any]:
        return {"code": 0, "msg": "chunk SQL mirror synced", "data": _store().sync_if_stale(force=req.force)}

    @router.post("/plan")
    def plan(req: ChunkSqlRequest) -> dict[str, Any]:
        store = _store()
        store.sync_if_stale()
        try:
            compiled = _compile_instruction(req.instruction, req.limit, store)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        started = time.perf_counter()
        rows = store.query(compiled["sql"], compiled["params"])
        elapsed = round(time.perf_counter() - started, 4)
        return {
            "code": 0,
            "msg": "read-only SQL plan generated",
            "data": {
                "instruction": req.instruction,
                "operation": compiled["operation"],
                "target": compiled["target"],
                "sql": compiled["sql"],
                "params": compiled["params"],
                "facts": compiled["facts"],
                "rows": rows,
                "row_count": len(rows),
                "elapsed_s": elapsed,
                "write_enabled": False,
                "source_of_truth": "retrieval_chunks.json + section_chunks.json + catalog.json",
            },
        }

    @router.post("/nl2sql")
    def nl2sql(req: Nl2SqlRequest) -> dict[str, Any]:
        """Standard Text-to-SQL entry point with a guarded execution boundary."""
        store = _store()
        store.sync_if_stale()
        try:
            from llm_router import create_message_with_fallback

            response, route = create_message_with_fallback(
                system=NL2SQL_SYSTEM + f"\nRequested mode: {req.mode}",
                messages=[{"role": "user", "content": req.instruction}],
                max_tokens=900,
                model=os.getenv("NL2SQL_MODEL", "").strip() or None,
                timeout=float(os.getenv("NL2SQL_TIMEOUT_S", "8")),
                retry_attempts=1,
                queue_timeout=2,
            )
            generated = _parse_nl2sql_json(_extract_llm_text(response))
            sql = _validate_generated_sql(generated["sql"], req.mode)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"NL2SQL 生成或校验失败: {exc}") from exc

        started = time.perf_counter()
        if req.mode == "query":
            # Keep standard SQL expressiveness inside the subquery while the
            # service, not the model, owns the response-size budget.
            executed_sql = f"SELECT * FROM ({sql}) AS nl2sql_result LIMIT ?"
            executed_params = [*generated["params"], req.limit]
            rows = store.execute_sql(executed_sql, executed_params)
            data = {
                "mode": req.mode,
                "sql": sql,
                "executed_sql": executed_sql,
                "params": generated["params"],
                "explanation": generated["explanation"],
                "operation": generated["operation"],
                "rows": rows,
                "row_count": len(rows),
                "write_enabled": False,
            }
        else:
            request_id = store.save_change_request(
                req.instruction, sql, generated["params"], generated["explanation"]
            )
            data = {
                "mode": req.mode,
                "request_id": request_id,
                "sql": sql,
                "params": generated["params"],
                "explanation": generated["explanation"],
                "status": "pending_review",
                "write_enabled": False,
            }
        data["elapsed_s"] = round(time.perf_counter() - started, 4)
        data["llm_route"] = getattr(route, "name", "")
        data["source_of_truth"] = "retrieval_chunks.json + section_chunks.json + catalog.json"
        return {"code": 0, "msg": "standard NL2SQL generated", "data": data}

    return router

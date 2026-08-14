"""Isolated latency-optimized vnext benchmark API on loopback port 8013."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator


SERVICE_DIR = Path(__file__).resolve().parent
load_dotenv(SERVICE_DIR / ".env")

from experiments.minimal_context_experiment import run_case_minimal  # noqa: E402
from product_router import ProductRouter  # noqa: E402
from retrieval_engine import RetrievalEngine  # noqa: E402


MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"
REQUEST_TIMEOUT_SECONDS = 120

_engine: RetrievalEngine | None = None
_router: ProductRouter | None = None
_engine_lock = threading.Lock()
_warmup_lock = threading.Lock()
_retrieval_warmup: dict[str, Any] = {
    "status": "pending",
    "started_at": None,
    "completed_at": None,
    "seconds": None,
    "error": "",
}


def get_runtime() -> tuple[RetrievalEngine, ProductRouter]:
    global _engine, _router
    if _engine is not None and _router is not None:
        return _engine, _router
    with _engine_lock:
        if _engine is None:
            engine = RetrievalEngine()
            engine.ensure_index()
            _engine = engine
        if _router is None:
            _router = ProductRouter(_engine.catalog, engine=_engine)
    return _engine, _router


def _warm_retrieval_clients() -> None:
    """Prime Dense and rerank after index startup without delaying readiness."""

    started = time.perf_counter()
    with _warmup_lock:
        _retrieval_warmup.update({
            "status": "running",
            "started_at": time.time(),
            "completed_at": None,
            "seconds": None,
            "error": "",
        })
    try:
        engine, _ = get_runtime()
        product = next(iter(engine.catalog), None)
        if not product:
            raise RuntimeError("manual catalog is empty")
        # This intentionally exercises the same remote embedding and rerank
        # clients as a real question. The synthetic query is never exposed as
        # user history or an answer, and its cache entry is harmless.
        engine.search_manual(
            ["product", "manual"],
            semantic_query="product manual",
            original_query="product manual",
            top_k=1,
            products=[product],
        )
    except Exception as exc:  # noqa: BLE001
        with _warmup_lock:
            _retrieval_warmup.update({
                "status": "failed",
                "completed_at": time.time(),
                "seconds": round(time.perf_counter() - started, 3),
                "error": type(exc).__name__,
            })
        return
    with _warmup_lock:
        _retrieval_warmup.update({
            "status": "complete",
            "completed_at": time.time(),
            "seconds": round(time.perf_counter() - started, 3),
            "error": "",
        })


class FastChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    product: str | None = Field(default=None, max_length=160)
    use_history_context: bool = False
    history_context: str = Field(default="", max_length=2400)
    context_packet: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class FastChatResponse(BaseModel):
    answer: str
    product: str
    route: dict[str, Any]
    selected_ids: list[str]
    sources: list[dict[str, Any]]
    pictures: dict[str, Any]
    model: str
    reasoning_effort: str
    timings: dict[str, Any]
    total_elapsed: float
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)


class WebChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    forced_product: str | None = Field(default=None, max_length=160)
    images: list[Any] = Field(default_factory=list)
    session_id: str | None = Field(default=None, max_length=240)
    use_history_context: bool = False
    history_context: str = Field(default="", max_length=2400)
    context_packet: dict[str, Any] = Field(default_factory=dict)
    stream: bool = True

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


app = FastAPI(title="Manual Retrieval Vnext Fast (isolated)", version="0.1.0")


_ELLIPTICAL_QUERY_NOISE_RE = re.compile(
    r"(?:第[一二三四五六七八九十0-9]+步|步骤|接着|然后|最后|前面|刚才|上面|这个|那个|它|该|"
    r"是什么|怎么办|怎么做|如何|怎样|能否|可以|要不要|需要|多少|哪些|什么|呢|吗|啊|呀|"
    r"最多|是否|还要|接下来|之后|再|一下)",
    re.IGNORECASE,
)


def _is_elliptical_followup(question: str) -> bool:
    """Return true only when the current text lacks a usable subject of its own."""

    text = str(question or "").strip()
    if not text:
        return False
    residual = _ELLIPTICAL_QUERY_NOISE_RE.sub("", text)
    # A remaining multi-character Chinese term (or a non-generic English word)
    # is usually a concrete object/condition, so retrieval should trust the
    # current question rather than let earlier turns dominate ranking.
    if len(re.findall(r"[\u4e00-\u9fff]", residual)) >= 2:
        return False
    english = re.findall(r"[a-z]{3,}", residual.lower())
    return not any(word not in {"the", "and", "then", "what", "how", "does", "that", "this"} for word in english)


def _history_retrieval_query(question: str, history_context: str, context_packet: dict[str, Any]) -> str:
    """Use the bounded conversation subject to resolve an elliptical query."""

    if not _is_elliptical_followup(question):
        return question

    latest_user = ""
    turns = context_packet.get("recent_turns", []) if isinstance(context_packet, dict) else []
    if isinstance(turns, list):
        for turn in reversed(turns):
            if isinstance(turn, dict) and str(turn.get("role") or "").lower() == "user":
                latest_user = str(turn.get("content") or "").strip()
                if latest_user:
                    break
    history = str(history_context or "").strip()
    if not latest_user and not history:
        return question
    # The immediately preceding user message alone is often another pronoun
    # question ("what about long-term storage?"). Retain the bounded turn
    # summary as well so the original object—e.g. a remote-control battery—is
    # not lost after three or more follow-ups. The current question remains
    # last and is still the primary ranking intent.
    subject = history[:1400] if history else latest_user[:600]
    if latest_user and latest_user not in subject:
        subject = f"{subject}\n最近追问：{latest_user[:400]}"
    return f"对话主题（仅用于消解指代）：{subject}\n当前追问：{question}"


def _is_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _clarification_response(question: str, route: dict[str, Any]) -> FastChatResponse:
    chinese = _is_chinese(question)
    answer = (
        "我暂时无法确认您问的是哪一种产品，因此不能依据手册给出答案。请提供产品名称、型号，或上传一张清晰的产品/铭牌照片后再问。"
        if chinese else
        "I cannot identify which product this question is about, so I cannot answer from a manual. Please provide the product name, model number, or a clear photo of the product or rating label."
    )
    return FastChatResponse(
        answer=answer, product="待确认产品" if chinese else "Product to confirm", route=route,
        selected_ids=[], sources=[], pictures={"answer": []}, model=MODEL,
        reasoning_effort=REASONING_EFFORT, timings={"retrieval_seconds": 0, "generation_seconds": 0, "section_count": 0},
        total_elapsed=0.0, retrieval_trace={"status": "product_clarification", "route": route},
    )


def _no_evidence_response(question: str, product: str, route: dict[str, Any], result: dict[str, Any], elapsed: float) -> FastChatResponse:
    chinese = _is_chinese(question)
    answer = (
        f"已锁定“{product}”手册，但其中没有找到能够直接回答此问题的内容，因此我不能编造答案。请核对型号、换一种描述方式，或提供相关页面/产品照片。"
        if chinese else
        f"I identified the {product} manual, but it does not contain information that directly answers this question, so I cannot invent an answer. Please verify the model, rephrase the question, or provide the relevant page or product photo."
    )
    return FastChatResponse(
        answer=answer, product=product, route={**route, "reason": "no_direct_manual_evidence"},
        selected_ids=[], sources=[], pictures={"answer": []}, model=MODEL,
        reasoning_effort=REASONING_EFFORT, timings=dict(result.get("timings") or {}),
        total_elapsed=round(elapsed, 3), retrieval_trace={
            "status": "no_direct_manual_evidence",
            "selected_manual": product,
            "route": route,
        },
    )


@app.on_event("startup")
async def warm_runtime() -> None:
    await asyncio.to_thread(get_runtime)
    asyncio.create_task(asyncio.to_thread(_warm_retrieval_clients))


@app.get("/health")
async def health() -> dict[str, Any]:
    engine, _ = await asyncio.to_thread(get_runtime)
    with _warmup_lock:
        warmup = dict(_retrieval_warmup)
    return {
        "status": "ok",
        "service": "manual-retrieval-vnext-fast",
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "retrieval_chunks": len(engine.retrieval_chunks),
        "products": len(engine.catalog),
        "retrieval_warmup": warmup,
    }


def _run_request(
    payload: FastChatRequest,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    token_callback: Callable[[str], None] | None = None,
) -> FastChatResponse:
    engine, router = get_runtime()
    history_enabled = bool(payload.use_history_context)
    history_context = str(payload.history_context or "").strip() if history_enabled else ""
    context_packet = payload.context_packet if history_enabled and isinstance(payload.context_packet, dict) else {}
    retrieval_query = _history_retrieval_query(payload.question, history_context, context_packet)
    if payload.product:
        requested_product = str(payload.product).strip()
        if requested_product in engine.catalog:
            product = requested_product
            route_reason = "request_product"
        else:
            decision = router.route(
                f"{requested_product}\n{payload.question}",
                top_n=3,
            )
            if not decision.products:
                raise ValueError(f"unknown product: {requested_product}")
            product = decision.products[0]
            route_reason = "request_product_alias"
        route_payload: dict[str, Any] = {
            "confidence": "forced",
            "reason": route_reason,
            "requested": requested_product,
            "resolved": product,
        }
    else:
        decision = router.route(payload.question, top_n=3)
        if not decision.products or (decision.confidence != "high" and len(decision.products) > 1):
            route = {
                "confidence": decision.confidence,
                "reason": "product_not_clear",
                "candidates": decision.products,
                "candidate_scores": [
                    {"manual": name, "score": round(float(score), 4)}
                    for name, score in (decision.debug_scores or [])[:5]
                ],
            }
            return _clarification_response(payload.question, route)
        product = decision.products[0]
        route_payload = {
            "confidence": decision.confidence,
            "reason": decision.reason,
            "candidates": decision.products,
            "candidate_scores": [
                {"manual": name, "score": round(float(score), 4)}
                for name, score in (decision.debug_scores or [])[:5]
            ],
        }

    if progress_callback:
        progress_callback(
            "status",
            {
                "stage": "scope",
                "message": f"已锁定手册：{product}",
            },
        )
        progress_callback(
            "status",
            {
                "stage": "knowledge",
                "message": "正在执行 BM25、Dense、RRF 与重排",
            },
        )

    started = time.perf_counter()
    def on_case_progress(stage: str, data: dict[str, object]) -> None:
        if not progress_callback or stage != "retrieval_ready":
            return
        partial_result = {
            "selected": data.get("selected") or [],
            "retrieval_trace": data.get("retrieval_trace") or {},
            "timings": {
                "retrieval_seconds": data.get("retrieval_seconds") or 0,
                "generation_seconds": 0,
                "context_chars": data.get("context_chars") or 0,
            },
            "context_chars": data.get("context_chars") or 0,
        }
        progress_callback(
            "audit",
            {
                "retrieval_trace": _public_retrieval_trace(
                    result=partial_result,
                    product=product,
                    route=route_payload,
                    question=payload.question,
                    total_elapsed=time.perf_counter() - started,
                ),
            },
        )
        progress_callback(
            "status",
            {
                "stage": "model",
                "message": "召回与证据选择完成，正在生成答案",
            },
        )

    # The answer provider can occasionally return an empty/transient failure
    # after retrieval has completed. Retry that generation path once with the
    # exact same request and evidence policy; do not broaden retrieval or
    # silently substitute a different product.
    result: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            result = run_case_minimal(
                engine,
                payload.question,
                product,
                model=MODEL,
                reasoning_effort=REASONING_EFFORT,
                section_limit=5,
                history_context=history_context,
                retrieval_query=retrieval_query,
                progress_callback=on_case_progress,
                token_callback=token_callback,
            )
            if result.get("no_manual_evidence"):
                return _no_evidence_response(
                    payload.question, product, route_payload, result, time.perf_counter() - started,
                )
            if str(result.get("answer") or "").strip():
                break
            raise RuntimeError("fast planner returned an empty answer")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= 2:
                raise
            if progress_callback:
                progress_callback(
                    "status",
                    {"stage": "model", "message": "生成响应异常，正在使用同一证据重试一次"},
                )
    if result is None:
        raise last_error or RuntimeError("fast planner failed")
    total_elapsed = time.perf_counter() - started
    answer = str(result.get("answer") or "").strip()
    if not answer:
        raise RuntimeError("fast planner returned an empty answer")
    sources = _public_sources(result, product, answer)
    retrieval_trace = _public_retrieval_trace(
        result=result,
        product=product,
        route=route_payload,
        question=payload.question,
        total_elapsed=total_elapsed,
    )
    if progress_callback:
        progress_callback(
            "status",
            {
                "stage": "compose",
                "message": "答案生成完成，正在整理来源与图示",
            },
        )
    return FastChatResponse(
        answer=answer,
        product=product,
        route=route_payload,
        selected_ids=[
            str(item.get("unit_id"))
            for item in (result.get("selected") or [])
            if item.get("unit_id")
        ],
        sources=sources,
        pictures=dict(result.get("picture_validation") or {}),
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        timings=dict(result.get("timings") or {}),
        total_elapsed=round(total_elapsed, 3),
        retrieval_trace=retrieval_trace,
    )


def _clean_source_text(value: str) -> str:
    text = re.sub(
        r"\[FIGURE_CONTENT\s+[^\]]+\].*?\[/FIGURE_CONTENT\]",
        " ",
        str(value or ""),
        flags=re.DOTALL,
    )
    text = re.sub(r"\[\[PIC:[^\]]+\]\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_ANSWER_PICTURE_RE = re.compile(r"\[\[PIC:([^\]]+)\]\]")


def _compact_evidence_text(value: str) -> str:
    value = _ANSWER_PICTURE_RE.sub("", str(value or ""))
    return re.sub(r"[^0-9A-Za-z\u3400-\u9fff]+", "", value).casefold()


def _answer_source_fragment(source_text: str, answer: str) -> str:
    """Return only source lines demonstrably used by the final answer."""

    answer_pictures = set(_ANSWER_PICTURE_RE.findall(answer))
    answer_lines = [
        _compact_evidence_text(line)
        for line in str(answer or "").splitlines()
    ]
    answer_lines = [line for line in answer_lines if len(line) >= 8]
    lines = [line.rstrip() for line in str(source_text or "").splitlines()]
    kept_indexes: set[int] = set()
    for index, line in enumerate(lines):
        compact = _compact_evidence_text(line)
        has_used_picture = bool(answer_pictures & set(_ANSWER_PICTURE_RE.findall(line)))
        literal_used = len(compact) >= 8 and any(
            compact in answer_line or answer_line in compact
            for answer_line in answer_lines
        )
        if has_used_picture or literal_used:
            kept_indexes.add(index)
            # A source picture anchor commonly follows the sentence it proves.
            if has_used_picture and index > 0 and _compact_evidence_text(lines[index - 1]):
                kept_indexes.add(index - 1)
    return "\n".join(lines[index] for index in sorted(kept_indexes)).strip()


def _public_sources(result: dict[str, Any], product: str, answer: str) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for index, unit in enumerate(result.get("selected_units") or [], start=1):
        heading = str(unit.get("heading") or "").strip()
        content = _clean_source_text(
            _answer_source_fragment(str(unit.get("text") or ""), answer)
        )
        if not content:
            continue
        sources.append(
            {
                "rank": index,
                "chunk_id": str(unit.get("unit_id") or f"vnext-{index}"),
                "manual": str(unit.get("product") or product),
                "section": heading,
                "group_id": f"{product}\u0000{heading}",
                "page": None,
                "score": None,
                "excerpt": content[:900],
                "content": content[:12000],
                "evidence_role": "primary" if index == 1 else "ranked",
                "document_order": unit.get("document_order"),
                "primary_evidence": index == 1,
            }
        )
    return sources


def _public_retrieval_trace(
    *,
    result: dict[str, Any],
    product: str,
    route: dict[str, Any],
    question: str,
    total_elapsed: float,
) -> dict[str, Any]:
    """Build a browser-safe, factual audit trace.

    This intentionally contains rankings and short manual excerpts only.  It
    must never expose provider prompts, request headers, credentials, or full
    private conversation context.
    """
    raw = dict(result.get("retrieval_trace") or {})
    selected = []
    for index, item in enumerate(result.get("selected") or [], start=1):
        selected.append({
            "rank": index,
            "role": "primary" if index == 1 else "supporting",
            "chunk_id": f"section-{item.get('section_id')}",
            "heading": str(item.get("heading") or ""),
            "excerpt": _clean_source_text(str(item.get("text") or ""))[:500],
        })
    timings = dict(result.get("timings") or {})
    retrieval_seconds = float(timings.get("retrieval_seconds") or 0)
    generation_seconds = float(timings.get("generation_seconds") or 0)
    return {
        "version": 1,
        "route": {
            "selected_manual": product,
            "confidence": route.get("confidence"),
            "reason": route.get("reason"),
            "candidates": route.get("candidates") or ([product] if product else []),
            "candidate_scores": route.get("candidate_scores") or [],
        },
        "query": {
            "original": question,
            "sparse": (raw.get("queries") or {}).get("sparse", ""),
            "semantic": (raw.get("queries") or {}).get("semantic", ""),
        },
        "retrieval": {
            "pipeline": ["BM25", "Dense", "RRF", "Rerank"],
            "dense": raw.get("dense") or {},
            "candidate_count": len(raw.get("candidates") or []),
            "filtered_count": int(raw.get("filtered_count") or 0),
            "candidates": list(raw.get("candidates") or [])[:20],
            "rerank": raw.get("rerank") or {},
        },
        "evidence": {
            "selected": selected,
            "context_chars": int(result.get("context_chars") or 0),
            "section_count": len(selected),
        },
        "timings": {
            "retrieval_seconds": round(retrieval_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "other_seconds": round(max(0.0, total_elapsed - retrieval_seconds - generation_seconds), 3),
            "total_seconds": round(total_elapsed, 3),
        },
    }


def _web_answer(value: str) -> str:
    return re.sub(r"\[\[PIC:[^\]]+\]\]", "<PIC>", str(value or "")).strip()


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _web_done(response: FastChatResponse, session_id: str) -> dict[str, Any]:
    pics = [str(item) for item in (response.pictures.get("answer") or []) if str(item)]
    # Older gateway processes forward `sources` verbatim but may discard newly
    # introduced top-level fields.  Carry the same safe trace on the first
    # source as a backwards-compatible envelope; citation renderers ignore the
    # extra property and the browser can read it immediately without a gateway
    # restart.  No prompt, key, header or history data is included.
    sources = [dict(source) for source in response.sources]
    if sources:
        sources[0]["retrieval_trace"] = response.retrieval_trace
    refusal = str(response.route.get("reason") or "") in {
        "product_not_clear", "no_direct_manual_evidence",
    }
    return {
        "answer": _web_answer(response.answer),
        "product": response.product,
        "pics": pics,
        "image_descriptions": [],
        "route": "clarify" if refusal else "tech",
        "sources": sources,
        "session_id": session_id,
        "elapsed": response.total_elapsed,
        "model": response.model,
        "reasoning_effort": response.reasoning_effort,
        "retrieval_trace": response.retrieval_trace,
    }


@app.post("/vnext/chat", response_model=FastChatResponse)
async def vnext_chat(payload: FastChatRequest) -> FastChatResponse:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run_request, payload),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="fast vnext request timed out") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"fast vnext request failed: {type(exc).__name__}",
        ) from exc


@app.post("/chat")
async def web_chat(payload: WebChatRequest, request: Request):
    if payload.images:
        raise HTTPException(
            status_code=400,
            detail="vnext fast text endpoint does not accept images",
        )
    request_id = request.headers.get("X-Request-Id") or f"vnext_req_{uuid.uuid4()}"
    session_id = payload.session_id or f"vnext_session_{uuid.uuid4().hex[:12]}"

    async def execute(
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        token_callback: Callable[[str], None] | None = None,
    ) -> FastChatResponse:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _run_request,
                FastChatRequest(
                    question=payload.question,
                    product=payload.forced_product,
                    use_history_context=payload.use_history_context,
                    history_context=payload.history_context,
                    context_packet=payload.context_packet,
                ),
                progress_callback,
                token_callback,
            ),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    if not payload.stream:
        try:
            response = await execute()
            return _web_done(response, session_id)
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="fast vnext request timed out") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"fast vnext request failed: {type(exc).__name__}",
            ) from exc

    async def generate():
        yield _sse(
            "status",
            {
                "stage": "accepted",
                "message": "请求已接收",
                "request_id": request_id,
                "session_id": session_id,
            },
        )
        progress_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def stream_progress(event: str, data: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(progress_queue.put_nowait, (event, data))

        def stream_delta(text: str) -> None:
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                ("delta", {"text": text}),
            )

        task = asyncio.create_task(execute(stream_progress, stream_delta))
        try:
            while not task.done():
                try:
                    event, data = await asyncio.wait_for(progress_queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                yield _sse(event, data)
            while not progress_queue.empty():
                event, data = progress_queue.get_nowait()
                yield _sse(event, data)
            response = await task
        except asyncio.TimeoutError:
            yield _sse("error", {"message": "fast vnext request timed out"})
            return
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"message": f"fast vnext request failed: {type(exc).__name__}"})
            return
        yield _sse("done", _web_done(response, session_id))

    return StreamingResponse(generate(), media_type="text/event-stream")

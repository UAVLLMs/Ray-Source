"""Isolated HTTP wrapper for the vnext manual evidence planner.

This module is deliberately separate from the production ``api_server.py``.
It binds to loopback when launched by the vnext start script and does not
modify or import the production chat pipeline.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


SERVICE_DIR = Path(__file__).resolve().parent
load_dotenv(SERVICE_DIR / ".env")

from experiments.evidence_coverage_vnext import run_case  # noqa: E402
from product_router import ProductRouter  # noqa: E402
from retrieval_engine import RetrievalEngine  # noqa: E402


MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "low"
REQUEST_TIMEOUT_SECONDS = 180

_engine: RetrievalEngine | None = None
_router: ProductRouter | None = None
_engine_lock = threading.Lock()


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


class VnextChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    product: str | None = Field(default=None, max_length=160)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class VnextChatResponse(BaseModel):
    answer: str
    product: str
    route: dict[str, Any]
    selected_ids: list[str]
    pictures: dict[str, Any]
    model: str
    reasoning_effort: str
    model_elapsed: float
    total_elapsed: float


app = FastAPI(title="Manual Retrieval Vnext (isolated)", version="0.1.0")


@app.on_event("startup")
async def warm_runtime() -> None:
    await asyncio.to_thread(get_runtime)


@app.get("/health")
async def health() -> dict[str, Any]:
    engine, _ = await asyncio.to_thread(get_runtime)
    return {
        "status": "ok",
        "service": "manual-retrieval-vnext",
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "retrieval_chunks": len(engine.retrieval_chunks),
        "products": len(engine.catalog),
    }


def _run_request(payload: VnextChatRequest) -> VnextChatResponse:
    engine, router = get_runtime()
    route_payload: dict[str, Any]
    if payload.product:
        if payload.product not in engine.catalog:
            raise ValueError(f"unknown product: {payload.product}")
        product = payload.product
        route_payload = {"confidence": "forced", "reason": "request_product"}
    else:
        decision = router.route(payload.question, top_n=3)
        if not decision.products:
            raise ValueError("unable to route question to a manual")
        product = decision.products[0]
        route_payload = {
            "confidence": decision.confidence,
            "reason": decision.reason,
            "candidates": decision.products,
        }

    started = time.perf_counter()
    result = run_case(
        engine,
        {
            "id": "api-request",
            "question": payload.question,
            "product": product,
            "expected_terms": [],
        },
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
    )
    total_elapsed = time.perf_counter() - started
    answer = str(result.get("answer") or "").strip()
    if not answer:
        raise RuntimeError("vnext planner returned an empty answer")
    return VnextChatResponse(
        answer=answer,
        product=product,
        route=route_payload,
        selected_ids=[
            str(item.get("unit_id"))
            for item in (result.get("selected") or [])
            if item.get("unit_id")
        ],
        pictures=dict(result.get("picture_validation") or {}),
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        model_elapsed=float(result.get("model_elapsed") or 0.0),
        total_elapsed=round(total_elapsed, 3),
    )


@app.post("/vnext/chat", response_model=VnextChatResponse)
async def vnext_chat(payload: VnextChatRequest) -> VnextChatResponse:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run_request, payload),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="vnext request timed out") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"vnext request failed: {type(exc).__name__}") from exc

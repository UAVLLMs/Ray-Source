"""Run every browser recommendation through the real 3011 chat gateway.

This is an audit tool only.  It neither edits data nor calls the retrieval
service directly; every request has the same unscoped payload as the browser.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
import uuid
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit-recommended-questions-20260804.jsonl"


def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def parse_done(body: str) -> dict | None:
    for block in body.split("\n\n"):
        if not block.startswith("event: done\n"):
            continue
        data = "\n".join(
            line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")
        )
        return json.loads(data)
    return None


def main() -> None:
    env = load_env()
    catalog = json.loads((ROOT / "public" / "ragv6-ui" / "answers.json").read_text(encoding="utf-8"))
    items = catalog["items"]
    headers = {
        "Authorization": f"Bearer {env['RAGV6_API_TOKEN']}",
        "Content-Type": "application/json",
    }

    def one(item: dict) -> dict:
        payload = {
            "question": item["question"],
            "forced_product": None,
            "session_id": f"raysource_recommended_audit_{uuid.uuid4().hex}",
            "stream": False,
            "model": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "use_history_context": False,
            "history_context": "",
            "context_packet": {},
            "history_product": "",
            "memory_epoch": "audit",
        }
        started = time.perf_counter()
        try:
            response = requests.post(
                "http://127.0.0.1:3011/ragv6-api/chat",
                headers=headers,
                json=payload,
                timeout=120,
            )
            done = parse_done(response.text)
            if not done:
                return {
                    "id": item["id"], "expected_product": item["product"],
                    "question": item["question"], "status": "no_done",
                    "http_status": response.status_code,
                    "detail": response.text[:300],
                }
            answer = str(done.get("answer") or "")
            return {
                "id": item["id"], "expected_product": item["product"],
                "question": item["question"], "status": "ok" if answer.strip() else "empty_answer",
                "actual_product": done.get("product"),
                "manuals": done.get("manuals") or [],
                "source_manuals": sorted({str(s.get("manual") or "") for s in done.get("sources") or []}),
                "source_count": len(done.get("sources") or []),
                "answer_chars": len(answer),
                "elapsed": done.get("elapsed"),
                "wall_seconds": round(time.perf_counter() - started, 3),
                "answer_preview": answer[:500],
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "id": item["id"], "expected_product": item["product"],
                "question": item["question"], "status": "exception",
                "detail": f"{type(exc).__name__}: {exc}"[:400],
            }

    completed_ids: set[int] = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            try:
                completed_ids.add(int(json.loads(line)["id"]))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
    pending = [item for item in items if int(item["id"]) not in completed_ids]
    print(f"resume completed={len(completed_ids)} pending={len(pending)}", flush=True)
    started = time.perf_counter()
    with OUT.open("a", encoding="utf-8") as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(one, item) for item in pending]
            for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                stream.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                stream.flush()
                if number % 20 == 0:
                    print(f"progress={number}/{len(pending)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    print(f"complete={len(completed_ids) + len(pending)} elapsed={time.perf_counter()-started:.1f}s report={OUT}", flush=True)


if __name__ == "__main__":
    main()

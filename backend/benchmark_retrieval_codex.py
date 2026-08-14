from __future__ import annotations

import json
import time

from retrieval_engine import RetrievalEngine


CASES = [
    ("空调手册", "如何给空调遥控器安装电池？"),
    ("空气净化器手册", "如何安装空气净化器的脚轮？"),
    ("洗碗机手册", "首次使用时，如何将洗碗机连接到排水口？"),
    ("电钻手册", "我的DCB101型号电钻指示灯闪烁时，这些闪烁标识代表什么含义？"),
    ("人体工学椅手册", "椅子的扶手使用一段时间后为什么会松动？"),
]


def run_round(engine: RetrievalEngine, name: str) -> dict:
    rows = []
    started = time.perf_counter()
    for product, question in CASES:
        t0 = time.perf_counter()
        results, filtered = engine.search_manual(
            [question],
            semantic_query=question,
            original_query=question,
            top_k=5,
            products=[product],
        )
        rows.append(
            {
                "product": product,
                "question": question,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                "filtered": filtered,
                "top": [
                    {
                        "product": item.product,
                        "heading": item.heading,
                        "role": item.source.get("evidence_role"),
                    }
                    for item in results[:3]
                ],
            }
        )
    return {
        "round": name,
        "total_ms": round((time.perf_counter() - started) * 1000, 1),
        "rows": rows,
    }


def main() -> None:
    t0 = time.perf_counter()
    engine = RetrievalEngine()
    engine.ensure_index()
    load_ms = round((time.perf_counter() - t0) * 1000, 1)
    print(json.dumps({"load_ms": load_ms}, ensure_ascii=False), flush=True)
    print(json.dumps(run_round(engine, "first"), ensure_ascii=False), flush=True)
    print(json.dumps(run_round(engine, "repeat"), ensure_ascii=False), flush=True)
    if hasattr(engine, "cache_stats"):
        print(json.dumps({"cache": engine.cache_stats()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

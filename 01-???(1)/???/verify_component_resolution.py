"""Corpus-wide regression for component-only manual questions.

This verifier derives up to three product-specific heading phrases per manual,
then checks the real BM25 + Dense + RRF retrieval path without supplying a
product name.  It also rejects a dangerous confidence outcome: answering with
high confidence when the top manual differs from the only manual containing
the selected phrase.

Run from this directory:
    python verify_component_resolution.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from api_server import _answer_confidence_decision
from product_router import ProductRouter
from retrieval_engine import RetrievalEngine


ROOT = Path(__file__).resolve().parent
CHUNKS_FILE = ROOT / "data" / "retrieval_chunks.json"
GENERIC_TERMS = (
    "安全", "警告", "操作", "说明", "概述", "维护", "清洁", "保养", "安装", "使用",
    "准备", "故障", "问题", "设置", "功能", "产品", "信息", "注意事项", "规格", "概览",
    "overview", "operation", "safety", "warning", "maintenance", "cleaning", "installation",
    "specifications", "guide", "introduction",
)
GENERIC_KEYS = {re.sub(r"[\s\W_]+", "", term).lower() for term in GENERIC_TERMS}
GENERIC_PREFIXES = (
    "使用前", "使用", "操作", "安装", "维护", "清洁", "保养", "故障", "安全", "注意",
    "产品", "功能", "设置", "using", "operation", "maintenance", "cleaning", "safety",
)


def _normalise_phrase(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", value).lower()


def _question(phrase: str) -> str:
    return f"如何{phrase}" if re.search(r"[\u4e00-\u9fff]", phrase) else f"how to {phrase}"


def _is_specific_phrase(key: str) -> bool:
    """Keep a component/function phrase, not a generic chapter category."""
    if key in GENERIC_KEYS:
        return False
    for prefix in GENERIC_PREFIXES:
        prefix_key = _normalise_phrase(prefix)
        if key.startswith(prefix_key) and len(key[len(prefix_key):]) < 5:
            return False
    return True


def build_cases() -> list[tuple[str, str, str]]:
    payload = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("retrieval_chunks", [])
    products_by_phrase: dict[str, set[str]] = defaultdict(set)
    candidates: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)

    for row in rows:
        product = str(row.get("product") or "").strip()
        heading = str(row.get("heading") or "").strip()
        parts = [part.strip() for part in re.split(r"[/｜|>]", heading) if part.strip()]
        for phrase in (parts[-2:] if len(parts) > 1 else parts):
            key = _normalise_phrase(phrase)
            if product and 5 <= len(key) <= 26:
                products_by_phrase[key].add(product)
                candidates[product][key] = (phrase, heading)

    cases: list[tuple[str, str, str]] = []
    for product, phrases in candidates.items():
        viable = sorted(
            (
                (len(key), phrase, heading)
                for key, (phrase, heading) in phrases.items()
                if len(products_by_phrase[key]) == 1
                # A specific heading such as “功能调节（高度/后仰/按摩）”
                # remains eligible, while “使用前检查” is a generic category
                # and must not be treated as product-identifying evidence.
                and _is_specific_phrase(key)
            ),
            reverse=True,
        )
        if not viable:
            continue
        chosen: list[tuple[int, str, str]] = []
        for index in (0, len(viable) // 2, len(viable) - 1):
            candidate = viable[index]
            if candidate[1] not in {item[1] for item in chosen}:
                chosen.append(candidate)
        cases.extend((product, phrase, heading) for _, phrase, heading in chosen)
    return cases


def main() -> None:
    cases = build_cases()
    engine = RetrievalEngine()
    engine.ensure_index()
    router = ProductRouter(engine.catalog, engine=engine)
    retrieval_failures: list[dict[str, object]] = []
    unsafe_high_confidence: list[dict[str, object]] = []

    for expected_product, phrase, heading in cases:
        question = _question(phrase)
        route = router.route(question)
        scoped_products = route.products if route.confidence == "high" and len(route.products) == 1 else []
        results, _filtered = engine.search_manual(
            [question], semantic_query=question, original_query=question,
            top_k=3, products=scoped_products,
        )
        ranked_products = [str(result.product) for result in results]
        if expected_product not in ranked_products:
            retrieval_failures.append({
                "expected_product": expected_product,
                "phrase": phrase,
                "heading": heading,
                "ranked_products": ranked_products,
            })
            continue

        decision = _answer_confidence_decision(
            results,
            products=scoped_products,
            route_candidates=scoped_products or list(route.products),
            visual_trace={},
            reviewed_answer=False,
        )
        if decision["level"] == "high" and ranked_products[0] != expected_product:
            unsafe_high_confidence.append({
                "expected_product": expected_product,
                "phrase": phrase,
                "heading": heading,
                "ranked_products": ranked_products,
                "route_reason": route.reason,
                "route_candidates": route.products,
                "confidence": decision,
            })

    report = {
        "case_count": len(cases),
        "product_count": len({product for product, _, _ in cases}),
        "retrieval_failures": retrieval_failures,
        "unsafe_high_confidence": unsafe_high_confidence,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if retrieval_failures or unsafe_high_confidence:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

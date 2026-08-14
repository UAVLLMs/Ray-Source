"""Build the website's exact-match recommended-answer cache from review data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import openpyxl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = Path(r"D:\新下载\question_public (4).csv")
DEFAULT_ANSWERS = Path(r"C:\Users\lenovo\Downloads\qwen3max_gemini.csv")
DEFAULT_FINALS = Path(r"C:\Users\lenovo\Downloads\决赛50题打分明细(3).xlsx")
DEFAULT_CATALOG = ROOT / "public" / "ragv6-ui" / "answers.json"
DEFAULT_OUTPUT = ROOT / "data" / "recommended-answer-cache.json"
CUSTOMER_SERVICE_PRODUCT = "\u5ba2\u670d\u552e\u540e"
FINAL_RECOMMENDATION_PRODUCT = "\u51b3\u8d5b\u7cbe\u9009"


def normalize_question(value: object) -> str:
    """Match only a canonical rendering of the same submitted question."""
    text = str(value or "").strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).casefold()


def question_hash(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


def read_csv(path: Path, encoding: str) -> list[dict[str, str]]:
    with path.open("r", encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def read_final_rows(path: Path) -> list[dict[str, object]]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    rows: list[dict[str, object]] = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if not isinstance(row[0], int):
            continue
        rows.append({
            "id": row[0],
            "question_type": str(row[1] or ""),
            "question": str(row[2] or "").strip(),
            "answer": str(row[3] or "").strip(),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--finals", type=Path, default=DEFAULT_FINALS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-catalog", action="store_true")
    args = parser.parse_args()

    questions = read_csv(args.questions, "utf-8-sig")
    # qwen3max_gemini.csv is GBK-family encoded, not UTF-8.
    answers = read_csv(args.answers, "gb18030")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    catalog_by_id = {str(row.get("id")): row for row in catalog.get("items", [])}
    question_by_id = {str(row.get("id")): str(row.get("question") or "").strip() for row in questions}
    answer_by_id = {str(row.get("id")): str(row.get("ret") or "").strip() for row in answers}

    if not questions or not answers:
        raise ValueError("question and answer CSV files must not be empty")
    if set(question_by_id) != set(answer_by_id):
        raise ValueError("question and answer CSV ids do not match")
    if any(not value for value in question_by_id.values()) or any(not value for value in answer_by_id.values()):
        raise ValueError("question and answer CSV files must not contain empty values")

    entries: list[dict[str, object]] = []
    for source_id in sorted(question_by_id, key=lambda value: int(value)):
        catalog_row = catalog_by_id.get(source_id, {})
        entries.append({
            "cache_id": f"public-{source_id}",
            "question": question_by_id[source_id],
            "question_hash": question_hash(question_by_id[source_id]),
            "answer": answer_by_id[source_id],
            "product": str(catalog_row.get("product") or ""),
            "answer_mode": "customer" if catalog_row.get("product") == CUSTOMER_SERVICE_PRODUCT else "manual",
            "source": "question_public_qwen3max_gemini",
            "source_id": int(source_id),
        })

    finals = read_final_rows(args.finals)
    if len(finals) != 50:
        raise ValueError(f"expected 50 final rows, found {len(finals)}")
    if any(not row["question"] or not row["answer"] for row in finals):
        raise ValueError("final workbook contains an empty question or answer")
    for row in finals:
        is_customer = "customer" in str(row["question_type"]).lower() or "客服" in str(row["question_type"])
        entries.append({
            "cache_id": f"final-{row['id']}",
            "question": row["question"],
            "question_hash": question_hash(str(row["question"])),
            "answer": row["answer"],
            "product": CUSTOMER_SERVICE_PRODUCT if is_customer else FINAL_RECOMMENDATION_PRODUCT,
            "answer_mode": "customer" if is_customer else "manual",
            "question_type": row["question_type"],
            "source": "finals_50_reviewed",
            "source_id": row["id"],
        })

    by_key: dict[str, list[dict[str, object]]] = {}
    for entry in entries:
        by_key.setdefault(normalize_question(entry["question"]), []).append(entry)
    ambiguous = {
        key: rows for key, rows in by_key.items()
        if len({str(row["answer"]) for row in rows}) > 1
    }
    # Requests identify a recommendation by question text only. Retain source
    # order for the one repeated question so gateway behavior is deterministic.
    for rows in by_key.values():
        for entry in rows:
            entry["cache_eligible"] = True

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "match_policy": "exact_normalized_question_without_media_or_history",
        "stats": {
            "entries": len(entries),
            "public_questions": len(questions),
            "final_questions": len(finals),
            "unique_normalized_questions": len(by_key),
            "ambiguous_normalized_questions": len(ambiguous),
            "cache_eligible_entries": sum(1 for entry in entries if entry["cache_eligible"]),
            "by_source": dict(Counter(str(entry["source"]) for entry in entries)),
        },
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.write_catalog:
        # Drop a prior final-review import by exact question text. Public CSV
        # ids are not guaranteed to be a contiguous 1..400 range, so an id
        # cutoff would remove valid public recommendations.
        final_question_keys = {normalize_question(row["question"]) for row in finals}
        catalog_items = [
            row for row in catalog.get("items", [])
            if normalize_question(row.get("question")) not in final_question_keys
        ]
        catalog_items_by_id = {str(row.get("id")): row for row in catalog_items}
        for source_id, question in question_by_id.items():
            row = catalog_items_by_id.get(source_id)
            if not row:
                raise ValueError(f"catalog is missing public question id {source_id}")
            row["question"] = question
            row["answer"] = answer_by_id[source_id]
        next_id = max(int(row.get("id") or 0) for row in catalog_items) + 1
        for row in finals:
            is_customer = "customer" in str(row["question_type"]).lower() or "客服" in str(row["question_type"])
            catalog_items.append({
                "id": next_id,
                "product": CUSTOMER_SERVICE_PRODUCT if is_customer else FINAL_RECOMMENDATION_PRODUCT,
                "question": row["question"],
                "answer": row["answer"],
                "images": [],
                "score": None,
                "recommendation_source": "finals_50_reviewed",
            })
            next_id += 1
        product_counts = Counter(str(row.get("product") or "") for row in catalog_items)
        existing_order = [str(row.get("name") or "") for row in catalog.get("products", [])]
        product_order = [name for name in existing_order if name in product_counts]
        if FINAL_RECOMMENDATION_PRODUCT in product_counts and FINAL_RECOMMENDATION_PRODUCT not in product_order:
            product_order.append(FINAL_RECOMMENDATION_PRODUCT)
        catalog["items"] = catalog_items
        catalog["products"] = [
            {"name": name, "count": product_counts[name]}
            for name in product_order
        ]
        args.catalog.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()

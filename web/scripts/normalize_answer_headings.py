"""Audit and normalize fixed-answer headings without changing answer content.

The checked-in recommendation table contains OCR-flattened manual headings such
as ``# 程序显示 程序显示区...``.  A browser must never guess where that
heading ends: this tool inserts only line breaks at deterministic source
boundaries, yielding ``# 程序显示\n程序显示区...``.  It deliberately does not
rewrite body prose, image anchors, punctuation, or existing emphasis.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "public" / "ragv6-ui" / "answers.json"
CACHE = ROOT / "data" / "recommended-answer-cache.json"
MANUAL_PRODUCTS = {"决赛精选"}


@dataclass(frozen=True)
class Change:
    before: str
    after: str
    heading_count: int


def is_manual_row(row: dict[str, object]) -> bool:
    """Only normalize manual/reference answers; service prose is untouched."""
    return str(row.get("answer_mode") or "manual") == "manual" and str(row.get("product") or "") != "客服售后"


def title_boundary(fragment: str) -> int | None:
    """Return the insertion point between an OCR-flattened heading and body.

    Every accepted rule is based on a manual-style title/body relationship.  A
    missing match is intentionally left alone and reported for manual review.
    """
    line = fragment.split("\n", 1)[0]
    if not line or line.lstrip() != line:
        return None

    # The body repeats the title in labels such as “程序显示 程序显示区…”.
    repeated = re.match(r"^(.{2,36}?) (\1)(?:区|功能|模式|按钮|显示|装置|系统|器|表|灯|键|孔|盖|架|盒|门|盘|滤网|设置|步骤|方法|说明)", line)
    if repeated:
        return repeated.end(1)

    # “控制台功能 控制台可…” is another common manual/OCR form: the
    # noun before 功能/说明 is repeated as the grammatical subject.
    suffix_repeat = re.match(
        r"^(.{2,36}?(?:功能|说明|数据|操作|设置|安装|维护|清洁|警告|注意事项|概览|部件|规格|步骤|存放|模式|显示|运行|安全|准备|故障排除)) (.+)$",
        line,
    )
    if suffix_repeat:
        title = suffix_repeat.group(1)
        stem = re.sub(r"(?:功能|说明|数据|操作|设置|安装|维护|清洁|警告|注意事项|概览|部件|规格|步骤|存放|模式|显示|运行|安全|准备|故障排除)$", "", title)
        if stem and suffix_repeat.group(2).startswith(stem):
            return len(title)

    # Named manual sections are conventionally short noun phrases.  The
    # extracted source frequently loses the line break immediately after a
    # phrase ending in one of these section terms (for example “按键功能
    # 阻力增加…”, “排水连接 排水管…”, or “高度调节 向上拉起…”).
    named_section = re.match(
        r"^(.{2,48}?(?:"
        r"功能|说明|数据|操作|设置|安装|维护|清洁|保养|调节|调整|运行|模式|显示|按钮|部件|组件|装备|安全|警告|注意事项|概览|规格|步骤|存放|连接|排水|洗涤剂|洗涤块|亮碟剂|滤网|系统|程序|电池|温度|餐具|物品|建议|高度|停机|介绍|使用|检查|更换|拆卸|组装|充电|开机|关机|故障排除"
        r")) (?=\S)",
        line,
    )
    if named_section:
        return named_section.end(1)

    # A short documented title followed by an imperative, reference pronoun,
    # numbered step, or image anchor.  The non-greedy prefix preserves the
    # original title exactly and only inserts a line break after it.
    starter = r"(?:本|该|此|这|通过|使用|按(?:下)?|请|将|可|需|为|在|从|要|如|若|当|对于|以下|图|[0-9]+[.、]|[-•*]|<PIC>)"
    direct = re.match(rf"^(.{{2,44}}?) (?={starter})", line)
    if direct:
        return direct.end(1)

    # English all-caps / title-style section names are already unambiguous
    # when followed by an ordinary sentence.
    english = re.match(r"^([A-Z][A-Za-z0-9 /&'()_-]{2,64}) (?=(?:The|This|Your|To|When|If|Before|After|Use|Check|Press|Remove|Install|Do|Never|Push|As|One|Severe|Risk|<PIC>|[0-9]+\.))", line)
    if english:
        return english.end(1)
    return None


def normalize_answer(answer: str) -> Change:
    """Insert newlines around proven flattened `#` headings, never edit prose."""
    matches = list(re.finditer(r"(?<!#)(#{1,6}) +", answer))
    inserts: set[int] = set()
    headings = 0
    for index, marker in enumerate(matches):
        # Make an embedded marker a standalone line.  Existing whitespace is
        # retained; this is a pure insertion, so text/audit anchors are stable.
        if marker.start() > 0 and answer[marker.start() - 1] != "\n":
            inserts.add(marker.start())
        end = matches[index + 1].start() if index + 1 < len(matches) else len(answer)
        boundary = title_boundary(answer[marker.end():end])
        if boundary is not None:
            absolute = marker.end() + boundary
            if absolute < len(answer) and answer[absolute] != "\n":
                inserts.add(absolute)
                headings += 1
    if not inserts:
        return Change(answer, answer, 0)
    normalized = "".join(
        ("\n" if position in inserts else "") + char
        for position, char in enumerate(answer)
    )
    return Change(answer, normalized, headings)


def audit_file(path: Path, collection_key: str) -> tuple[dict[str, object], list[tuple[int, Change]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload[collection_key]
    changes: list[tuple[int, Change]] = []
    for row_index, row in enumerate(rows):
        if not is_manual_row(row):
            continue
        result = normalize_answer(str(row.get("answer") or ""))
        if result.after != result.before:
            changes.append((row_index, result))
    return payload, changes


def print_report(label: str, payload: dict[str, object], rows: list[tuple[int, Change]], key: str) -> None:
    records = payload[key]
    print(f"{label}: total={len(records)} changed={len(rows)} title_boundaries={sum(c.heading_count for _, c in rows)}")
    for row_index, change in rows:
        row = records[row_index]
        identifier = row.get("cache_id", row.get("id", row_index))
        print(f"  {identifier}: inserted_newlines={len(change.after) - len(change.before)} title_boundaries={change.heading_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-json", action="store_true", help="write transformed payloads as JSON to stdout")
    args = parser.parse_args()
    catalog, catalog_changes = audit_file(CATALOG, "items")
    cache, cache_changes = audit_file(CACHE, "entries")
    if args.emit_json:
        for index, change in catalog_changes:
            catalog["items"][index]["answer"] = change.after
        for index, change in cache_changes:
            cache["entries"][index]["answer"] = change.after
        print(json.dumps({"catalog": catalog, "cache": cache}, ensure_ascii=False, indent=2))
        return
    print_report("answers.json", catalog, catalog_changes, "items")
    print_report("recommended-answer-cache.json", cache, cache_changes, "entries")


if __name__ == "__main__":
    main()

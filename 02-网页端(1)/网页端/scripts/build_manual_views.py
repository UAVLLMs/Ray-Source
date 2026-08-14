"""Build lightweight per-manual pages for fast RAG source navigation.

The original directory is a single large HTML document. Source links normally
need only one manual, so this generator keeps the original book DOM intact but
writes one small page per manual and a safe alias manifest for the Node gateway.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "public" / "ragv6-manual-index" / "manual-index-source.html"
OUTPUT = ROOT / "public" / "ragv6-manual-index" / "manuals"
MANIFEST = OUTPUT / "manifest.json"


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def clean_math_markup(value: str) -> str:
    """Turn common OCR/LaTeX fragments into readable manual text."""
    if not value or "$" not in value:
        return value

    def expression(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        # The source commonly nests a command inside a formatting command,
        # for example \pmb{\bigtriangleup}; unwrap both levels before mapping.
        for _ in range(3):
            unwrapped = re.sub(
                r"\\(?:pmb|mathbf|mathsf|mathrm|text|operatorname)\s*\{((?:[^{}]|\{[^{}]*\})*)\}",
                r"\1",
                expr,
            )
            if unwrapped == expr:
                break
            expr = unwrapped
        symbols = {
            r"\bigtriangleup": "▲", r"\bigtriangledown": "▼",
            r"\triangleup": "▲", r"\triangledown": "▼",
            r"\triangle": "△", r"\blacktriangleleft": "◀",
            r"\blacktriangleright": "▶", r"\leftrightarrow": "↔",
            r"\leftarrow": "←", r"\rightarrow": "→",
            r"\circ": "°", r"\star": "*", r"\times": "×",
            r"\,": " ", r"~": " ",
        }
        for source, target in symbols.items():
            expr = expr.replace(source, target)
        expr = re.sub(r"\^\{([^{}]+)\}", r"^\1", expr)
        expr = re.sub(r"_\{([^{}]+)\}", r"_\1", expr)
        expr = re.sub(r"\\[a-zA-Z]+", "", expr)
        expr = re.sub(r"[{}]", "", expr)
        return re.sub(r"\s+", " ", expr).strip()

    return re.sub(r"\$([^$]+)\$", expression, value)


def clean_tree_markup(root: BeautifulSoup) -> None:
    for node in root.find_all(string=True):
        if node.parent and node.parent.name in {"script", "style"}:
            continue
        cleaned = clean_math_markup(str(node))
        if cleaned != str(node):
            node.replace_with(cleaned)


def main() -> None:
    source = BeautifulSoup(SOURCE.read_text(encoding="utf-8"), "html.parser")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    existing_manuals = []
    if MANIFEST.exists():
        try:
            existing_manuals = json.loads(MANIFEST.read_text(encoding="utf-8")).get("manuals", [])
        except (json.JSONDecodeError, OSError, AttributeError):
            existing_manuals = []

    manuals = []
    for index, book in enumerate(source.select("main.wrap > .bk"), start=1):
        title = clean_text(book.select_one(".bt1").get_text(" ", strip=True) if book.select_one(".bt1") else "")
        translated = clean_text(book.select_one(".tr").get_text(" ", strip=True) if book.select_one(".tr") else "")
        meta = clean_text(book.select_one(".bf").get_text(" ", strip=True) if book.select_one(".bf") else "")
        if not title:
            continue

        filename = f"manual-{index:02d}.html"
        page = BeautifulSoup("<!doctype html><html lang='zh-CN'><head></head><body></body></html>", "html.parser")
        page.head.append(copy.copy(source.head.title) if source.head and source.head.title else page.new_tag("title"))
        if not page.head.title.string:
            page.head.title.string = title
        for style in source.head.find_all("style") if source.head else []:
            page.head.append(copy.copy(style))

        top = page.new_tag("header", attrs={"class": "top"})
        top_inner = page.new_tag("div", attrs={"class": "in"})
        heading = page.new_tag("strong")
        heading.string = title
        top_inner.append(heading)
        if translated:
            subtitle = page.new_tag("span")
            subtitle.string = f"  {translated}"
            top_inner.append(subtitle)
        full = page.new_tag("a", href="/rag/manual-index/")
        full.string = "Open full directory"
        full["style"] = "margin-left:auto;color:#fff;text-decoration:none;font-size:12px"
        top_inner.append(full)
        top.append(top_inner)
        page.body.append(top)

        wrapper = page.new_tag("main", attrs={"class": "wrap"})
        # Reparse the fragment so text-node cleanup cannot mutate or miss the
        # source tree through BeautifulSoup's shallow-copy internals.
        cloned_book = BeautifulSoup(str(book), "html.parser").find("section")
        cloned_book["class"] = list(set(cloned_book.get("class", [])) | {"open"})
        clean_tree_markup(cloned_book)
        wrapper.append(cloned_book)
        page.body.append(wrapper)
        page.body.append(page.new_tag("script", src="/rag/manual-index/navigator.js"))
        page.body.append(page.new_tag("script", src="/rag/manual-view/progressive-loader.js"))
        page.body.append(page.new_tag("script"))
        page.body.contents[-1].string = """
document.querySelectorAll('.bh,.hd').forEach((node) => node.addEventListener('click', () => {
  const owner = node.closest('.bk,.nd'); if (owner) owner.classList.toggle('open');
}));
"""
        (OUTPUT / filename).write_text(str(page), encoding="utf-8")
        manuals.append({"index": index, "file": filename, "title": title, "aliases": [value for value in (title, translated, meta) if value]})

    generated_files = {manual["file"] for manual in manuals}
    for manual in existing_manuals:
        filename = str(manual.get("file", ""))
        if (
            re.fullmatch(r"manual-\d{2}\.html", filename)
            and filename not in generated_files
            and (OUTPUT / filename).exists()
        ):
            manuals.append(manual)

    manuals.sort(key=lambda item: int(item.get("index", 0)))
    retained_files = {manual["file"] for manual in manuals}
    for old_page in OUTPUT.glob("manual-*.html"):
        if old_page.name not in retained_files:
            old_page.unlink()

    # The current source can be a directory shell while the full manual pages
    # are retained from the previous build. Sanitize those pages too, so a
    # display fix is applied consistently to both newly generated and retained
    # manuals.
    for page_file in OUTPUT.glob("manual-*.html"):
        page = BeautifulSoup(page_file.read_text(encoding="utf-8"), "html.parser")
        clean_tree_markup(page)
        page_file.write_text(str(page), encoding="utf-8")

    # The public directory route serves the full index directly, so sanitize
    # those entry pages as well as the per-manual views.
    for page_file in (ROOT / "public" / "ragv6-manual-index").glob("*.html"):
        page = BeautifulSoup(page_file.read_text(encoding="utf-8"), "html.parser")
        clean_tree_markup(page)
        page_file.write_text(str(page), encoding="utf-8")

    MANIFEST.write_text(json.dumps({"manuals": manuals}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manuals": len(manuals), "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

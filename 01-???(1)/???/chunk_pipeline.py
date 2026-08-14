"""RAGV6 新手册切分、质量检查、发布与回滚流水线。

该模块只负责把一本新手册转换为 RAGV6 已有的数据结构：

* ``手册_v4/<产品>.md``：可追溯的规范化源文档；
* ``data/manual_sections/<产品>.json``：面向人工审阅的章节文档；
* ``data/section_chunks.json``：完整父章节证据；
* ``data/retrieval_chunks.json``：用于 BM25/FAISS 召回的定位块；
* ``data/catalog.json``：产品、章节、字符数与标签目录；
* ``data/section_summaries.json``：与父章节绑定的短摘要。

设计原则是“先预览、后发布、发布必备份、索引显式重建”。切分阶段不调用
大模型，因此在隔离环境也能复现；语义边界由标题层级、段落、句子、步骤、
警告块和图片锚点共同判断。向量索引仍复用 ``RetrievalEngine.build_index``。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Any, Iterable
import zipfile
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MANUAL_SOURCE_DIR = ROOT / "手册_v4"
MANUAL_SECTIONS_DIR = DATA_DIR / "manual_sections"
ADMIN_DIR = DATA_DIR / "chunk-admin"
BACKUP_DIR = ADMIN_DIR / "backups"
STATE_PATH = ADMIN_DIR / "state.json"

SECTION_CHUNKS_PATH = DATA_DIR / "section_chunks.json"
RETRIEVAL_CHUNKS_PATH = DATA_DIR / "retrieval_chunks.json"
CATALOG_PATH = DATA_DIR / "catalog.json"
SECTION_SUMMARIES_PATH = DATA_DIR / "section_summaries.json"

_PUBLISH_LOCK = threading.RLock()
_MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<url>[^)]+)\)")
_PIC_ANCHOR_RE = re.compile(r"\[\[PIC:(?P<id>[^\]]+)\]\]", re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:第?[一二三四五六七八九十百\d]+[章节部分篇]|"
    r"\d+(?:\.\d+){0,3}[、.)．]\s*)\s*(?P<title>.+?)\s*$",
    re.IGNORECASE,
)
_STEP_LINE_RE = re.compile(
    r"^\s*(?:步骤\s*)?(?:\d{1,3}[.)、．:]|[A-Za-z][.)、．:]|"
    r"[①②③④⑤⑥⑦⑧⑨⑩]|[-*•])\s*",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？；!?;])\s+|(?<=[.!?])\s+(?=[A-Z0-9])")
_SPACE_RE = re.compile(r"[ \t]+")
_SAFE_NAME_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


@dataclass(slots=True)
class SplitOptions:
    """可调切分参数。

    ``target_chars`` 是希望的检索块长度，``max_chars`` 是硬上限；
    ``overlap_chars`` 只复制完整句子/步骤，不会从句子中间截断。
    """

    target_chars: int = 720
    min_chars: int = 160
    max_chars: int = 1100
    overlap_chars: int = 90
    infer_headings: bool = True
    drop_table_of_contents: bool = True

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "SplitOptions":
        raw = value or {}
        default_target = _clamp_int(os.getenv("CHUNK_TARGET_CHARS"), 720, 240, 2200)
        default_minimum = _clamp_int(os.getenv("CHUNK_MIN_CHARS"), 160, 60, default_target)
        default_maximum = _clamp_int(os.getenv("CHUNK_MAX_CHARS"), 1100, default_target, 4000)
        default_overlap = _clamp_int(
            os.getenv("CHUNK_OVERLAP_CHARS"),
            90,
            0,
            min(400, default_target // 2),
        )
        target = _clamp_int(raw.get("target_chars"), default_target, 240, 2200)
        minimum = _clamp_int(raw.get("min_chars"), min(default_minimum, target), 60, target)
        maximum = _clamp_int(raw.get("max_chars"), max(default_maximum, target), target, 4000)
        overlap = _clamp_int(
            raw.get("overlap_chars"),
            min(default_overlap, target // 2),
            0,
            min(400, target // 2),
        )
        return cls(
            target_chars=target,
            min_chars=minimum,
            max_chars=maximum,
            overlap_chars=overlap,
            infer_headings=bool(raw.get("infer_headings", True)),
            drop_table_of_contents=bool(raw.get("drop_table_of_contents", True)),
        )


@dataclass(slots=True)
class ImageAnchor:
    pic_id: str
    char_offset: int
    source: str = ""
    caption: str = ""


@dataclass(slots=True)
class ParsedSection:
    heading_path: list[str]
    text: str
    pics: list[str] = field(default_factory=list)
    pic_captions: list[str] = field(default_factory=list)
    image_anchors: list[ImageAnchor] = field(default_factory=list)
    source_start_line: int = 0
    source_end_line: int = 0

    @property
    def heading(self) -> str:
        return " / ".join(part for part in self.heading_path if part)


@dataclass(slots=True)
class TextUnit:
    text: str
    start: int
    end: int


@dataclass(slots=True)
class ChunkSpan:
    text: str
    start: int
    end: int
    split_kind: str


class ChunkPipelineError(ValueError):
    """用户输入、文件格式或发布前校验错误。"""


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_manual_name(value: str) -> str:
    name = _SAFE_NAME_RE.sub("_", str(value or "").strip())
    name = re.sub(r"\s+", " ", name).strip(" ._")
    if not name:
        raise ChunkPipelineError("手册名称不能为空")
    if len(name) > 96:
        name = name[:96].rstrip()
    if name in {".", ".."}:
        raise ChunkPipelineError("手册名称无效")
    return name


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChunkPipelineError(f"无法读取 {path.name}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def decode_uploaded_source(
    *,
    filename: str,
    content_base64: str | None = None,
    text: str | None = None,
) -> str:
    """把浏览器上传内容统一成 UTF-8 文本。

    Markdown/TXT 直接解码；DOCX 使用标准库读取 ``word/document.xml``；
    PDF 在环境安装 ``pypdf`` 时启用。未知文本编码会依次尝试 UTF-8、GB18030。
    """

    if text is not None and str(text).strip():
        return normalize_source_text(str(text))
    if not content_base64:
        raise ChunkPipelineError("请上传文件或提供手册正文")
    try:
        raw = base64.b64decode(content_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ChunkPipelineError("文件 Base64 编码无效") from exc
    if len(raw) > 25 * 1024 * 1024:
        raise ChunkPipelineError("单个手册文件不能超过 25MB")

    suffix = Path(filename or "manual.txt").suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return normalize_source_text(_decode_text_bytes(raw))
    if suffix == ".docx":
        return normalize_source_text(_decode_docx(raw))
    if suffix == ".pdf":
        return normalize_source_text(_decode_pdf(raw))
    raise ChunkPipelineError("当前支持 .md、.txt、.docx 和 .pdf 手册")


def _decode_text_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _decode_docx(raw: bytes) -> str:
    try:
        with tempfile.TemporaryFile() as fp:
            fp.write(raw)
            fp.seek(0)
            with zipfile.ZipFile(fp) as archive:
                xml_payload = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise ChunkPipelineError("DOCX 文件损坏或缺少正文") from exc

    root = ElementTree.fromstring(xml_payload)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    lines: list[str] = []
    for paragraph in root.iter(namespace + "p"):
        fragments = [node.text or "" for node in paragraph.iter(namespace + "t")]
        value = "".join(fragments).strip()
        if value:
            lines.append(value)
    if not lines:
        raise ChunkPipelineError("DOCX 中没有可提取的文本")
    return "\n\n".join(lines)


def _decode_pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise ChunkPipelineError("PDF 导入需要安装 pypdf；Markdown/TXT/DOCX 可直接使用") from exc
    try:
        with tempfile.TemporaryFile() as fp:
            fp.write(raw)
            fp.seek(0)
            reader = PdfReader(fp)
            pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:  # noqa: BLE001
        raise ChunkPipelineError(f"PDF 文本提取失败: {exc}") from exc
    text = "\n\n".join(page for page in pages if page)
    if not text:
        raise ChunkPipelineError("PDF 没有可提取文字；扫描版 PDF 请先执行 OCR")
    return text


def normalize_source_text(text: str) -> str:
    value = html.unescape(str(text or ""))
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\u0000", "")
    value = "\n".join(_SPACE_RE.sub(" ", line).rstrip() for line in value.splitlines())
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip()


def detect_language(text: str) -> str:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "zh" if cjk >= max(8, latin * 0.18) else "en"


def _clean_heading(value: str) -> str:
    text = _MARKDOWN_IMAGE_RE.sub("", value)
    text = _PIC_ANCHOR_RE.sub("", text)
    text = re.sub(r"[*_`~]+", "", text)
    return re.sub(r"\s+", " ", text).strip(" #\t")


def _looks_like_plain_heading(line: str, previous_blank: bool, next_blank: bool) -> bool:
    value = line.strip()
    if not value or len(value) > 80:
        return False
    if _STEP_LINE_RE.match(value):
        return False
    if _NUMBERED_HEADING_RE.match(value):
        return True
    if not (previous_blank or next_blank):
        return False
    if value.endswith(("。", "！", "？", ".", "!", "?", "；", ";", "：", ":")):
        return False
    words = value.split()
    if 1 <= len(words) <= 9 and value.isupper() and any(ch.isalpha() for ch in value):
        return True
    return bool(
        re.search(
            r"(概述|简介|说明|安装|操作|使用|维护|清洁|故障|故障排除|规格|安全|警告|"
            r"保修|附录|准备|连接|设置|overview|introduction|installation|"
            r"operation|maintenance|cleaning|troubleshooting|specifications|warning)$",
            value,
            re.IGNORECASE,
        )
    )


def infer_markdown_headings(source: str, manual_name: str) -> str:
    """为无 Markdown 标题的纯文本补充保守的二级标题。

    只提升“短行 + 独立成段 + 标题词/编号”行，避免把操作步骤误判为章节。
    """

    if any(_HEADING_RE.match(line) for line in source.splitlines()):
        return source
    lines = source.splitlines()
    result = [f"# {manual_name}", ""]
    for index, line in enumerate(lines):
        previous_blank = index == 0 or not lines[index - 1].strip()
        next_blank = index + 1 >= len(lines) or not lines[index + 1].strip()
        if _looks_like_plain_heading(line, previous_blank, next_blank):
            result.extend([f"## {_clean_heading(line)}", ""])
        else:
            result.append(line)
    return "\n".join(result)


def _image_id(alt: str, url: str) -> str:
    candidate = (alt or "").strip()
    if not candidate:
        candidate = Path(url.split("?", 1)[0]).stem
    candidate = _SAFE_NAME_RE.sub("_", candidate).strip(" ._")
    return candidate[:128] or f"image_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:10]}"


def _extract_images(text: str) -> tuple[str, list[ImageAnchor]]:
    """移除图片语法，同时记录图片在纯文本中的近似字符位置。"""

    anchors: list[ImageAnchor] = []
    output: list[str] = []
    cursor = 0
    clean_length = 0
    pattern = re.compile(
        rf"{_MARKDOWN_IMAGE_RE.pattern}|{_PIC_ANCHOR_RE.pattern}",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        prefix = text[cursor : match.start()]
        output.append(prefix)
        clean_length += len(prefix)
        groups = match.groupdict()
        if groups.get("id"):
            pic_id = safe_manual_name(groups["id"])
            source = ""
            caption = pic_id
        else:
            source = str(groups.get("url") or "")
            caption = str(groups.get("alt") or "")
            pic_id = _image_id(caption, source)
        anchors.append(
            ImageAnchor(
                pic_id=pic_id,
                char_offset=clean_length,
                source=source,
                caption=caption or pic_id,
            )
        )
        cursor = match.end()
    suffix = text[cursor:]
    output.append(suffix)
    clean = "".join(output)
    clean = re.sub(r"[ \t]+\n", "\n", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, anchors


def parse_sections(
    manual_name: str,
    source: str,
    options: SplitOptions | None = None,
) -> tuple[str, list[ParsedSection]]:
    """按 Markdown 标题树或纯文本推断标题生成父章节。"""

    opts = options or SplitOptions()
    normalized = normalize_source_text(source)
    if opts.infer_headings:
        normalized = infer_markdown_headings(normalized, manual_name)
    lines = normalized.splitlines()
    heading_stack: list[str] = []
    document_title = manual_name
    body: list[str] = []
    body_start = 1
    sections: list[ParsedSection] = []
    pending_root_images: list[ImageAnchor] = []

    def flush(end_line: int) -> None:
        nonlocal body, body_start, pending_root_images
        raw = "\n".join(body).strip()
        body = []
        if not raw:
            return
        clean, anchors = _extract_images(raw)
        if not clean and not anchors:
            return
        path = [part for part in heading_stack if part and part != document_title]
        if not path:
            path = ["概述" if detect_language(clean) == "zh" else "Overview"]
        if opts.drop_table_of_contents and _is_table_of_contents(path, clean):
            return
        if pending_root_images:
            anchors = [*pending_root_images, *anchors]
            pending_root_images = []
        sections.append(
            ParsedSection(
                heading_path=path,
                text=clean,
                pics=_dedupe(anchor.pic_id for anchor in anchors),
                pic_captions=[
                    anchor.caption
                    for anchor in _dedupe_anchors(anchors)
                ],
                image_anchors=_dedupe_anchors(anchors),
                source_start_line=body_start,
                source_end_line=max(body_start, end_line),
            )
        )

    for index, line in enumerate(lines, start=1):
        match = _HEADING_RE.match(line)
        if not match:
            body.append(line)
            continue
        flush(index - 1)
        level = len(match.group(1))
        raw_heading = match.group(2)
        heading_clean, heading_images = _extract_images(raw_heading)
        title = _clean_heading(heading_clean)
        if level == 1 and not heading_stack:
            document_title = title or manual_name
            heading_stack = [document_title]
            pending_root_images.extend(heading_images)
        else:
            while len(heading_stack) >= level:
                heading_stack.pop()
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(title or f"Section {index}")
            pending_root_images.extend(heading_images)
        body_start = index + 1
    flush(len(lines))

    if not sections:
        clean, anchors = _extract_images(normalized)
        if clean:
            sections.append(
                ParsedSection(
                    heading_path=["概述" if detect_language(clean) == "zh" else "Overview"],
                    text=clean,
                    pics=_dedupe(anchor.pic_id for anchor in anchors),
                    pic_captions=[anchor.caption for anchor in _dedupe_anchors(anchors)],
                    image_anchors=_dedupe_anchors(anchors),
                    source_start_line=1,
                    source_end_line=len(lines),
                )
            )
    if not sections:
        raise ChunkPipelineError("没有从手册中识别出可用正文")
    return document_title or manual_name, sections


def _is_table_of_contents(path: list[str], text: str) -> bool:
    heading = " ".join(path).lower()
    if any(token in heading for token in ("table of contents", "目录", "contents")):
        return len(text) < 3500
    return False


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _dedupe_anchors(values: Iterable[ImageAnchor]) -> list[ImageAnchor]:
    result: list[ImageAnchor] = []
    seen: set[tuple[str, int]] = set()
    for value in values:
        key = (value.pic_id, value.char_offset)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _logical_units(text: str, max_chars: int) -> list[TextUnit]:
    """生成不会破坏步骤/句子的最小切分单元。"""

    units: list[TextUnit] = []
    paragraph_matches = list(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.DOTALL))
    for paragraph_match in paragraph_matches:
        paragraph = paragraph_match.group(0).strip()
        if not paragraph:
            continue
        paragraph_start = paragraph_match.start() + (
            len(paragraph_match.group(0)) - len(paragraph_match.group(0).lstrip())
        )
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) > 1 and sum(bool(_STEP_LINE_RE.match(line)) for line in lines) >= 2:
            search_from = paragraph_start
            for line in lines:
                position = text.find(line, search_from)
                position = position if position >= 0 else search_from
                units.extend(_split_oversized_unit(line, position, max_chars))
                search_from = position + len(line)
            continue
        units.extend(_split_oversized_unit(paragraph, paragraph_start, max_chars))
    if not units and text.strip():
        start = text.find(text.strip())
        units.extend(_split_oversized_unit(text.strip(), max(0, start), max_chars))
    return units


def _split_oversized_unit(value: str, start: int, max_chars: int) -> list[TextUnit]:
    if len(value) <= max_chars:
        return [TextUnit(value, start, start + len(value))]
    parts: list[TextUnit] = []
    cursor = 0
    sentences = [part for part in _SENTENCE_BOUNDARY_RE.split(value) if part.strip()]
    if len(sentences) <= 1:
        sentences = re.findall(r".{1,%d}(?:\s+|$)" % max_chars, value, re.DOTALL)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        found = value.find(sentence, cursor)
        found = found if found >= 0 else cursor
        if len(sentence) <= max_chars:
            parts.append(TextUnit(sentence, start + found, start + found + len(sentence)))
        else:
            for offset in range(0, len(sentence), max_chars):
                segment = sentence[offset : offset + max_chars].strip()
                if segment:
                    local = found + offset
                    parts.append(TextUnit(segment, start + local, start + local + len(segment)))
        cursor = found + len(sentence)
    return parts


def split_section_text(text: str, options: SplitOptions) -> list[ChunkSpan]:
    units = _logical_units(text, options.max_chars)
    if not units:
        return []
    if len(text) <= options.max_chars:
        return [ChunkSpan(text=text.strip(), start=0, end=len(text), split_kind="whole")]

    groups: list[list[TextUnit]] = []
    current: list[TextUnit] = []
    current_length = 0
    for unit in units:
        separator = 2 if current else 0
        projected = current_length + separator + len(unit.text)
        if current and projected > options.max_chars and current_length >= options.min_chars:
            groups.append(current)
            overlap_units: list[TextUnit] = []
            overlap_length = 0
            for previous in reversed(current):
                projected_overlap = overlap_length + len(previous.text) + (2 if overlap_units else 0)
                if projected_overlap > options.overlap_chars:
                    break
                overlap_units.insert(0, previous)
                overlap_length = projected_overlap
            current = overlap_units
            current_length = overlap_length
            if current and current_length + 2 + len(unit.text) > options.max_chars:
                current = []
                current_length = 0
        projected = current_length + (2 if current else 0) + len(unit.text)
        if current and current_length >= options.target_chars and projected > options.target_chars:
            groups.append(current)
            current = []
            current_length = 0
        current.append(unit)
        current_length += len(unit.text) + (2 if len(current) > 1 else 0)
    if current:
        groups.append(current)

    if len(groups) > 1:
        last_text = "\n\n".join(unit.text for unit in groups[-1])
        previous_text = "\n\n".join(unit.text for unit in groups[-2])
        if len(last_text) < options.min_chars and len(previous_text) + len(last_text) + 2 <= options.max_chars:
            groups[-2].extend(groups[-1])
            groups.pop()

    spans: list[ChunkSpan] = []
    for group in groups:
        chunk_text = "\n\n".join(unit.text for unit in group).strip()
        if not chunk_text:
            continue
        spans.append(
            ChunkSpan(
                text=chunk_text,
                start=min(unit.start for unit in group),
                end=max(unit.end for unit in group),
                split_kind="semantic_window",
            )
        )
    return spans


def infer_tags(heading: str, text: str, pics: list[str]) -> list[str]:
    value = f"{heading}\n{text}".lower()
    tags: list[str] = []
    rules = [
        ("warning", r"\b(?:danger|warning|caution)\b|危险|警告|注意|严禁|不得"),
        ("procedure", r"\b(?:step|install|remove|press|connect|clean|replace|setup)\b|"
         r"步骤|安装|拆卸|按下|连接|清洁|更换|设置"),
        ("troubleshooting", r"\btroubleshoot|problem|error|fault|does not|cannot\b|"
         r"故障|问题|错误|无法|不工作"),
        ("parts", r"\b(?:parts?|components?|overview|control panel)\b|部件|组件|配件|控制面板"),
        ("specification", r"\b(?:specifications?|voltage|capacity|dimensions?|weight)\b|"
         r"规格|电压|容量|尺寸|重量"),
        ("warranty", r"\b(?:warranty|guarantee|support)\b|保修|质保|售后"),
    ]
    for tag, pattern in rules:
        if re.search(pattern, value, re.IGNORECASE):
            tags.append(tag)
    if pics:
        tags.append("has_pic")
    if _STEP_LINE_RE.search(text) or len(re.findall(r"(?:^|\n)\s*\d+[.)、．]?\s*", text)) >= 2:
        tags.append("procedure")
    return _dedupe(tags)


def _summary(text: str, limit: int = 180) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    if len(value) <= limit:
        return value
    return value[: max(1, limit - 3)].rstrip(" ,，。") + "..."


def _pics_for_span(section: ParsedSection, span: ChunkSpan) -> list[str]:
    matched = [
        anchor.pic_id
        for anchor in section.image_anchors
        if span.start <= anchor.char_offset <= span.end
    ]
    if not matched and len(section.pics) == 1 and len(section.text) <= 1400:
        matched = list(section.pics)
    return _dedupe(matched)


def build_artifacts(
    manual_name: str,
    source: str,
    options: SplitOptions | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把源文档完整转换为可预览、可发布的 RAGV6 资产。"""

    product = safe_manual_name(manual_name)
    opts = options if isinstance(options, SplitOptions) else SplitOptions.from_mapping(options)
    document_title, parsed = parse_sections(product, source, opts)
    language = detect_language(source)
    section_chunks: list[dict[str, Any]] = []
    retrieval_chunks: list[dict[str, Any]] = []
    manual_sections: list[dict[str, Any]] = []

    for section_id, section in enumerate(parsed):
        heading_path = [part for part in section.heading_path if part]
        heading = section.heading or document_title
        tags = infer_tags(heading, section.text, section.pics)
        special = "warning" in tags
        parent = {
            "product": product,
            "lang": language,
            "section_id": section_id,
            "heading": heading,
            "heading_path": heading_path,
            "chapter": heading_path[0] if heading_path else heading,
            "subheading": heading_path[1] if len(heading_path) > 1 else "",
            "subsubheading": heading_path[2] if len(heading_path) > 2 else "",
            "heading_level": min(max(len(heading_path), 1), 6),
            "text": section.text,
            "pics": list(section.pics),
            "char_len": len(section.text),
            "pic_count": len(section.pics),
            "is_special": special,
            "summary": _summary(section.text),
            "tags": tags,
            "figure_refs": [],
            "linked_pics": list(section.pics),
            "evidence_pics": list(section.pics),
            "figure_links": {},
            "fact_linked_pics": [],
            "fact_links": {},
            "concept_linked_pics": [],
            "concept_links": {},
            "chunk_manager": {
                "source_start_line": section.source_start_line,
                "source_end_line": section.source_end_line,
            },
        }
        section_chunks.append(parent)
        manual_sections.append(
            {
                "heading": heading,
                "heading_path": heading_path,
                "text": section.text,
                "pics": list(section.pics),
                "pic_captions": list(section.pic_captions),
                "chunk_manager": {
                    "source_start_line": section.source_start_line,
                    "source_end_line": section.source_end_line,
                },
            }
        )

        spans = split_section_text(section.text, opts)
        for subchunk_id, span in enumerate(spans):
            chunk_pics = _pics_for_span(section, span)
            retrieval_tags = infer_tags(heading, span.text, chunk_pics)
            retrieval_chunks.append(
                {
                    "product": product,
                    "lang": language,
                    "parent_section_id": section_id,
                    "source_section_ids": [section_id],
                    "subchunk_id": subchunk_id,
                    "heading": heading,
                    "text": span.text,
                    "pics": chunk_pics,
                    "char_start": span.start,
                    "char_end": span.end,
                    "char_len": len(span.text),
                    "pic_count": len(chunk_pics),
                    "is_special": "warning" in retrieval_tags,
                    "summary": _summary(span.text, 160),
                    "tags": retrieval_tags,
                    "split_kind": span.split_kind,
                    "chunk_id": len(retrieval_chunks),
                    "caption_aux": "",
                    "linked_pics": chunk_pics,
                }
            )

    report = validate_artifacts(product, source, section_chunks, retrieval_chunks, opts)
    catalog_entry = {
        "lang": language,
        "total_chars": sum(row["char_len"] for row in section_chunks),
        "total_pics": sum(int(row["pic_count"]) for row in section_chunks),
        "section_count": len(section_chunks),
        "retrieval_chunk_count": len(retrieval_chunks),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "generated_at": utc_now(),
        "split_options": asdict(opts),
        "sections": [
            {
                "id": row["section_id"],
                "title": row["heading"],
                "summary": row["summary"],
                "char_len": row["char_len"],
                "pic_count": row["pic_count"],
                "tags": row["tags"],
            }
            for row in section_chunks
        ],
    }
    return {
        "manual": product,
        "document_title": document_title,
        "source": normalize_source_text(source),
        "source_sha256": catalog_entry["source_sha256"],
        "options": asdict(opts),
        "manual_document": {
            "manual": product,
            "section_count": len(manual_sections),
            "source_sha256": catalog_entry["source_sha256"],
            "generated_at": utc_now(),
            "sections": manual_sections,
        },
        "section_chunks": section_chunks,
        "retrieval_chunks": retrieval_chunks,
        "catalog_entry": catalog_entry,
        "quality": report,
    }


def validate_artifacts(
    product: str,
    source: str,
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    options: SplitOptions,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not sections:
        errors.append("没有生成父章节")
    if not chunks:
        errors.append("没有生成检索块")
    if any(not str(row.get("text") or "").strip() for row in sections):
        errors.append("父章节中存在空正文")
    if any(not str(row.get("text") or "").strip() for row in chunks):
        errors.append("检索块中存在空正文")

    section_ids = [int(row.get("section_id", -1)) for row in sections]
    if section_ids != list(range(len(sections))):
        errors.append("section_id 不连续")
    if any(int(row.get("parent_section_id", -1)) not in set(section_ids) for row in chunks):
        errors.append("检索块存在无效 parent_section_id")

    too_short = [row for row in chunks if int(row.get("char_len", 0)) < options.min_chars]
    too_long = [row for row in chunks if int(row.get("char_len", 0)) > options.max_chars]
    if too_short:
        warnings.append(f"{len(too_short)} 个检索块短于 {options.min_chars} 字")
    if too_long:
        errors.append(f"{len(too_long)} 个检索块超过 {options.max_chars} 字硬上限")
    untitled = [row for row in sections if not str(row.get("heading") or "").strip()]
    if untitled:
        warnings.append(f"{len(untitled)} 个章节缺少标题")

    covered_chars = sum(min(len(row.get("text", "")), options.target_chars) for row in chunks)
    source_chars = len(re.sub(r"\s+", "", source))
    duplicate_texts = len(chunks) - len({str(row.get("text") or "") for row in chunks})
    if duplicate_texts:
        warnings.append(f"发现 {duplicate_texts} 个完全重复检索块")
    if len(sections) > 600:
        warnings.append("单手册章节超过 600，建议检查标题识别是否过细")

    return {
        "status": "error" if errors else ("warning" if warnings else "ready"),
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            "source_chars": source_chars,
            "section_count": len(sections),
            "retrieval_chunk_count": len(chunks),
            "average_chunk_chars": round(
                sum(int(row.get("char_len", 0)) for row in chunks) / max(1, len(chunks)),
                1,
            ),
            "min_chunk_chars": min((int(row.get("char_len", 0)) for row in chunks), default=0),
            "max_chunk_chars": max((int(row.get("char_len", 0)) for row in chunks), default=0),
            "short_chunk_count": len(too_short),
            "long_chunk_count": len(too_long),
            "picture_count": len(_dedupe(pic for row in sections for pic in row.get("pics", []))),
            "estimated_coverage": round(min(1.0, covered_chars / max(1, source_chars)), 4),
        },
    }


def get_repository_status(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    section_path = data_dir / SECTION_CHUNKS_PATH.name
    retrieval_path = data_dir / RETRIEVAL_CHUNKS_PATH.name
    catalog_path = data_dir / CATALOG_PATH.name
    catalog = _read_json(catalog_path, {})
    sections = _read_json(section_path, [])
    chunks = _read_json(retrieval_path, [])
    state = _read_json(data_dir / "chunk-admin" / "state.json", {})
    index_dir = data_dir / "index"
    index_files = [index_dir / "dense.faiss", index_dir / "retrieval_index.pkl"]
    latest_data_mtime = max(
        (path.stat().st_mtime for path in (section_path, retrieval_path, catalog_path) if path.exists()),
        default=0,
    )
    latest_index_mtime = min(
        (path.stat().st_mtime for path in index_files if path.exists()),
        default=0,
    )
    return {
        "manual_count": len(catalog) if isinstance(catalog, dict) else 0,
        "section_count": len(sections) if isinstance(sections, list) else 0,
        "retrieval_chunk_count": len(chunks) if isinstance(chunks, list) else 0,
        "index_ready": all(path.exists() for path in index_files),
        "index_stale": bool(latest_data_mtime and latest_data_mtime > latest_index_mtime),
        "pending_manuals": list(state.get("pending_manuals") or []),
        "last_publish_at": state.get("last_publish_at"),
        "last_index_build_at": state.get("last_index_build_at"),
    }


def list_manuals(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    catalog = _read_json(data_dir / CATALOG_PATH.name, {})
    state = _read_json(data_dir / "chunk-admin" / "state.json", {})
    pending = set(state.get("pending_manuals") or [])
    if not isinstance(catalog, dict):
        return []
    result = []
    for product, meta in catalog.items():
        result.append(
            {
                "manual": product,
                "lang": meta.get("lang", ""),
                "section_count": int(meta.get("section_count", 0)),
                "retrieval_chunk_count": int(meta.get("retrieval_chunk_count", 0)),
                "total_chars": int(meta.get("total_chars", 0)),
                "total_pics": int(meta.get("total_pics", 0)),
                "source_sha256": meta.get("source_sha256", ""),
                "generated_at": meta.get("generated_at"),
                "index_status": "pending_rebuild" if product in pending else "published",
            }
        )
    return sorted(result, key=lambda row: str(row["manual"]).casefold())


def load_manual_detail(product: str, data_dir: Path = DATA_DIR) -> dict[str, Any]:
    manual = safe_manual_name(product)
    catalog = _read_json(data_dir / CATALOG_PATH.name, {})
    if manual not in catalog:
        raise ChunkPipelineError(f"手册不存在: {manual}")
    sections = [
        row
        for row in _read_json(data_dir / SECTION_CHUNKS_PATH.name, [])
        if str(row.get("product") or "") == manual
    ]
    chunks = [
        row
        for row in _read_json(data_dir / RETRIEVAL_CHUNKS_PATH.name, [])
        if str(row.get("product") or "") == manual
    ]
    return {
        "manual": manual,
        "catalog": catalog[manual],
        "sections": sections,
        "retrieval_chunks": chunks,
    }


def publish_artifacts(
    artifacts: dict[str, Any],
    *,
    data_dir: Path = DATA_DIR,
    source_dir: Path = MANUAL_SOURCE_DIR,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """事务式合并新手册数据并标记索引待重建。"""

    product = safe_manual_name(str(artifacts.get("manual") or ""))
    quality = artifacts.get("quality") or {}
    if quality.get("errors"):
        raise ChunkPipelineError("质量检查未通过：" + "；".join(quality["errors"]))

    with _PUBLISH_LOCK:
        catalog_path = data_dir / CATALOG_PATH.name
        section_path = data_dir / SECTION_CHUNKS_PATH.name
        retrieval_path = data_dir / RETRIEVAL_CHUNKS_PATH.name
        summaries_path = data_dir / SECTION_SUMMARIES_PATH.name
        manual_sections_dir = data_dir / "manual_sections"
        manual_path = manual_sections_dir / f"{product}.json"
        source_path = source_dir / f"{product}.md"

        catalog = _read_json(catalog_path, {})
        sections = _read_json(section_path, [])
        chunks = _read_json(retrieval_path, [])
        summaries = _read_json(summaries_path, {})
        if not isinstance(catalog, dict) or not isinstance(sections, list) or not isinstance(chunks, list):
            raise ChunkPipelineError("现有 chunk 数据结构无效，已停止发布")
        exists = product in catalog
        if exists and not replace_existing:
            raise ChunkPipelineError("同名手册已存在；确认覆盖后才能发布")

        backup_id = _create_backup(
            product=product,
            files=[catalog_path, section_path, retrieval_path, summaries_path, manual_path, source_path],
            data_dir=data_dir,
        )
        merged_sections = [row for row in sections if str(row.get("product") or "") != product]
        merged_sections.extend(artifacts["section_chunks"])
        merged_chunks = [row for row in chunks if str(row.get("product") or "") != product]
        merged_chunks.extend(artifacts["retrieval_chunks"])
        for chunk_id, row in enumerate(merged_chunks):
            row["chunk_id"] = chunk_id
        catalog[product] = artifacts["catalog_entry"]
        summaries = {
            key: value
            for key, value in summaries.items()
            if not str(key).startswith(f"{product}|")
        }
        for row in artifacts["section_chunks"]:
            summaries[f"{product}|{row['section_id']}"] = row["summary"]

        staging_root = Path(tempfile.mkdtemp(prefix="chunk-publish-", dir=str(data_dir)))
        try:
            staged_data = staging_root / "data"
            staged_source = staging_root / "source"
            _write_json(staged_data / CATALOG_PATH.name, catalog)
            _write_json(staged_data / SECTION_CHUNKS_PATH.name, merged_sections)
            _write_json(staged_data / RETRIEVAL_CHUNKS_PATH.name, merged_chunks)
            _write_json(staged_data / SECTION_SUMMARIES_PATH.name, summaries)
            _write_json(staged_data / "manual.json", artifacts["manual_document"])
            staged_source.mkdir(parents=True, exist_ok=True)
            (staged_source / "manual.md").write_text(
                artifacts["source"].rstrip() + "\n",
                encoding="utf-8",
            )

            data_dir.mkdir(parents=True, exist_ok=True)
            manual_sections_dir.mkdir(parents=True, exist_ok=True)
            source_dir.mkdir(parents=True, exist_ok=True)
            os.replace(staged_data / CATALOG_PATH.name, catalog_path)
            os.replace(staged_data / SECTION_CHUNKS_PATH.name, section_path)
            os.replace(staged_data / RETRIEVAL_CHUNKS_PATH.name, retrieval_path)
            os.replace(staged_data / SECTION_SUMMARIES_PATH.name, summaries_path)
            os.replace(staged_data / "manual.json", manual_path)
            os.replace(staged_source / "manual.md", source_path)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

        state_path = data_dir / "chunk-admin" / "state.json"
        state = _read_json(state_path, {})
        pending = _dedupe([*(state.get("pending_manuals") or []), product])
        state.update(
            {
                "pending_manuals": pending,
                "last_publish_at": utc_now(),
                "last_publish_manual": product,
                "last_backup_id": backup_id,
            }
        )
        _write_json(state_path, state)
        return {
            "manual": product,
            "created": not exists,
            "replaced": exists,
            "section_count": len(artifacts["section_chunks"]),
            "retrieval_chunk_count": len(artifacts["retrieval_chunks"]),
            "backup_id": backup_id,
            "index_status": "pending_rebuild",
            "repository": get_repository_status(data_dir),
        }


def _create_backup(product: str, files: list[Path], data_dir: Path) -> str:
    backup_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + re.sub(
        r"[^A-Za-z0-9_-]+",
        "-",
        product,
    ).strip("-")[:40]
    root = data_dir / "chunk-admin" / "backups" / backup_id
    suffix = 1
    while root.exists():
        root = data_dir / "chunk-admin" / "backups" / f"{backup_id}-{suffix}"
        suffix += 1
    backup_id = root.name
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"backup_id": backup_id, "manual": product, "created_at": utc_now(), "files": []}
    for path in files:
        item = {"path": str(path.resolve()), "exists": path.exists()}
        if path.exists() and path.is_file():
            target = root / f"{len(manifest['files']):02d}-{path.name}"
            shutil.copy2(path, target)
            item["backup_file"] = target.name
            item["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest["files"].append(item)
    _write_json(root / "manifest.json", manifest)
    return backup_id


def list_backups(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    root = data_dir / "chunk-admin" / "backups"
    if not root.exists():
        return []
    result = []
    for entry in sorted(root.iterdir(), reverse=True):
        manifest_path = entry / "manifest.json"
        if manifest_path.is_file():
            result.append(_read_json(manifest_path, {}))
    return result


def rollback_backup(backup_id: str, data_dir: Path = DATA_DIR) -> dict[str, Any]:
    safe_id = re.sub(r"[^A-Za-z0-9T_-]", "", backup_id or "")
    root = data_dir / "chunk-admin" / "backups" / safe_id
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ChunkPipelineError("备份不存在")
    manifest = _read_json(manifest_path, {})
    restored: list[str] = []
    with _PUBLISH_LOCK:
        for item in manifest.get("files", []):
            destination = Path(str(item.get("path") or ""))
            backup_file = item.get("backup_file")
            if backup_file:
                source = root / str(backup_file)
                expected = str(item.get("sha256") or "")
                actual = hashlib.sha256(source.read_bytes()).hexdigest()
                if expected and actual != expected:
                    raise ChunkPipelineError(f"备份校验失败: {source.name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                restored.append(str(destination))
            elif not item.get("exists") and destination.exists() and destination.is_file():
                destination.unlink()
                restored.append(str(destination))
        state_path = data_dir / "chunk-admin" / "state.json"
        state = _read_json(state_path, {})
        state.update(
            {
                "pending_manuals": _dedupe(
                    [*(state.get("pending_manuals") or []), str(manifest.get("manual") or "")]
                ),
                "last_rollback_at": utc_now(),
                "last_rollback_backup_id": safe_id,
            }
        )
        _write_json(state_path, state)
    return {
        "backup_id": safe_id,
        "manual": manifest.get("manual"),
        "restored_files": restored,
        "index_status": "pending_rebuild",
    }


def mark_index_built(data_dir: Path = DATA_DIR) -> None:
    state_path = data_dir / "chunk-admin" / "state.json"
    state = _read_json(state_path, {})
    state.update(
        {
            "pending_manuals": [],
            "last_index_build_at": utc_now(),
            "last_index_build_status": "succeeded",
        }
    )
    _write_json(state_path, state)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="RAGV6 新手册一键 Chunk 切分程序")
    parser.add_argument("source", type=Path, help="Markdown/TXT/DOCX/PDF 手册文件")
    parser.add_argument("--manual", help="产品/手册名称；默认使用文件名")
    parser.add_argument("--target-chars", type=int, default=720)
    parser.add_argument("--min-chars", type=int, default=160)
    parser.add_argument("--max-chars", type=int, default=1100)
    parser.add_argument("--overlap-chars", type=int, default=90)
    parser.add_argument("--preview-json", type=Path, help="将完整预览写入 JSON")
    parser.add_argument("--publish", action="store_true", help="发布到现有数据仓库")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    raw = args.source.read_bytes()
    source = decode_uploaded_source(
        filename=args.source.name,
        content_base64=base64.b64encode(raw).decode("ascii"),
    )
    manual = args.manual or args.source.stem
    options = SplitOptions.from_mapping(vars(args))
    artifacts = build_artifacts(manual, source, options)
    metrics = artifacts["quality"]["metrics"]
    print(
        json.dumps(
            {
                "manual": artifacts["manual"],
                "quality": artifacts["quality"]["status"],
                **metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.preview_json:
        _write_json(args.preview_json, artifacts)
    if args.publish:
        result = publish_artifacts(
            artifacts,
            replace_existing=args.replace_existing,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.rebuild_index:
        if not args.publish:
            raise ChunkPipelineError("--rebuild-index 必须与 --publish 同时使用")
        from retrieval_engine import RetrievalEngine

        engine = RetrievalEngine()
        engine.build_index()
        mark_index_built()
        print(json.dumps({"index_status": "succeeded"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

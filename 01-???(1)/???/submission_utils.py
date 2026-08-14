"""
提交文件共用工具。

设计原则：
1. 绝大多数格式控制已前置到 SYSTEM_PROMPT（语气、长度、分段、<PIC> 使用、禁联系方式）
2. 这里只做「最小清洗 + 安全网」：去 markdown 符号、平台不兼容字符、泄漏的图片文件名、编造的联系方式
3. 保留换行（\\n）和段落分隔（\\n\\n），不扁平化 —— 技术答案范例 B 需要多段结构
"""

from __future__ import annotations

import json
import re


# ───────── Markdown 符号清理 ─────────
_MD_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_MD_HEADING_RE = re.compile(r"(?m)^#{1,6}\s*")
_MD_LIST_RE = re.compile(r"(?m)^[\s]*[-*+]\s+")
_MD_NUM_LIST_RE = re.compile(r"(?m)^[\s]*\d+[\.\)]\s+")
_MD_CODE_FENCE_RE = re.compile(r"```[^\n]*\n?")
_MD_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_TABLE_SEP_RE = re.compile(r"(?m)^[\s|:\-]+$")
_PIPE_RE = re.compile(r"[ \t]*\|[ \t]*")
_BLOCKQUOTE_RE = re.compile(r"(?m)^\s*>\s?")
_MD_ESCAPE_RE = re.compile(r"\\([*_`\[\]()])")

# ───────── <PIC> 变体规范化 ─────────
_PIC_NAME_PATTERN = r"[A-Za-z][A-Za-z0-9_]*\d+_\d+"
_PIC_NAME_RE = re.compile(rf"\b({_PIC_NAME_PATTERN})\b")
# 先按 [[PIC:...]] 结构抽取，再对中间名称做最小清洗；不要预设文件名格式
_INLINE_PIC_RE = re.compile(r"\[\[\s*PIC\s*:\s*([^\]\n]+?)\s*\]\]", re.IGNORECASE)
_PIC_TAG_VARIANT_RE = re.compile(r"<\s*PIC\s*>", re.IGNORECASE)
_PIC_CLOSE_VARIANT_RE = re.compile(r"<\s*/\s*PIC\s*>", re.IGNORECASE)
_PIC_WITH_FILENAME_RE = re.compile(
    rf"<\s*PIC\s*>\s*({_PIC_NAME_PATTERN})?\s*<\s*/\s*PIC\s*>",
    re.IGNORECASE,
)
# 裸 PIC 单词：前后不是字母/尖括号。用于修复 LLM 写成 "PIC" 漏掉尖括号的情况
_BARE_PIC_RE = re.compile(r"(?<![<A-Za-z])PIC(?![A-Za-z>])")
# LLM 偶尔写成 [PIC] / (PIC) / 《PIC》 等括号变体
_BRACKET_PIC_RE = re.compile(r"[\[【(（《]\s*PIC\s*[\]】)）》]", re.IGNORECASE)

# ───────── 客服问题禁用的联系方式 ─────────
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"\b(?:400[-\s]?[0-9A-ZXx*]{3,4}[-\s]?[0-9A-ZXx*]{4}|"
    r"1[3-9]\d{9}|0\d{2,3}-\d{7,8})\b"
)
_CUSTOMER_CONTACT_TOKENS_RE = re.compile(
    r"客服热线|客服电话|官方客服|官方网站|官网|微信(?:号|公众号)?|小程序|12315"
)


# ───────── 基础清洗 ─────────

def _drop_unsupported_chars(text: str) -> str:
    """去掉竞赛平台常见不兼容字符（variation selector、超 BMP emoji）。"""
    if not text:
        return ""
    cleaned: list[str] = []
    for ch in text:
        code = ord(ch)
        if code == 0xFE0F:  # variation selector-16
            continue
        if code > 0xFFFF:  # 超 BMP（含 emoji），竞赛平台常误报
            continue
        cleaned.append(ch)
    return "".join(cleaned)


def sanitize_submission_text(text: str) -> str:
    """最小清洗：统一换行、去 BOM、去平台不兼容字符。"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    return _drop_unsupported_chars(text)


# ───────── Markdown 扁平化（保留 \n） ─────────

def _strip_markdown(text: str) -> str:
    """去除 markdown 语法符号，保留换行和段落结构。"""
    text = sanitize_submission_text(text)
    text = _MD_CODE_FENCE_RE.sub("", text)
    text = _TABLE_SEP_RE.sub("", text)
    text = _PIPE_RE.sub(" ", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_LIST_RE.sub("", text)
    text = _MD_NUM_LIST_RE.sub(lambda m: m.group(0).strip() + " ", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _MD_ESCAPE_RE.sub(r"\1", text)
    return text


def _normalize_pic_markers(text: str) -> str:
    """把 <pic> / <PIC>filename</PIC> / [PIC] / 裸 PIC 等各种变体统一成 <PIC>。"""
    text = _PIC_WITH_FILENAME_RE.sub("<PIC>", text)
    text = _BRACKET_PIC_RE.sub("<PIC>", text)
    text = _PIC_TAG_VARIANT_RE.sub("<PIC>", text)
    text = _PIC_CLOSE_VARIANT_RE.sub("", text)
    text = re.sub(rf"<PIC>\s*({_PIC_NAME_PATTERN})", "<PIC>", text)
    text = re.sub(rf"({_PIC_NAME_PATTERN})\s*</PIC>", "", text)
    return text


def _normalize_manual_reference_symbols(text: str) -> str:
    """Normalize parser/model spellings of the manual's circled callouts."""
    circled = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤", "6": "⑥", "7": "⑦", "8": "⑧", "9": "⑨", "10": "⑩"}
    text = re.sub(
        r"\$?\\textcircled\{(?:\\scriptsize\{)?([1-9]|10)\}?\}\$?",
        lambda match: circled.get(match.group(1), match.group(1)),
        text,
    )
    # In this source format, \circledcirc is the second callout. Models can
    # visually misread it as any circled Latin letter or a hollow circle. When
    # that glyph occurs on the same callout line as ① and ③, restore it to ②.
    text = text.replace(r"$\circledcirc$", "②").replace(r"\circledcirc", "②")
    lines = []
    for line in text.splitlines(keepends=True):
        if "①" in line and "③" in line:
            line = re.sub(r"[ⓐ-ⓩⒶ-Ⓩ○]", "②", line)
        lines.append(line)
    return "".join(lines)


def _repair_bare_pic(text: str, expected_count: int) -> str:
    """
    兜底：LLM 偶尔把 <PIC> 写成裸 PIC 单词。
    仅当期望有图片 (expected_count>0) 时才修复，避免误伤正文中出现的 "PIC" 词。
    """
    if expected_count <= 0:
        return text
    current = text.count("<PIC>")
    if current >= expected_count:
        return text
    return _BARE_PIC_RE.sub("<PIC>", text, count=expected_count - current)


def _strip_leaked_filenames(text: str, pics: list[str]) -> str:
    """安全网：LLM 理论上不该写文件名，这里兜底把漏网的清掉。"""
    if pics:
        escaped = sorted((re.escape(p) for p in set(pics) if p), key=len, reverse=True)
        if escaped:
            text = re.sub(r"\b(?:%s)\b" % "|".join(escaped), "", text)
    text = _PIC_NAME_RE.sub("", text)
    return text


def _tidy_whitespace(text: str) -> str:
    """保留 \\n\\n 段落分隔，收敛多余空白。"""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def inject_inline_pic_refs(text: str, pics: list[str]) -> str:
    """
    把正文中的 <PIC> 按顺序替换成 [[PIC:文件名]]，供模型在回答中原样保留。
    这是给 LLM 看的中间格式，最终提交前会再转回 <PIC> + pics 列表。
    """
    if not text or not pics or "<PIC>" not in text:
        return text
    parts = text.split("<PIC>")
    rebuilt = parts[0]
    for idx, tail in enumerate(parts[1:]):
        marker = f"[[PIC:{pics[idx]}]]" if idx < len(pics) else "<PIC>"
        rebuilt += marker + tail
    return rebuilt


def extract_inline_pic_refs(text: str) -> tuple[str, list[str]]:
    """从 [[PIC:文件名]] 中抽取图片顺序，并替换回 <PIC>。"""
    pics: list[str] = []

    def repl(match: re.Match) -> str:
        pic_name = match.group(1).strip()
        if pic_name:
            pics.append(pic_name)
        return "<PIC>"

    normalized = _INLINE_PIC_RE.sub(repl, text or "")
    return normalized, pics


# ───────── 对外接口 ─────────

def is_customer_service_question(question_id: int) -> bool:
    """id < 64 为通用客服问题（硬路由）。"""
    return question_id < 64


def flatten_answer(text: str) -> str:
    """旧接口：扁平化成单行。保留供旧脚本兼容，新流程不用。"""
    if not text:
        return ""
    text = _strip_markdown(text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return sanitize_submission_text(text.strip())


def normalize_technical_answer(answer: str, pics: list[str]) -> tuple[str, list[str]]:
    """
    规范化技术答案：
    - 保留换行和段落结构（范例 B 有空行分段）
    - 去 markdown 符号 / <PIC> 变体 / 泄漏的文件名
    - 把 <PIC> 个数与 pics 数量对齐：多余的 <PIC> 删掉，pics 裁到 ≤ <PIC> 数
    - 不再对技术题图片数量做硬限制，避免多图题被后处理截断
    """
    text = _strip_markdown(answer)
    text = _normalize_manual_reference_symbols(text)
    text, inline_pics = extract_inline_pic_refs(text)
    text = _normalize_pic_markers(text)
    pics = _dedupe_keep_order(inline_pics + pics)
    text = _repair_bare_pic(text, len(pics))
    text = _strip_leaked_filenames(text, pics)

    pic_count_in_text = text.count("<PIC>")
    if pic_count_in_text > len(pics):
        pieces = text.split("<PIC>")
        rebuilt = pieces[0]
        for i, tail in enumerate(pieces[1:], start=1):
            rebuilt += ("<PIC>" if i <= len(pics) else "") + tail
        text = rebuilt
    elif pic_count_in_text < len(pics):
        pics = pics[:pic_count_in_text]

    text = _tidy_whitespace(text).strip()
    if not text:
        text = "未在手册中找到相关内容，建议联系售后确认。"
        pics = []
    return sanitize_submission_text(text), pics


def normalize_customer_service_answer(answer: str) -> str:
    """
    规范化客服答案：
    - 去 markdown、去所有 <PIC>、去泄漏文件名
    - 删除编造的电话/邮箱/URL/客服热线/微信/官网/小程序/12315
    - 范例均为单段，不强制扁平化（保留 LLM 输出的段落结构）
    """
    text = _strip_markdown(answer)
    text, _inline_pics = extract_inline_pic_refs(text)
    text = _normalize_pic_markers(text)
    text = text.replace("<PIC>", "")
    text = _strip_leaked_filenames(text, [])
    text = _URL_RE.sub("", text)
    text = _EMAIL_RE.sub("", text)
    text = _PHONE_RE.sub("", text)
    text = _CUSTOMER_CONTACT_TOKENS_RE.sub("", text)

    text = _tidy_whitespace(text).strip()
    if not text:
        text = "您好，您的问题已收到，我们会尽快为您处理。"
    return sanitize_submission_text(text)


def format_submission_ret(question_id: int, answer: str, pics: list[str]) -> str:
    """生成最终 CSV 的 ret 字段内容。"""
    if is_customer_service_question(question_id):
        return normalize_customer_service_answer(answer)

    normalized_answer, normalized_pics = normalize_technical_answer(answer, pics)
    if not normalized_pics:
        return normalized_answer
    pics_json = json.dumps(normalized_pics, ensure_ascii=False)
    return sanitize_submission_text(f"{normalized_answer},{pics_json}")


if __name__ == "__main__":
    sample_tech = """表带尺寸

表带尺寸如下所示。注意：单独销售的配件表带可能略有差异。
<PIC>

环境条件
<PIC>
"""
    sample_cs = """您好，非常抱歉给您带来困扰！维修后短期内出现同样故障，属于我们的维修失误，支持免费重新维修。请您提供维修单号，客服热线 400-800-1234 联系我们。"""

    print("--- 技术 ---")
    print(normalize_technical_answer(sample_tech, ["Manual16_51", "Manual16_52"]))
    print()
    print("--- 客服 ---")
    print(normalize_customer_service_answer(sample_cs))

"""决赛题公网媒体输入预处理。

初赛接口以 ``images=[data:image/...;base64,...]`` 为标准输入，但决赛公开题中
可能直接把图片 URL 或内容页短链写进 question。该模块把两种输入统一成现有
多模态链路可消费的 Base64 data URL，并从普通网页提取标题、摘要和代表图。

实现只负责“读取用户显式提供的媒体”，不参与知识检索，也不写入产品答案。
所有由图片识别出的事实仍需在后续 RAG 阶段由手册证据确认。
"""

from __future__ import annotations

import base64
import html
import ipaddress
import re
import socket
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx


_URL_RE = re.compile(
    r"(?<![\w@])(?:https?://|www\.)[^\s<>\"']+",
    re.IGNORECASE,
)
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}>，。；：！？）】》"
_REDIRECT_CODES = {301, 302, 303, 307, 308}
_SUPPORTED_IMAGE_TYPES = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/webp": "webp",
}


@dataclass
class MediaIngestResult:
    """一次媒体归一化的结果及可审计信息。"""

    question: str
    images: list[str]
    discovered_urls: list[str] = field(default_factory=list)
    fetched_images: list[str] = field(default_factory=list)
    page_contexts: list[dict[str, str]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def trace(self) -> dict[str, object]:
        """返回不含图片字节的 trace，避免日志泄露或体积膨胀。"""

        return {
            "discovered_urls": self.discovered_urls,
            "fetched_images": self.fetched_images,
            "page_contexts": self.page_contexts,
            "errors": self.errors,
            "input_image_count": len(self.images) - len(self.fetched_images),
            "resolved_image_count": len(self.images),
        }


class _OpenGraphParser(HTMLParser):
    """提取网页中与问答最相关的 Open Graph 元数据。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}
        self._inside_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {str(k).lower(): (v or "") for k, v in attrs}
        if tag.lower() == "title":
            self._inside_title = True
            return
        if tag.lower() != "meta":
            return
        key = (attrs_map.get("property") or attrs_map.get("name") or "").lower()
        value = attrs_map.get("content", "").strip()
        if key and value:
            self.meta.setdefault(key, []).append(html.unescape(value))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._inside_title = False

    def handle_data(self, data: str) -> None:
        if self._inside_title and data.strip():
            self._title_parts.append(data.strip())

    @property
    def title(self) -> str:
        return " ".join(self._title_parts).strip()


def extract_http_urls(text: str) -> list[str]:
    """按出现顺序提取并去重 URL，并把裸 ``www.`` 地址规范化为 HTTPS。"""

    urls: list[str] = []
    for match in _URL_RE.finditer(text or ""):
        url = match.group(0).rstrip(_TRAILING_URL_PUNCTUATION)
        if url.lower().startswith("www."):
            url = f"https://{url}"
        if url and url not in urls:
            urls.append(url)
    return urls


def text_without_http_urls(text: str) -> str:
    """移除已识别 URL，供上层判断用户是否只提交了一个链接。"""

    remainder = _URL_RE.sub(" ", text or "")
    remainder = re.sub(r"\s+", " ", remainder).strip()
    return remainder.strip(_TRAILING_URL_PUNCTUATION + " ")


def _assert_public_http_url(url: str) -> None:
    """拒绝本机、内网和保留地址，避免公网 URL 输入形成 SSRF。"""

    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("仅支持带有效主机名的 HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("不允许 URL 携带用户凭据")

    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        }
    except socket.gaierror as exc:
        raise ValueError("URL 主机名无法解析") from exc

    if not addresses:
        raise ValueError("URL 主机名未解析到地址")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("不允许访问内网、本机或保留地址")


def _bounded_get(
    client: httpx.Client,
    url: str,
    *,
    max_bytes: int,
    max_redirects: int = 5,
) -> tuple[bytes, str, str]:
    """逐跳校验重定向并限制响应大小，返回内容、类型和最终 URL。"""

    current = url
    for _ in range(max_redirects + 1):
        _assert_public_http_url(current)
        with client.stream("GET", current, follow_redirects=False) as response:
            if response.status_code in _REDIRECT_CODES:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("重定向响应缺少 Location")
                current = urljoin(current, location)
                continue
            response.raise_for_status()
            declared = response.headers.get("content-length", "").strip()
            if declared.isdigit() and int(declared) > max_bytes:
                raise ValueError(f"远程内容超过 {max_bytes} 字节限制")

            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"远程内容超过 {max_bytes} 字节限制")
                chunks.append(chunk)
            media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            return b"".join(chunks), media_type, str(response.url)
    raise ValueError("远程地址重定向次数过多")


def _detect_image_type(content: bytes, media_type: str) -> str | None:
    """同时检查 Content-Type 与文件签名，避免把 HTML 伪装成图片。"""

    normalized = _SUPPORTED_IMAGE_TYPES.get(media_type)
    if content.startswith(b"\xff\xd8\xff"):
        signature = "jpeg"
    elif content.startswith(b"\x89PNG\r\n\x1a\n"):
        signature = "png"
    elif len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        signature = "webp"
    else:
        signature = None
    if normalized and signature and normalized != signature:
        return None
    return signature or normalized


def _to_data_url(content: bytes, image_type: str) -> str:
    return f"data:image/{image_type};base64,{base64.b64encode(content).decode('ascii')}"


def _last_meta(parser: _OpenGraphParser, *keys: str) -> str:
    """优先同名 meta 的最后一个值；内容页通常在站点默认值后覆盖文章值。"""

    for key in keys:
        for value in reversed(parser.meta.get(key, [])):
            if value.strip():
                return value.strip()
    return ""


def _longest_meta(parser: _OpenGraphParser, *keys: str) -> str:
    """摘要优先信息量最大的值，过滤“某平台欢迎语”一类默认描述。"""

    values = [
        value.strip()
        for key in keys
        for value in parser.meta.get(key, [])
        if value.strip()
    ]
    return max(values, key=len, default="")


def _compact_text(value: str, limit: int = 800) -> str:
    value = re.sub(r"\s+", " ", html.unescape(value or "")).strip()
    return value[:limit]


def ingest_question_media(
    question: str,
    images: list[str],
    *,
    max_images: int = 3,
    max_image_bytes: int = 5 * 1024 * 1024,
    max_page_bytes: int = 2 * 1024 * 1024,
    timeout_seconds: float = 15.0,
) -> MediaIngestResult:
    """把 question 中的图片 URL/网页短链归一化到现有多模态输入。

    处理顺序遵循用户输入顺序，已有 Base64 图片优先保留。普通网页仅提取
    title/description 和第一张代表图，不下载视频，也不执行页面脚本。
    单个 URL 失败不会中断整轮问答，失败原因写入 trace 供排查。
    """

    result = MediaIngestResult(question=question, images=list(images[:max_images]))
    result.discovered_urls = extract_http_urls(question)
    if not result.discovered_urls or len(result.images) >= max_images:
        return result
    resolved_source_urls: list[str] = []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    timeout = httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 8.0))
    page_notes: list[str] = []

    with httpx.Client(headers=headers, timeout=timeout, trust_env=True) as client:
        for source_url in result.discovered_urls:
            if len(result.images) >= max_images:
                break
            try:
                content, media_type, final_url = _bounded_get(
                    client,
                    source_url,
                    max_bytes=max(max_image_bytes, max_page_bytes),
                )
                image_type = _detect_image_type(content, media_type)
                if image_type:
                    if len(content) > max_image_bytes:
                        raise ValueError("远程图片超过单图大小限制")
                    result.images.append(_to_data_url(content, image_type))
                    result.fetched_images.append(final_url)
                    resolved_source_urls.append(source_url)
                    continue

                if media_type not in {"text/html", "application/xhtml+xml"}:
                    raise ValueError(f"不支持的远程内容类型: {media_type or 'unknown'}")
                if len(content) > max_page_bytes:
                    raise ValueError("网页内容超过解析大小限制")

                parser = _OpenGraphParser()
                parser.feed(content.decode("utf-8", errors="replace"))
                title = _compact_text(_last_meta(parser, "og:title", "twitter:title") or parser.title, 240)
                description = _compact_text(
                    _longest_meta(parser, "og:description", "description", "twitter:description"),
                    800,
                )
                page_context = {
                    "source_url": source_url,
                    "final_url": final_url,
                    "title": title,
                    "description": description,
                }
                result.page_contexts.append(page_context)
                if title or description:
                    page_notes.append(
                        "外部内容页（仅作为用户输入语境）: "
                        f"标题={title or '未提供'}；描述={description or '未提供'}"
                    )

                representative = _last_meta(parser, "og:image", "twitter:image", "twitter:image:src")
                if representative and len(result.images) < max_images:
                    representative_url = urljoin(final_url, representative)
                    image_bytes, image_media_type, image_final_url = _bounded_get(
                        client,
                        representative_url,
                        max_bytes=max_image_bytes,
                    )
                    representative_type = _detect_image_type(image_bytes, image_media_type)
                    if representative_type:
                        result.images.append(_to_data_url(image_bytes, representative_type))
                        result.fetched_images.append(image_final_url)
                        resolved_source_urls.append(source_url)
            except Exception as exc:  # 单个外链失败时仍允许文本/RAG链路继续工作。
                result.errors.append({"url": source_url, "error": str(exc)[:300]})

    # 已成功转换为图片输入的 URL 不再参与语义检索。保留它只会污染 BM25
    # 关键词和模型的问题重写；原始地址仍完整记录在 trace.discovered_urls。
    semantic_question = question
    for resolved_url in resolved_source_urls:
        semantic_question = semantic_question.replace(resolved_url, " ")
    semantic_question = re.sub(r"[ \t]+", " ", semantic_question)
    semantic_question = re.sub(r" *\n *", "\n", semantic_question).strip()
    result.question = semantic_question or question
    if page_notes:
        result.question = f"{result.question}\n\n" + "\n".join(page_notes)
    return result

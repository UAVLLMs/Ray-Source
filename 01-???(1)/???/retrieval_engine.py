"""
V5 检索引擎：

- 稀疏检索：jieba + rank_bm25
- 稠密检索：硅基流动 Pro/BAAI/bge-m3 + FAISS(IP)
- 融合：RRF
- 重排：硅基流动 BAAI/bge-reranker-v2-m3

说明：
- 当前默认 embedding/rerank 均走硅基流动线上服务，代码内保留默认 key。
- 本地 8091/8090 仅作为显式 fallback：只有通过 EMBEDDING_BASE_URL/RERANK_BASE_URL 覆盖到本地时才会使用。
- rerank 不再做全库预编码，而是在召回后对 `(query, doc)` 候选对打分。
"""

from __future__ import annotations

import json
import logging
import lzma
import os
import pickle
import re
import threading
import struct
import sys
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss

# jieba 0.42.1 imports pkg_resources only to locate its bundled dictionary.
# On some Windows hosts that import can enter a very slow setuptools/WMI scan.
# Force jieba's built-in direct-file fallback during its import, then restore
# the interpreter state so unrelated modules can still use pkg_resources.
_pkg_resources_missing = object()
_pkg_resources_previous = sys.modules.get("pkg_resources", _pkg_resources_missing)
sys.modules["pkg_resources"] = None
try:
    import jieba
finally:
    if _pkg_resources_previous is _pkg_resources_missing:
        sys.modules.pop("pkg_resources", None)
    else:
        sys.modules["pkg_resources"] = _pkg_resources_previous
    del _pkg_resources_missing, _pkg_resources_previous

import numpy as np
import requests
from rank_bm25 import BM25Okapi
from tqdm import tqdm

log = logging.getLogger(__name__)

from dotenv import load_dotenv
from rerank_client import RerankClient, RerankError
from shared_retrieval_cache import SharedRetrievalCache

load_dotenv()

try:
    from config_runtime import apply_default_env

    apply_default_env()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"

RETRIEVAL_CHUNKS_PATH = DATA_DIR / "retrieval_chunks.json"
SECTION_CHUNKS_PATH = DATA_DIR / "section_chunks.json"
CATALOG_PATH = DATA_DIR / "catalog.json"

FAISS_PATH = INDEX_DIR / "dense.faiss"
COMPACT_DENSE_PATH = INDEX_DIR / "dense_vectors.float32.shuffle.xz"
METADATA_PATH = INDEX_DIR / "retrieval_index.pkl"
_COMPACT_DENSE_MAGIC = b"RAGV6D1\0"

DEFAULT_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
DEFAULT_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Pro/BAAI/bge-m3")
DEFAULT_MAX_CONCURRENCY = int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "4"))
DEFAULT_RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", "https://api.siliconflow.cn/v1")
DEFAULT_RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")
DEFAULT_RERANK_MODEL = os.getenv("RERANK_MODEL_ALIAS", "BAAI/bge-reranker-v2-m3")
DEFAULT_RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1").lower() not in {"0", "false", "no"}
RERANK_FALLBACK_LOG_PATH = Path(os.getenv("RERANK_FALLBACK_LOG", "")).expanduser() if os.getenv("RERANK_FALLBACK_LOG") else None
RERANK_TIMING_LOG_PATH = Path(os.getenv("RERANK_TIMING_LOG", "")).expanduser() if os.getenv("RERANK_TIMING_LOG") else None
_RERANK_FALLBACK_LOCK = threading.Lock()
_RERANK_CONTEXT = threading.local()
RETURN_PARENT_SECTION = os.getenv("RETURN_PARENT_SECTION", "1").lower() not in {"0", "false", "no"}
SEARCH_CACHE_SIZE = max(0, int(os.getenv("RETRIEVAL_SEARCH_CACHE_SIZE", "512")))
DENSE_QUERY_CACHE_SIZE = max(0, int(os.getenv("RETRIEVAL_DENSE_CACHE_SIZE", "512")))
RERANK_CACHE_SIZE = max(0, int(os.getenv("RETRIEVAL_RERANK_CACHE_SIZE", "512")))
RERANK_CANDIDATE_LIMIT = max(16, int(os.getenv("RERANK_CANDIDATE_LIMIT", "32")))


def normalize_product_name(product: str) -> str:
    return product.lower().replace("手册", "").strip()


# 段级别名：让非手册术语（如 "battery conversion"）在 BM25/dense 全库搜索时
# 仍能命中正确段。不改产品路由，只影响 search_manual(products=[]) 全库搜的排名。
_SECTION_ALIASES: dict[str, list[str]] = {}
_aliases_path = DATA_DIR / "section_aliases.json"
if _aliases_path.exists():
    try:
        with open(_aliases_path) as f:
            _SECTION_ALIASES = json.load(f).get("aliases", {})
    except Exception:
        pass


def build_searchable_text(chunk: dict) -> str:
    # caption_aux（info_table 表格 OCR 全文）仅用于召回，不进模型正文。
    # 不截断：靠加大 bge-m3 ubatch（8192）让整表完整进向量；极少数超模型 token
    # 上限的由 _embed 分片平均处理，不丢尾部信息。
    parts = [
        chunk.get("product", ""),
        chunk.get("heading", ""),
        chunk.get("summary", ""),
        chunk.get("text", ""),
        chunk.get("caption_aux", ""),
    ]
    # 注入段级业务别名（如 "battery conversion" → Battery switches 段）
    alias_key = f"{chunk.get('product', '')}|{chunk.get('heading', '')}"
    aliases = _SECTION_ALIASES.get(alias_key, [])
    if aliases:
        parts.append("\n".join(aliases))
    return "\n".join(part for part in parts if part).strip()


# rerank 输入长度上限：reranker 实测 ~1000 字符内安全（10117 字才 500）。截到 900 留余量。
_RERANK_MAX_CHARS = 900


def build_rerank_text(chunk: dict) -> str:
    """rerank 输入文本：用完整 caption_aux（含 info_table 表格数据，如 'MAXIMUM LOAD 160kg'），
    并把 caption_aux 提前，保证表格数据进 rerank（否则 query='max load' 匹配不到中文表名）；
    整体截断到 _RERANK_MAX_CHARS 避免超 reranker 上限触发 500（500 会让整批回退 RRF）。
    关键规格/数据通常在表头/前部，截尾影响小。召回端仍用完整 caption_aux，不受此截断影响。"""
    parts = [
        chunk.get("product", ""),
        chunk.get("heading", ""),
        chunk.get("caption_aux", ""),   # 提前：info_table 表格数据优先进 rerank
        chunk.get("summary", ""),
        chunk.get("text", ""),
    ]
    return "\n".join(part for part in parts if part).strip()[:_RERANK_MAX_CHARS]


# dense 单片字符上限：中文 1 字≈1 token，留足余量在 bge-m3 ubatch(512) 内。
# 注意：这只切 dense 的 embedding 输入，BM25 仍用完整 search_texts（关键词全覆盖）。
_EMBED_MAX_SEG_CHARS = 380


def split_text_for_embedding(text: str, max_chars: int = _EMBED_MAX_SEG_CHARS) -> list[str]:
    """按行（markdown 表格行/段边界）切片，每片 ≤ max_chars，不切断整行。
    长 info_table 表格靠多片 + mean-pooling 完整进 dense 向量，零删尾部信息。"""
    text = text or ""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    segs: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for line in text.split("\n"):
        ll = len(line) + 1
        if cur and cur_len + ll > max_chars:
            segs.append("\n".join(cur))
            cur, cur_len = [], 0
        if len(line) > max_chars:  # 单行超长（罕见），硬切
            for j in range(0, len(line), max_chars):
                segs.append(line[j:j + max_chars])
            continue
        cur.append(line)
        cur_len += ll
    if cur:
        segs.append("\n".join(cur))
    return [s for s in segs if s.strip()] or [text[:max_chars]]


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _tokenize_english(text: str) -> list[str]:
    """纯英文/数字/型号文本的简单空白分词，不做 stemming 以避免破坏型号/缩写匹配。"""
    import re
    tokens: list[str] = []
    for raw in re.split(r'[\s,;:!?()\[\]{}"\'<>]+', text):
        w = raw.strip().lower()
        if not w:
            continue
        # 保留原始词项（型号/缩写对 BM25 关键词匹配至关重要）
        # 同时补充去标点版本（如 "safety." → "safety"）
        cleaned = "".join(ch for ch in w if ch.isalnum() or ch in "._-+/")
        if cleaned and cleaned != w:
            tokens.append(cleaned)
        tokens.append(w)
    return list(dict.fromkeys(tokens))  # 去重保序


def tokenize_mixed(text: str) -> list[str]:
    """
    中英混合分词：
    - 中文：jieba 分词 + 双字片段补充
    - 英文/型号/数字：简单空白分词，保留原始词项（不做 stemming，避免破坏型号和缩写匹配）
    """
    text = text.strip().lower()
    if not text:
        return []

    # 快速检测是否包含中文
    has_cjk = contains_cjk(text)

    tokens: list[str] = []
    for word in jieba.cut_for_search(text):
        word = word.strip().lower()
        if not word:
            continue
        if contains_cjk(word):
            if len(word) == 1:
                tokens.append(word)
            else:
                tokens.append(word)
                for i in range(len(word) - 1):
                    tokens.append(word[i:i + 2])
        else:
            cleaned = "".join(ch for ch in word if ch.isalnum() or ch in "._-+/")
            if cleaned:
                tokens.append(cleaned)

    # 对英文文本补充空白分词，弥补 jieba 对纯英文分词过粗的缺陷
    if not has_cjk or any(ch.isascii() and ch.isalpha() for ch in text):
        eng_tokens = _tokenize_english(text)
        for t in eng_tokens:
            if t not in tokens:
                tokens.append(t)

    return tokens



@dataclass
class SearchResult:
    """检索结果的统一返回结构。

    chunk_id/source 保留召回单元信息；text/pics 通常来自完整 parent section，使 agent 看到同一主题下的正文、警告和图片锚点，而不是只看到命中的短 chunk。
    """
    chunk_id: int
    product: str
    heading: str
    text: str
    pics: list[str]
    score: float
    source: dict


class EmbeddingClient:
    """硅基流动 OpenAI-compatible embedding 客户端。

    只封装 /embeddings 调用和返回顺序恢复；并发、分片、mean-pooling 和索引构建由 RetrievalEngine 统一控制。
    """
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        timeout: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = (
            float(os.getenv("EMBEDDING_QUERY_TIMEOUT_SECONDS", "6"))
            if timeout is None
            else float(timeout)
        )
        # A query embedding is issued for every cache miss. Reusing a session
        # per worker thread retains HTTPS connections without sharing a
        # requests.Session across concurrent FastAPI workers.
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = False
            self._thread_local.session = session
        return session

    def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []

        response = self._session().post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": texts,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = sorted(payload["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]


class RetrievalEngine:
    """手册 RAG 的统一检索后端。

    构建 BM25 与 FAISS dense 两套索引，查询时按产品范围召回候选、用 RRF 融合稀疏/稠密结果，再用 reranker 精排。默认 chunk 负责定位，最终返回 parent section，兼顾召回精度和证据完整性。
    """
    def __init__(
        self,
        retrieval_chunks_path: Path = RETRIEVAL_CHUNKS_PATH,
        section_chunks_path: Path = SECTION_CHUNKS_PATH,
        catalog_path: Path = CATALOG_PATH,
        index_dir: Path = INDEX_DIR,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        rerank_base_url: str = DEFAULT_RERANK_BASE_URL,
        rerank_api_key: str = DEFAULT_RERANK_API_KEY,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        rerank_enabled: bool = DEFAULT_RERANK_ENABLED,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        self.retrieval_chunks_path = Path(retrieval_chunks_path)
        self.section_chunks_path = Path(section_chunks_path)
        self.catalog_path = Path(catalog_path)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.faiss_path = self.index_dir / FAISS_PATH.name
        self.metadata_path = self.index_dir / METADATA_PATH.name

        self.embedding_model = embedding_model
        self.max_concurrency = max(1, max_concurrency)
        self.client = EmbeddingClient(base_url=base_url, api_key=api_key)
        self.rerank_enabled = rerank_enabled
        self.rerank_client = RerankClient(
            base_url=rerank_base_url,
            api_key=rerank_api_key,
            model=rerank_model,
        )

        self.retrieval_chunks: list[dict] = []
        self.section_chunks: list[dict] = []
        self.catalog: dict = {}
        self.section_lookup: dict[tuple[str, int], dict] = {}

        self.search_texts: list[str] = []
        self.bm25: BM25Okapi | None = None
        self.tokenized_docs: list[list[str]] = []

        self.dense_index: faiss.Index | None = None
        self.dense_vectors: np.ndarray | None = None

        # All caches are tied to the loaded index generation.  They store
        # generic retrieval intermediates/results, never authored answers.
        self._cache_lock = threading.RLock()
        self._index_generation = "unloaded"
        self._search_cache: OrderedDict[tuple, tuple[list[SearchResult], int]] = OrderedDict()
        self._dense_query_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._rerank_cache: OrderedDict[tuple, list[int]] = OrderedDict()
        self._shared_cache = SharedRetrievalCache()
        self._cache_counters = {
            "search_hit": 0,
            "search_miss": 0,
            "dense_hit": 0,
            "dense_miss": 0,
            "rerank_hit": 0,
            "rerank_miss": 0,
        }

    @staticmethod
    def _normalize_cache_text(value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip()).casefold()

    def _cache_get(self, cache: OrderedDict, key: tuple, counter: str):
        with self._cache_lock:
            if key in cache:
                value = cache.pop(key)
                cache[key] = value
                self._cache_counters[f"{counter}_hit"] += 1
                return value
        shared = self._shared_cache.get(
            counter,
            key,
            result_type=SearchResult if counter == "search" else None,
        )
        if shared is not None:
            self._cache_counters[f"{counter}_hit"] += 1
            limit = {"search": SEARCH_CACHE_SIZE, "dense": DENSE_QUERY_CACHE_SIZE, "rerank": RERANK_CACHE_SIZE}[counter]
            self._cache_put(cache, key, shared, limit, shared=False)
            return shared
        self._cache_counters[f"{counter}_miss"] += 1
        return None

    def _cache_put(self, cache: OrderedDict, key: tuple, value, limit: int, *, shared: bool = True) -> None:
        if limit <= 0:
            return
        with self._cache_lock:
            cache.pop(key, None)
            cache[key] = value
            while len(cache) > limit:
                cache.popitem(last=False)
        if shared:
            kind = "search" if cache is self._search_cache else "dense" if cache is self._dense_query_cache else "rerank"
            self._shared_cache.put(kind, key, value)

    def _reset_retrieval_caches(self) -> None:
        parts = []
        compact_dense_path = self.index_dir / COMPACT_DENSE_PATH.name
        for path in (self.metadata_path, self.faiss_path, compact_dense_path):
            if path.exists():
                stat = path.stat()
                parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        self._index_generation = "|".join(parts) or f"memory:{len(self.retrieval_chunks)}"
        with self._cache_lock:
            self._search_cache.clear()
            self._dense_query_cache.clear()
            self._rerank_cache.clear()
            for key in self._cache_counters:
                self._cache_counters[key] = 0

    def cache_stats(self) -> dict[str, int | str]:
        with self._cache_lock:
            return {
                **self._cache_counters,
                "search_entries": len(self._search_cache),
                "dense_entries": len(self._dense_query_cache),
                "rerank_entries": len(self._rerank_cache),
                "index_generation": self._index_generation,
                "shared": self._shared_cache.snapshot(),
            }

    def load_documents(self) -> None:
        with open(self.retrieval_chunks_path, "r", encoding="utf-8") as f:
            self.retrieval_chunks = json.load(f)
        with open(self.section_chunks_path, "r", encoding="utf-8") as f:
            self.section_chunks = json.load(f)
        # 图文合一 LLM 章节总结（gen_section_summaries.py 产物）：挂到 llm_summary，
        # 供章节元数据、审计和后续检索上下文使用，旧截断 summary 仅作兜底。
        summ_path = self.section_chunks_path.parent / "section_summaries.json"
        if summ_path.exists():
            try:
                llm_summaries = json.loads(summ_path.read_text(encoding="utf-8"))
                for section in self.section_chunks:
                    s = llm_summaries.get(f"{section['product']}|{section['section_id']}", "")
                    if s and not s.startswith("__ERROR__"):
                        section["llm_summary"] = s
            except Exception:
                pass
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        self.section_lookup = {
            (section["product"], section["section_id"]): section
            for section in self.section_chunks
        }
        self.search_texts = [build_searchable_text(chunk) for chunk in self.retrieval_chunks]

        # 产品路由索引：product_name -> [chunk_id, ...]
        self.product_chunk_ids: dict[str, list[int]] = {}
        for i, chunk in enumerate(self.retrieval_chunks):
            product = chunk["product"]
            if product not in self.product_chunk_ids:
                self.product_chunk_ids[product] = []
            self.product_chunk_ids[product].append(i)

    def build_index(self, batch_size: int = 32) -> None:
        self.load_documents()

        print(f"构建 BM25 索引: {len(self.search_texts)} 文档")
        self.tokenized_docs = [
            tokenize_mixed(text)
            for text in tqdm(self.search_texts, desc="BM25 分词", unit="doc")
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        print(
            f"构建 dense 向量索引: {len(self.search_texts)} 文档, "
            f"batch_size={batch_size}, concurrency={self.max_concurrency}"
        )
        dense_vectors = self._embed_corpus(self.search_texts, self.embedding_model, batch_size)
        dense_vectors = self._l2_normalize(dense_vectors)

        self.dense_vectors = dense_vectors

        dimension = dense_vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(dense_vectors)
        self.dense_index = index

        faiss.write_index(index, str(self.faiss_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(
                {
                    "embedding_model": self.embedding_model,
                    "retrieval_chunks": self.retrieval_chunks,
                    "section_chunks": self.section_chunks,
                    "catalog": self.catalog,
                    "search_texts": self.search_texts,
                    "tokenized_docs": self.tokenized_docs,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        self._reset_retrieval_caches()

    def load_index(self) -> None:
        compact_dense_path = self.index_dir / COMPACT_DENSE_PATH.name
        if not self.metadata_path.exists() or not (self.faiss_path.exists() or compact_dense_path.exists()):
            raise FileNotFoundError("索引文件不存在，请先运行 build_index。")

        with open(self.metadata_path, "rb") as f:
            data = pickle.load(f)

        self.embedding_model = data["embedding_model"]
        self.retrieval_chunks = data["retrieval_chunks"]
        self.section_chunks = data["section_chunks"]
        # 缓存里的 section_chunks 不含 llm_summary（索引早于总结生成），这里补挂。
        summ_path = self.section_chunks_path.parent / "section_summaries.json"
        if summ_path.exists():
            try:
                llm_summaries = json.loads(summ_path.read_text(encoding="utf-8"))
                for section in self.section_chunks:
                    s = llm_summaries.get(f"{section['product']}|{section['section_id']}", "")
                    if s and not s.startswith("__ERROR__"):
                        section["llm_summary"] = s
            except Exception:
                pass
        self.catalog = data["catalog"]
        self.search_texts = data["search_texts"]
        self.tokenized_docs = data["tokenized_docs"]

        # FAISS rows are positional, but the stored document payload can lag
        # behind a reviewed Chunk/Section correction. Overlay current source
        # documents only when their identities and order still match the
        # persisted index, then rebuild the in-memory BM25 representation.
        with open(self.retrieval_chunks_path, "r", encoding="utf-8") as f:
            current_retrieval_chunks = json.load(f)
        with open(self.section_chunks_path, "r", encoding="utf-8") as f:
            current_section_chunks = json.load(f)
        persisted_ids = [str(item.get("chunk_id")) for item in self.retrieval_chunks]
        current_ids = [str(item.get("chunk_id")) for item in current_retrieval_chunks]
        if persisted_ids == current_ids:
            self.retrieval_chunks = current_retrieval_chunks
            self.section_chunks = current_section_chunks
            self.search_texts = [build_searchable_text(chunk) for chunk in self.retrieval_chunks]
            self.tokenized_docs = [tokenize_mixed(text) for text in self.search_texts]
        else:
            log.warning(
                "current retrieval chunk identities differ from persisted FAISS rows; using persisted metadata"
            )

        self.section_lookup = {
            (section["product"], section["section_id"]): section
            for section in self.section_chunks
        }
        self.bm25 = BM25Okapi(self.tokenized_docs)
        # 用 deserialize_index 读字节，避免 Windows 上中文路径导致 FAISS C++ fopen 失败
        if self.faiss_path.exists():
            with open(self.faiss_path, "rb") as f:
                faiss_bytes = f.read()
            self.dense_index = faiss.deserialize_index(np.frombuffer(faiss_bytes, dtype=np.uint8))
            dense_vectors = data.get("dense_vectors")
            if dense_vectors is None:
                dense_vectors = self.dense_index.reconstruct_n(0, self.dense_index.ntotal)
            self.dense_vectors = np.ascontiguousarray(dense_vectors, dtype=np.float32)
        else:
            dense_vectors = self._load_compact_dense_vectors()
            self.dense_index = faiss.IndexFlatIP(dense_vectors.shape[1])
            self.dense_index.add(dense_vectors)
            self.dense_vectors = dense_vectors

        self.product_chunk_ids = {}
        for i, chunk in enumerate(self.retrieval_chunks):
            product = chunk["product"]
            if product not in self.product_chunk_ids:
                self.product_chunk_ids[product] = []
            self.product_chunk_ids[product].append(i)
        self._reset_retrieval_caches()

    def _load_compact_dense_vectors(self) -> np.ndarray:
        payload = (self.index_dir / COMPACT_DENSE_PATH.name).read_bytes()
        header_size = len(_COMPACT_DENSE_MAGIC) + 8
        if payload[:len(_COMPACT_DENSE_MAGIC)] != _COMPACT_DENSE_MAGIC:
            raise ValueError("invalid compact dense index magic")
        rows, dimensions = struct.unpack("<II", payload[len(_COMPACT_DENSE_MAGIC):header_size])
        decoded = lzma.decompress(payload[header_size:])
        expected = rows * dimensions * 4
        if len(decoded) != expected:
            raise ValueError(f"compact dense byte length {len(decoded)} != {expected}")
        byte_planes = np.frombuffer(decoded, dtype=np.uint8).reshape(4, dimensions, rows)
        raw = np.ascontiguousarray(byte_planes.transpose(2, 1, 0))
        vectors = raw.view(np.float32).reshape(rows, dimensions)
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def ensure_index(self) -> None:
        if self.dense_index is not None and self.bm25 is not None:
            return
        self.load_index()

    def search_manual(
        self,
        keywords: list[str],
        *,
        semantic_query: str = "",
        original_query: str = "",
        top_k: int = 8,
        products: list[str] | None = None,
        diagnostics: dict | None = None,
    ) -> tuple[list[SearchResult], int]:
        """统一检索入口：BM25 20 + 向量 20 → 合并去重 → 关键词 rerank 取 6 + 用户问题 rerank 取 4 → 合并去重后按 rank 截断。
        返回 (结果列表, 被过滤数量)；当前不再使用固定 rerank 分数阈值。
        """
        self.ensure_index()
        sparse_query = " ".join(keyword.strip() for keyword in keywords if keyword.strip())
        original_query = (original_query or "").strip()
        semantic_query = (semantic_query or "").strip()

        dense_query = semantic_query or sparse_query
        if not sparse_query and not dense_query:
            return [], 0

        search_cache_key = (
            self._index_generation,
            tuple(self._normalize_cache_text(keyword) for keyword in keywords if keyword.strip()),
            self._normalize_cache_text(semantic_query),
            self._normalize_cache_text(original_query),
            int(top_k),
            tuple(sorted(self._normalize_cache_text(product) for product in (products or []))),
        )
        # A UI audit trace needs the actual ranked lists from this request.  Do
        # not return the final-response cache in that mode: lower-level BM25,
        # embedding and rerank caches still apply, while the trace remains
        # accurate and request-local.
        cached_search = None if diagnostics is not None else self._cache_get(
            self._search_cache, search_cache_key, "search"
        )
        if cached_search is not None:
            return deepcopy(cached_search)

        recall_n = 20
        per_keyword_recall_n = 5
        allowed_doc_ids: list[int] | None = None
        if products:
            allowed: set[int] = set()
            for product in products:
                allowed.update(self.product_chunk_ids.get(product, []))
            allowed_doc_ids = sorted(allowed)

        retrieval_question = original_query or semantic_query or sparse_query or dense_query
        sparse_doc_ids = self._sparse_recall(
            sparse_query or dense_query,
            top_n=recall_n,
            allowed_doc_ids=allowed_doc_ids,
        )
        heading_doc_ids = self._heading_recall(retrieval_question, top_n=recall_n, allowed_doc_ids=allowed_doc_ids)
        # Model-generated tool keywords are supplementary recall hints, not a
        # replacement for the user's literal wording.  Keep an independent
        # BM25 channel for the original question so a precise component term
        # (for example, "脚轮") cannot be displaced by a broader generated
        # keyword from a neighboring section (for example, "滤网").
        original_sparse_doc_ids: list[int] = []
        if original_query and original_query != (sparse_query or dense_query):
            original_sparse_doc_ids = self._sparse_recall(
                original_query,
                top_n=recall_n,
                allowed_doc_ids=allowed_doc_ids,
            )
        # Dense is request-scoped: every fresh search attempts the embedding
        # call.  A failure may only affect this request and must never disable
        # Dense for the worker process or for later users.
        dense_state: dict[str, object] = {}
        dense_doc_ids = self._dense_recall(
            dense_query if dense_query else sparse_query,
            top_n=recall_n,
            allowed_doc_ids=allowed_doc_ids,
            dense_state=dense_state,
        )

        if not products:
            reorder_query = original_query or semantic_query or sparse_query
            original_sparse_doc_ids = self._reorder_by_lang(reorder_query, original_sparse_doc_ids)
            sparse_doc_ids = self._reorder_by_lang(reorder_query, sparse_doc_ids)
            dense_doc_ids = self._reorder_by_lang(reorder_query, dense_doc_ids)

        keyword_phrases = [
            keyword.strip()
            for keyword in keywords
            if keyword and keyword.strip()
        ]
        extra_sparse_doc_ids: list[int] = []
        extra_dense_doc_ids: list[int] = []
        for phrase in keyword_phrases:
            extra_sparse_doc_ids.extend(
                self._sparse_recall(
                    phrase,
                    top_n=per_keyword_recall_n,
                    allowed_doc_ids=allowed_doc_ids,
                )
            )
            # One semantic query embedding already covers the complete
            # question. Per-keyword Dense calls multiply network latency and
            # provide little value; BM25 remains the lexical expansion channel.

        if not products:
            reorder_query = original_query or semantic_query or sparse_query
            extra_sparse_doc_ids = self._reorder_by_lang(reorder_query, extra_sparse_doc_ids)
            extra_dense_doc_ids = self._reorder_by_lang(reorder_query, extra_dense_doc_ids)

        # Fuse complementary channels by reciprocal rank instead of source-order
        # concatenation.  Identical ranked lists are included only once so a
        # repeated keyword cannot accidentally double-weight BM25.
        ranked_lists: list[list[int]] = []
        seen_ranked_lists: set[tuple[int, ...]] = set()
        for ranked in (
            heading_doc_ids,
            original_sparse_doc_ids,
            sparse_doc_ids,
            dense_doc_ids,
            extra_sparse_doc_ids,
            extra_dense_doc_ids,
        ):
            deduped = list(dict.fromkeys(ranked))
            signature = tuple(deduped)
            if deduped and signature not in seen_ranked_lists:
                seen_ranked_lists.add(signature)
                ranked_lists.append(deduped)
        candidates = self._rrf_merge(
            ranked_lists,
            top_n=RERANK_CANDIDATE_LIMIT,
        )
        if not candidates:
            return [], 0

        # Keep the RRF score and every channel rank for the audit sidebar.
        # This is explanatory metadata only; it deliberately does not alter
        # the fusion or final ranking.
        if diagnostics is not None:
            rrf_scores: dict[int, float] = {}
            channel_ranks: dict[int, dict[str, int]] = {}
            channel_names = (
                "heading", "original_bm25", "keyword_bm25", "dense",
                "keyword_expansion_bm25", "keyword_expansion_dense",
            )
            for channel_name, ranked in zip(channel_names, (
                heading_doc_ids,
                original_sparse_doc_ids,
                sparse_doc_ids,
                dense_doc_ids,
                extra_sparse_doc_ids,
                extra_dense_doc_ids,
            )):
                for position, doc_id in enumerate(ranked, start=1):
                    channel_ranks.setdefault(doc_id, {})[channel_name] = position
            for ranked in ranked_lists:
                for position, doc_id in enumerate(ranked, start=1):
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + position)
            diagnostics.update({
                "queries": {
                    "sparse": sparse_query,
                    "semantic": dense_query,
                    "original": original_query or dense_query,
                },
                "channel_ranks": channel_ranks,
                "rrf_scores": rrf_scores,
                "candidate_ids": list(candidates),
                "dense": {
                    "attempted": bool(dense_state.get("attempted")),
                    "available": bool(dense_state.get("available")),
                    "fallback": not bool(dense_state.get("available")),
                    "error": str(dense_state.get("error") or ""),
                },
            })

        keyword_rerank_query = semantic_query or sparse_query
        keyword_top = self._rerank_candidates(
            keyword_rerank_query,
            candidates,
            top_n=6,
        )[:6]

        user_top: list[int] = []
        if original_query and original_query != keyword_rerank_query:
            user_top = self._rerank_candidates(
                original_query,
                candidates,
                top_n=4,
            )[:4]

        if diagnostics is not None:
            diagnostics["rerank"] = {
                "keyword_query": keyword_rerank_query,
                "keyword_ranks": {doc_id: rank for rank, doc_id in enumerate(keyword_top, start=1)},
                "original_query": original_query if user_top else "",
                "original_ranks": {doc_id: rank for rank, doc_id in enumerate(user_top, start=1)},
            }

        # Cross-encoders can favor a long, broadly related passage over a short
        # leaf section that names the requested feature exactly. Promote at most
        # one lexical heading anchor, and only when coarse retrieval independently
        # placed it in the BM25 or dense top three. This calibrates granularity;
        # it neither expands the candidate pool nor relies on product/image ids.
        heading_anchor = self._exact_leaf_heading_anchor(
            original_query or keyword_rerank_query,
            candidates,
            trusted_ids=(
                set(original_sparse_doc_ids[:3])
                | set(sparse_doc_ids[:3])
                | set(dense_doc_ids[:3])
                | set(heading_doc_ids[:3])
            ),
        )

        seen: set[int] = set()
        final_ids: list[int] = []
        for doc_id in heading_anchor + keyword_top + user_top:
            heading_key = self._result_dedup_key(doc_id)
            if heading_key in seen:
                continue
            seen.add(heading_key)
            final_ids.append(doc_id)

        final_ids, evidence_roles = self._rank_structural_evidence(
            question=original_query or semantic_query or sparse_query,
            ranked_ids=final_ids,
            candidate_ids=candidates,
            max_sections=min(top_k, 8),
        )
        final_ids, relevance_metadata, filtered_count = self._filter_extremely_irrelevant_evidence(
            question=retrieval_question,
            ranked_ids=final_ids,
            evidence_roles=evidence_roles,
            dense_query_array=(
                dense_state.get("query_array")
                if isinstance(dense_state.get("query_array"), np.ndarray)
                else None
            ),
            dense_attempted=bool(dense_state.get("attempted")),
        )
        if diagnostics is not None:
            candidates_trace: list[dict] = []
            final_rank = {doc_id: rank for rank, doc_id in enumerate(final_ids, start=1)}
            rerank_trace = diagnostics.get("rerank", {})
            channel_ranks = diagnostics.get("channel_ranks", {})
            rrf_scores = diagnostics.get("rrf_scores", {})
            # `_filter_extremely_irrelevant_evidence` computes calibrated
            # values for final evidence only.  The audit table should also
            # show raw BM25/Dense values for candidates that were considered
            # but not selected, so calculate the same transparent inputs for
            # the full RRF pool without affecting selection.
            audit_bm25: dict[int, float] = {}
            audit_dense: dict[int, float] = {}
            if self.bm25 is not None:
                audit_tokens = tokenize_mixed(retrieval_question)
                if audit_tokens:
                    raw_scores = self.bm25.get_scores(audit_tokens)
                    audit_bm25 = {doc_id: float(raw_scores[doc_id]) for doc_id in candidates}
            if self.dense_vectors is not None:
                audit_key = (
                    self._index_generation,
                    self.embedding_model,
                    self._normalize_cache_text(retrieval_question),
                )
                audit_query_vector = self._cache_get(self._dense_query_cache, audit_key, "dense")
                if audit_query_vector is not None:
                    audit_vectors = self.dense_vectors[np.asarray(candidates, dtype=np.int64)]
                    audit_dense = {
                        doc_id: float(value)
                        for doc_id, value in zip(candidates, audit_vectors @ audit_query_vector[0])
                    }
            audit_top_bm25 = max(audit_bm25.values(), default=0.0)
            for rrf_rank, doc_id in enumerate(candidates, start=1):
                chunk = self.retrieval_chunks[doc_id]
                relevance = dict(relevance_metadata.get(doc_id, {}))
                if "bm25_raw" not in relevance:
                    bm25_raw = audit_bm25.get(doc_id, 0.0)
                    relevance["bm25_raw"] = round(bm25_raw, 6)
                    relevance["bm25_relative"] = round(bm25_raw / audit_top_bm25, 6) if audit_top_bm25 else 0.0
                if "dense_cosine" not in relevance:
                    dense_raw = audit_dense.get(doc_id)
                    relevance["dense_cosine"] = round(dense_raw, 6) if dense_raw is not None else None
                candidates_trace.append({
                    "rrf_rank": rrf_rank,
                    "rrf_score": round(float(rrf_scores.get(doc_id, 0.0)), 8),
                    "chunk_id": int(chunk.get("chunk_id", doc_id)),
                    "internal_id": doc_id,
                    "heading": str(chunk.get("heading") or ""),
                    "excerpt": re.sub(r"\s+", " ", str(chunk.get("text") or "")).strip()[:380],
                    "channel_ranks": channel_ranks.get(doc_id, {}),
                    "keyword_rerank_rank": rerank_trace.get("keyword_ranks", {}).get(doc_id),
                    "original_rerank_rank": rerank_trace.get("original_ranks", {}).get(doc_id),
                    "final_rank": final_rank.get(doc_id),
                    "evidence_role": evidence_roles.get(doc_id, "not_selected"),
                    "selected": doc_id in final_rank,
                    **relevance,
                })
            diagnostics.update({
                "candidates": candidates_trace,
                "filtered_count": filtered_count,
                "final_ids": list(final_ids),
            })
        response = (self._build_results(
            final_ids[:top_k],
            evidence_roles=evidence_roles,
            relevance_metadata=relevance_metadata,
        ), filtered_count)
        self._cache_put(
            self._search_cache,
            search_cache_key,
            deepcopy(response),
            SEARCH_CACHE_SIZE,
        )
        return response

    def _filter_extremely_irrelevant_evidence(
        self,
        *,
        question: str,
        ranked_ids: list[int],
        evidence_roles: dict[int, str] | None = None,
        dense_query_array: np.ndarray | None = None,
        dense_attempted: bool = False,
    ) -> tuple[list[int], dict[int, dict[str, float | int | str | None]], int]:
        """Drop only candidates that are simultaneously weak in BM25 and Dense."""
        evidence_roles = evidence_roles or {}
        if not ranked_ids:
            return [], {}, 0
        query_tokens = tokenize_mixed(question)
        bm25_values: dict[int, float] = {}
        if query_tokens and self.bm25 is not None:
            scores = self.bm25.get_scores(query_tokens)
            bm25_values = {doc_id: float(scores[doc_id]) for doc_id in ranked_ids}
        dense_values: dict[int, float] = {}
        if self.dense_vectors is not None and question.strip():
            # Reuse the query vector from the main Dense recall.  If that
            # request already failed, stay on BM25 for this request only;
            # never retry the same failed remote call in the filter stage.
            query_array = dense_query_array
            if query_array is None and not dense_attempted:
                dense_cache_key = (self._index_generation, self.embedding_model, self._normalize_cache_text(question))
                query_array = self._cache_get(self._dense_query_cache, dense_cache_key, "dense")
                if query_array is None:
                    try:
                        query_vector = self.client.embed_texts([question], self.embedding_model)[0]
                        query_array = self._l2_normalize(np.asarray([query_vector], dtype=np.float32))
                        self._cache_put(self._dense_query_cache, dense_cache_key, query_array.copy(), DENSE_QUERY_CACHE_SIZE)
                    except Exception:
                        query_array = None
            if query_array is not None:
                vectors = self.dense_vectors[np.asarray(ranked_ids, dtype=np.int64)]
                scores = vectors @ query_array[0]
                dense_values = {doc_id: float(scores[index]) for index, doc_id in enumerate(ranked_ids)}
        top_bm25 = max(bm25_values.values(), default=0.0)
        top_dense = max(dense_values.values(), default=None)
        query_heading_terms = self._heading_terms(question)
        # Core means direct answer evidence, not merely a chunk containing a
        # common query word. Remove question operators and generic actions when
        # checking whether a second result names the same subject as the query.
        normalized_topic_question = re.sub(
            r"(?:是什么|是啥|啥意思|什么意思|怎么|如何|怎样|哪些|哪个|是否|能否|可以吗|请问)",
            " ",
            question or "",
        )
        query_topic_terms = self._heading_terms(normalized_topic_question) - {
            "add", "adjust", "assemble", "charge", "clean", "close", "connect",
            "enable", "insert", "install", "maintain", "open", "operate", "remove",
            "replace", "set", "start", "stop", "use",
            "添加", "调节", "组装", "充电", "清洁", "关闭", "连接", "开启",
            "插入", "安装", "维护", "操作", "取出", "拆卸", "更换", "设置",
            "启动", "停止", "使用",
        }
        kept: list[int] = []
        metadata: dict[int, dict[str, float | int | str | None]] = {}
        provisional: list[tuple[int, int, bool, int, float, float, float]] = []
        filtered = 0
        for rank, doc_id in enumerate(ranked_ids):
            chunk = self.retrieval_chunks[doc_id]
            bm25_raw = bm25_values.get(doc_id, 0.0)
            bm25_relative = bm25_raw / top_bm25 if top_bm25 > 0 else 0.0
            dense_raw = dense_values.get(doc_id)
            dense_gap = max(0.0, float(top_dense) - float(dense_raw)) if top_dense is not None and dense_raw is not None else None
            dense_relative = max(0.0, 1.0 - dense_gap / 0.22) if dense_gap is not None else 0.0
            heading_hits = len(query_heading_terms & self._heading_terms(chunk.get("heading", "")))
            heading_coverage = heading_hits / max(1, len(query_heading_terms)) if query_heading_terms else 0.0
            topic_heading_hits = len(query_topic_terms & self._heading_terms(chunk.get("heading", "")))
            role = evidence_roles.get(doc_id, "ranked")
            structural = role in {"primary", "support", "related"}
            extremely_low = bool(
                rank > 0 and not structural and heading_hits == 0
                and top_bm25 > 0 and bm25_relative < 0.025
                and dense_gap is not None and dense_gap >= 0.20
            )
            combined = min(1.0, 0.44 * min(1.0, bm25_relative) + 0.46 * dense_relative + 0.10 * min(1.0, heading_coverage * 2.0))
            metadata[doc_id] = {
                "bm25_raw": round(bm25_raw, 6), "bm25_relative": round(bm25_relative, 6),
                "dense_cosine": round(dense_raw, 6) if dense_raw is not None else None,
                "dense_gap_from_top": round(dense_gap, 6) if dense_gap is not None else None,
                "heading_hits": heading_hits, "heading_coverage": round(heading_coverage, 6),
                "topic_heading_hits": topic_heading_hits,
                "combined_relevance": round(combined, 6),
            }
            provisional.append(
                (rank, doc_id, structural, topic_heading_hits, combined, bm25_relative, dense_relative)
            )
            if extremely_low:
                filtered += 1
            else:
                kept.append(doc_id)
        if not kept:
            kept = [ranked_ids[0]]
            filtered = max(0, len(ranked_ids) - 1)

        # Exactly one result anchors the answer. A structurally selected primary
        # section is more authoritative than an ordinary reranker hit; support
        # and all remaining hits stay optional related evidence.
        primary_kept = [
            row for row in provisional
            if row[1] in kept and evidence_roles.get(row[1]) == "primary"
        ]
        ordinary_kept = [row for row in provisional if row[1] in kept and not row[2]]
        anchor_rows = primary_kept or ordinary_kept
        core_ids: set[int] = {anchor_rows[0][1]} if anchor_rows else set()

        for _rank, doc_id, structural, _topic_hits, _combined, _bm25_rel, _dense_rel in provisional:
            if doc_id not in kept:
                tier = "extremely_low"
            elif doc_id in core_ids:
                tier = "core"
            else:
                tier = "related"
            metadata[doc_id]["relevance_tier"] = tier
        return kept, metadata, filtered

    def _rank_structural_evidence(
        self,
        *,
        question: str,
        ranked_ids: list[int],
        candidate_ids: list[int],
        max_sections: int,
    ) -> tuple[list[int], dict[int, str]]:
        """Complete a small manual topic from its heading hierarchy.

        Retrieval scores rank isolated chunks. A broad procedural question often
        names their shared parent heading instead (for example, a setup topic with
        separate power and signal-connection leaves). When the question exactly
        covers a meaningful ancestor heading, complete that small sibling group.
        Sections containing concrete list steps become primary evidence; warnings
        and notes remain supporting evidence. No product, chunk id, or manual
        phrase is encoded here.
        """
        if not ranked_ids or max_sections < 2:
            return ranked_ids, {}

        requested_count = self._requested_prefix_count(question)
        procedural = self._is_procedural_question(question)
        enumerative = self._is_enumeration_question(question)
        if not procedural and not enumerative and requested_count is None:
            return ranked_ids, {}

        query_terms = self._heading_terms(question)
        if not query_terms:
            return ranked_ids, {}

        ranked_position = {doc_id: index for index, doc_id in enumerate(ranked_ids)}
        candidate_position = {doc_id: index for index, doc_id in enumerate(candidate_ids)}

        def representative(doc_ids: list[int]) -> int:
            return min(
                doc_ids,
                key=lambda doc_id: (
                    0 if doc_id in ranked_position else 1,
                    ranked_position.get(doc_id, candidate_position.get(doc_id, len(candidate_ids))),
                    doc_id,
                ),
            )

        # A feature overview often names its dependent subsections as bullet
        # items while the detailed explanation lives in the next sibling
        # section. Pull in only siblings under the same level-2 heading whose
        # leaf title overlaps a named bullet.
        linked_ids: list[int] = []
        linked_seed_ids: list[int] = []
        for seed_doc_id in ranked_ids[:3]:
            seed_chunk = self.retrieval_chunks[seed_doc_id]
            seed_parent_id = int(seed_chunk.get("parent_section_id", seed_doc_id))
            seed_section = self.section_lookup.get(
                (seed_chunk.get("product", ""), seed_parent_id)
            ) or seed_chunk
            bullet_terms = []
            for line in str(seed_section.get("text") or "").splitlines():
                match = re.match(r"^\s*(?:[•*-]+|\(?\d+\)?[.)、])\s*(.+?)\s*$", line)
                if not match:
                    continue
                terms = self._heading_terms(match.group(1))
                if len(terms) >= 2:
                    bullet_terms.append(terms)
            if not bullet_terms:
                continue
            seed_parts = self._heading_parts(seed_section.get("heading", ""))
            if len(seed_parts) < 3:
                continue
            product = seed_chunk.get("product", "")
            prefix = tuple(seed_parts[:2])
            section_pool = getattr(self, "section_chunks", self.section_lookup.values())
            for section in section_pool:
                if section.get("product") != product:
                    continue
                section_id = int(section.get("section_id", section.get("parent_section_id", 0)))
                if section_id == seed_parent_id:
                    continue
                parts = self._heading_parts(section.get("heading", ""))
                if len(parts) < 3 or tuple(parts[:2]) != prefix:
                    continue
                leaf_terms = self._heading_terms(parts[-1])
                if not any(
                    len(leaf_terms & named_terms) >= 2
                    and len(leaf_terms & named_terms) / len(named_terms) >= 0.45
                    for named_terms in bullet_terms
                ):
                    continue
                sibling_docs = [
                    doc_id
                    for doc_id in self.product_chunk_ids.get(product, [])
                    if int(self.retrieval_chunks[doc_id].get(
                        "parent_section_id", doc_id
                    )) == section_id
                ]
                if sibling_docs:
                    linked_ids.append(representative(sibling_docs))
                    linked_seed_ids.append(seed_doc_id)
                if len(linked_ids) >= 2:
                    break
            if len(linked_ids) >= 2:
                break
        linked_ids = list(dict.fromkeys(linked_ids))
        if linked_ids:
            completed = [*ranked_ids, *linked_ids]
            roles = {
                **{doc_id: "primary" for doc_id in linked_seed_ids},
                **{doc_id: "support" for doc_id in linked_ids},
            }
            return completed, roles


        # A narrowly named procedure can depend on the immediately preceding
        # sibling warning (for example, disconnect power before cleaning).  Add
        # only that adjacent safety preface, never arbitrary neighboring
        # procedures.  The final formatter restores document order.
        if procedural:
            top_doc_id = ranked_ids[0]
            top_chunk = self.retrieval_chunks[top_doc_id]
            top_parent_id = int(top_chunk.get("parent_section_id", top_doc_id))
            top_section = self.section_lookup.get(
                (top_chunk.get("product", ""), top_parent_id)
            ) or top_chunk
            top_parts = self._heading_parts(
                top_section.get("heading", top_chunk.get("heading", ""))
            )
            top_leaf_terms = self._heading_terms(top_parts[-1] if top_parts else "")
            top_tags = {
                str(tag).strip().lower()
                for tag in top_section.get("tags", [])
            }
            if (
                len(top_parts) >= 2
                and len(top_leaf_terms) >= 2
                and top_leaf_terms.issubset(query_terms)
                and "procedure" in top_tags
            ):
                product = top_chunk.get("product", "")
                prefix = tuple(top_parts[:-1])
                sibling_sections = sorted(
                    (
                        section
                        for section in self.section_chunks
                        if section.get("product") == product
                        and tuple(self._heading_parts(section.get("heading", ""))[:-1]) == prefix
                    ),
                    key=lambda section: int(section.get("section_id", 0)),
                )
                sibling_ids = [
                    int(section.get("section_id", 0))
                    for section in sibling_sections
                ]
                if top_parent_id in sibling_ids:
                    top_index = sibling_ids.index(top_parent_id)
                    if top_index > 0:
                        support_section = sibling_sections[top_index - 1]
                        support_tags = {
                            str(tag).strip().lower()
                            for tag in support_section.get("tags", [])
                        }
                        support_leaf_terms = self._heading_terms(
                            self._heading_parts(support_section.get("heading", ""))[-1]
                        )
                        support_terms = {
                            "warning", "warnings", "caution", "note", "notes",
                            "notice", "precaution", "precautions", "requirement",
                            "requirements", "safety", "danger", "警告", "注意",
                            "须知", "要求", "安全",
                        }
                        if (
                            support_tags & {"warning", "safety"}
                            or support_leaf_terms & support_terms
                        ):
                            support_parent_id = int(support_section.get("section_id", 0))
                            support_docs = [
                                doc_id
                                for doc_id in self.product_chunk_ids.get(product, [])
                                if int(self.retrieval_chunks[doc_id].get(
                                    "parent_section_id", doc_id
                                )) == support_parent_id
                            ]
                            if support_docs:
                                support_doc_id = representative(support_docs)
                                selected_keys = {
                                    self._result_dedup_key(top_doc_id),
                                    self._result_dedup_key(support_doc_id),
                                }
                                remainder = [
                                    doc_id for doc_id in ranked_ids
                                    if self._result_dedup_key(doc_id) not in selected_keys
                                ]
                                return (
                                    [top_doc_id, support_doc_id] + remainder,
                                    {
                                        top_doc_id: "primary",
                                        support_doc_id: "support",
                                    },
                                )

        best_match: tuple[int, int, int, tuple[str, ...], str] | None = None
        for rank, doc_id in enumerate(ranked_ids[:8]):
            chunk = self.retrieval_chunks[doc_id]
            parts = self._heading_parts(chunk.get("heading", ""))
            for depth, component in enumerate(parts[:-1], start=1):
                component_terms = self._heading_terms(component)
                if len(component_terms) < 2 or not component_terms.issubset(query_terms):
                    continue
                score = (len(component_terms), depth, -rank)
                if best_match is None or score > best_match[:3]:
                    best_match = (*score, tuple(parts[:depth]), chunk.get("product", ""))

        if best_match is None:
            return ranked_ids, {}

        prefix = best_match[3]
        product = best_match[4]
        parent_docs: dict[int, list[int]] = {}
        for doc_id in self.product_chunk_ids.get(product, []):
            chunk = self.retrieval_chunks[doc_id]
            parts = self._heading_parts(chunk.get("heading", ""))
            if tuple(parts[:len(prefix)]) != prefix:
                continue
            parent_id = int(chunk.get("parent_section_id", doc_id))
            parent_docs.setdefault(parent_id, []).append(doc_id)

        # Explicit requests such as "the first five" are ordered selections,
        # not relevance-ranked top-k questions. Select the requested number of
        # sibling sections in manual order when the named parent is exact.
        if requested_count is not None and len(parent_docs) >= requested_count:
            selected_parent_ids = sorted(parent_docs)[:requested_count]
            selected = [
                representative(parent_docs[parent_id])
                for parent_id in selected_parent_ids
            ]
            selected_keys = {self._result_dedup_key(doc_id) for doc_id in selected}
            remainder = [
                doc_id for doc_id in ranked_ids
                if self._result_dedup_key(doc_id) not in selected_keys
            ]
            return (
                selected + remainder,
                {doc_id: "primary" for doc_id in selected},
            )

        # Large branches are catalog categories, not one answer unit. In that
        # case preserve ordinary reranking instead of flooding the answer.
        if not 2 <= len(parent_docs) <= max_sections:
            return ranked_ids, {}

        primary: list[int] = []
        support: list[int] = []
        related: list[int] = []
        for parent_id, doc_ids in parent_docs.items():
            doc_id = representative(doc_ids)
            chunk = self.retrieval_chunks[doc_id]
            section = self.section_lookup.get((product, parent_id)) or chunk
            tags = {str(tag).strip().lower() for tag in section.get("tags", [])}
            leaf = self._heading_parts(section.get("heading", chunk.get("heading", "")))[-1]
            leaf_terms = self._heading_terms(leaf)
            support_terms = {
                "warning", "warnings", "caution", "note", "notes", "notice",
                "precaution", "precautions", "requirement", "requirements",
                "safety", "danger", "警告", "注意", "须知", "要求", "安全",
            }
            is_support = bool(tags & {"warning", "safety"}) or bool(leaf_terms & support_terms)
            list_items = sum(
                1
                for line in str(section.get("text") or "").splitlines()
                if re.match(r"^\s*(?:[•*-]+|\(?\d+\)?[.)、])\s*", line)
            )
            if enumerative:
                # The user named the exact parent category and asked what it
                # contains. Every direct child is therefore primary evidence,
                # even when the source prose is not formatted as a list.
                primary.append(doc_id)
            elif is_support:
                support.append(doc_id)
            elif list_items >= 2 or "procedure" in tags:
                primary.append(doc_id)
            else:
                related.append(doc_id)

        if not primary:
            return ranked_ids, {}

        source_order = lambda doc_id: (
            int(self.retrieval_chunks[doc_id].get("parent_section_id", doc_id)),
            doc_id,
        )
        primary.sort(key=source_order)
        support.sort(key=source_order)
        related.sort(key=source_order)
        family_ids = primary + support + related
        family_keys = {self._result_dedup_key(doc_id) for doc_id in family_ids}
        remainder = [
            doc_id for doc_id in ranked_ids
            if self._result_dedup_key(doc_id) not in family_keys
        ]
        roles = {
            **{doc_id: "primary" for doc_id in primary},
            **{doc_id: "support" for doc_id in support},
            **{doc_id: "related" for doc_id in related},
        }
        if enumerative:
            # A complete exact-parent enumeration is self-contained. Do not
            # append reranked neighbours from the next manual branches.
            return family_ids, roles
        return family_ids + remainder, roles

    @staticmethod
    def _requested_prefix_count(question: str) -> int | None:
        """Parse an explicit 'first N items' limit in Chinese or English."""
        value = (question or "").strip().lower()
        match = re.search(
            r"(?:前\s*(?P<zh>[一二三四五六七八九十])\s*(?:条|项|个)|"
            r"(?:前|first)\s*(?P<num>\d{1,2})\s*(?:条|项|个|items?|points?)?)",
            value,
        )
        if not match:
            english = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            }
            word_match = re.search(
                r"\bfirst\s+(one|two|three|four|five|six|seven|eight|nine|ten)"
                r"\s*(?:items?|points?)?\b",
                value,
            )
            return english.get(word_match.group(1)) if word_match else None
        if match.group("num"):
            count = int(match.group("num"))
        else:
            zh_values = {
                "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
            }
            count = zh_values.get(match.group("zh"), 0)
        return count if 1 <= count <= 10 else None

    @staticmethod
    def _is_procedural_question(question: str) -> bool:
        q = (question or "").strip().lower()
        return bool(
            re.search(r"\bhow\b|\bsteps?\b|\bprocedure\b|\binstructions?\b", q)
            or any(marker in q for marker in ("如何", "怎样", "怎么", "步骤", "流程"))
        )

    @staticmethod
    def _is_enumeration_question(question: str) -> bool:
        """Detect requests for the complete contents of a named category."""
        q = re.sub(r"\s+", " ", (question or "").strip().lower())
        if any(
            marker in q
            for marker in (
                "有哪些", "都有哪些", "有什么", "都有什么", "包括什么",
                "包含什么", "由什么组成", "分别是什么", "列出", "枚举",
                "概述一下",
            )
        ):
            return True
        return bool(
            re.search(
                r"\b(?:what|which)\s+(?:kinds?|types?|categories|"
                r"characteristics?|features?|components?|parts?|functions?|"
                r"items?|elements?|indicators?|states?|statuses?|signals?|symbols?)\b",
                q,
            )
            or re.search(r"\b(?:list|enumerate)\b", q)
            or re.search(
                r"\bwhat\b.*\b(?:include|includes|included|contain|contains|"
                r"consist|consists|comprise|comprises|have|has)\b",
                q,
            )
        )

    @staticmethod
    def _heading_parts(heading: str) -> list[str]:
        return [re.sub(r"\s+", " ", part).strip().casefold() for part in heading.split("/") if part.strip()]

    @staticmethod
    def _heading_terms(value: str) -> set[str]:
        stop = {
            "a", "an", "and", "are", "be", "can", "do", "does", "for",
            "how", "i", "if", "in", "is", "it", "of", "on", "or", "the",
            "this", "to", "use", "using", "what", "when", "with", "you", "your",
        }
        output: set[str] = set()
        for token in re.findall(r"[a-z][a-z0-9-]*", (value or "").lower()):
            if token in stop or len(token) < 3:
                continue
            if token.endswith("ing") and len(token) > 5:
                token = token[:-3]
            elif token.endswith("es") and len(token) > 4:
                token = token[:-2]
            elif token.endswith("s") and len(token) > 3:
                token = token[:-1]
            output.add(token)
        for token in jieba.lcut(value or ""):
            token = token.strip()
            if len(token) >= 2 and contains_cjk(token):
                output.add(token)
        # Manual headings often use the generic product noun while users use a
        # market name or a common misspelling. Heading matching is lexical, so
        # add only identity-equivalent aliases.
        lowered = (value or "").casefold()
        if re.search(r"\b(?:jet\s*ski|jetski|jstski|waverunner)\b", lowered):
            output.add("watercraft")
        return output

    def _heading_recall(self, query: str, top_n: int, allowed_doc_ids: list[int] | None = None) -> list[int]:
        """Recall by complete secondary/tertiary heading paths before RRF."""
        query_terms = self._heading_terms(query)
        if not query_terms:
            return []
        doc_ids = allowed_doc_ids if allowed_doc_ids is not None else range(len(self.retrieval_chunks))
        scored: list[tuple[float, int]] = []
        normalized_query = re.sub(r"\s+", " ", (query or "").strip().casefold())
        for doc_id in doc_ids:
            heading = str(self.retrieval_chunks[doc_id].get("heading") or "")
            parts = self._heading_parts(heading)
            heading_terms = self._heading_terms(heading)
            hits = len(query_terms & heading_terms)
            if hits == 0:
                continue
            coverage = hits / max(1, len(query_terms))
            precision = hits / max(1, len(heading_terms))
            exact_component = any(len(part) >= 3 and part in normalized_query for part in parts)
            leaf_hits = len(query_terms & self._heading_terms(parts[-1] if parts else ""))
            score = coverage * 0.56 + precision * 0.28 + min(leaf_hits, 2) * 0.08 + (0.45 if exact_component else 0.0)
            if hits >= 2 or exact_component or coverage >= 0.25:
                scored.append((score, int(doc_id)))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc_id for _score, doc_id in scored[:top_n]]

    def _exact_leaf_heading_anchor(
        self,
        query: str,
        candidate_ids: list[int],
        *,
        trusted_ids: set[int],
    ) -> list[int]:
        """Return one coarse-validated candidate whose leaf topic matches query."""

        query_terms = self._heading_terms(query)
        scored: list[tuple[int, int, int]] = []
        for order, doc_id in enumerate(candidate_ids):
            if doc_id not in trusted_ids:
                continue
            heading = self.retrieval_chunks[doc_id].get("heading", "")
            component_matches: list[int] = []
            for component in heading.split("/"):
                component_terms = self._heading_terms(component)
                if component_terms and component_terms.issubset(query_terms):
                    component_matches.append(len(component_terms))
            if component_matches:
                scored.append((max(component_matches), -order, doc_id))
        if not scored:
            return []
        scored.sort(reverse=True)
        return [scored[0][2]]

    def keyword_search(
        self,
        keywords: list[str],
        top_k: int = 8,
        products: list[str] | None = None,
        semantic_query: str = "",
    ) -> tuple[list[SearchResult], int]:
        """兼容旧接口：内部复用统一检索。"""
        return self.search_manual(
            keywords,
            semantic_query=semantic_query,
            original_query="",
            top_k=top_k,
            products=products,
        )

    def vector_search(
        self,
        query: str,
        top_k: int = 8,
        products: list[str] | None = None,
    ) -> tuple[list[SearchResult], int]:
        """兼容旧接口：内部复用统一检索。"""
        keywords = tokenize_mixed(query)
        return self.search_manual(
            keywords,
            semantic_query=query,
            original_query="",
            top_k=top_k,
            products=products,
        )

    def _filter_by_products(self, doc_ids: list[int], products: list[str]) -> list[int]:
        """按产品名过滤候选 chunk。"""
        allowed = set()
        for p in products:
            allowed.update(self.product_chunk_ids.get(p, []))
        return [doc_id for doc_id in doc_ids if doc_id in allowed]

    def _reorder_by_lang(self, query: str, doc_ids: list[int]) -> list[int]:
        """全库召回时按问题语言重排：同语言 chunk 优先，跨语言保留兜底。

        判定语言用产品名而非 chunk lang 字段（后者不可靠 — Earphones 等英文产品被标 zh）。
        中文产品名以"手册"结尾；其余视为英文产品。
        软优先：稳定按 (lang_match_priority, original_rank) 排序，rerank 仍能让
        跨语言的高相关章节回到 top。
        """
        if not doc_ids:
            return doc_ids
        question_is_zh = contains_cjk(query)
        scored: list[tuple[int, int, int]] = []
        for rank, doc_id in enumerate(doc_ids):
            product = self.retrieval_chunks[doc_id].get("product", "")
            product_is_zh = product.endswith("手册")
            priority = 0 if product_is_zh == question_is_zh else 1
            scored.append((priority, rank, doc_id))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [doc_id for _, _, doc_id in scored]

    def _build_results(
        self,
        doc_ids: list[int],
        *,
        evidence_roles: dict[int, str] | None = None,
        relevance_metadata: dict[int, dict[str, float | int | str | None]] | None = None,
    ) -> list[SearchResult]:
        """把命中的 chunk id 转成 agent 可读证据。

        默认返回完整 parent section 的正文和图片，同时在 source 中保留实际命中的 chunk，方便 trace 解释“为什么召回到这一节”。
        """
        evidence_roles = evidence_roles or {}
        relevance_metadata = relevance_metadata or {}
        results = []
        for rank, doc_id in enumerate(doc_ids, start=1):
            chunk = self.retrieval_chunks[doc_id]
            section = self.section_lookup.get((chunk["product"], chunk["parent_section_id"]))

            use_parent = RETURN_PARENT_SECTION and section is not None
            return_text = section["text"] if use_parent else chunk["text"]
            return_pics = (
                section.get("evidence_pics", section.get("pics", []))
                if use_parent
                else list(dict.fromkeys(
                    (chunk.get("pics") or []) + (chunk.get("linked_pics") or [])
                ))
            )
            return_heading = section.get("heading", chunk["heading"]) if use_parent else chunk["heading"]

            relevance = relevance_metadata.get(doc_id, {})
            score = float(relevance.get("combined_relevance", 1.0 / rank))
            results.append(
                SearchResult(
                    chunk_id=doc_id,
                    product=chunk["product"],
                    heading=return_heading,
                    text=return_text,
                    pics=return_pics,
                    score=score,
                    source={
                        "matched_chunk_id": chunk.get("chunk_id", doc_id),
                        "matched_chunk_text": chunk["text"],
                        "matched_chunk_pics": chunk["pics"],
                        "matched_subchunk_id": chunk.get("subchunk_id"),
                        "matched_split_kind": chunk.get("split_kind"),
                        "evidence_role": evidence_roles.get(doc_id, "ranked"),
                        "document_order": int(chunk.get("parent_section_id", doc_id)),
                        "parent_section_id": chunk["parent_section_id"],
                        "source_section_ids": chunk.get("source_section_ids", []),
                        "section_summary": section["summary"] if section else "",
                        "section_tags": section["tags"] if section else [],
                        "section_text": section["text"] if section else "",
                        "section_pics": (
                            section.get("evidence_pics", section.get("pics", []))
                            if section else []
                        ),
                        "physical_section_pics": (section.get("pics") or []) if section else [],
                        "linked_section_pics": (section.get("linked_pics") or []) if section else [],
                        "fact_linked_section_pics": (section.get("fact_linked_pics") or []) if section else [],
                        "fact_links": (section.get("fact_links") or {}) if section else {},
                        "concept_linked_section_pics": (section.get("concept_linked_pics") or []) if section else [],
                        "concept_links": (section.get("concept_links") or {}) if section else {},
                        "section_heading": (section.get("heading") or "") if section else "",
                        "return_mode": "parent_section" if use_parent else "matched_chunk",
                        "relevance": relevance,
                    },
                )
            )
        return results

    def _sparse_recall(self, query: str, top_n: int, allowed_doc_ids: list[int] | None = None) -> list[int]:
        assert self.bm25 is not None
        query_tokens = tokenize_mixed(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        if allowed_doc_ids is not None:
            ranked_ids = sorted(
                allowed_doc_ids,
                key=lambda idx: float(scores[idx]),
                reverse=True,
            )
        else:
            ranked_ids = [int(idx) for idx in np.argsort(scores)[::-1]]
        return ranked_ids[:top_n]

    def _dense_recall(
        self,
        query: str,
        top_n: int,
        allowed_doc_ids: list[int] | None = None,
        dense_state: dict[str, object] | None = None,
    ) -> list[int]:
        assert self.dense_index is not None
        assert self.dense_vectors is not None
        if dense_state is not None:
            dense_state.clear()
            dense_state.update({"attempted": True, "available": False, "query": query})
        dense_cache_key = (
            self._index_generation,
            self.embedding_model,
            self._normalize_cache_text(query),
        )
        cached_query_array = self._cache_get(
            self._dense_query_cache,
            dense_cache_key,
            "dense",
        )
        if cached_query_array is None:
            try:
                query_vector = self.client.embed_texts([query], self.embedding_model)[0]
            except Exception as exc:
                # Keep this failure local to the current search.  The next
                # request must attempt Dense again; do not poison the worker.
                if dense_state is not None:
                    dense_state["error"] = str(exc)
                print(f"dense query embedding unavailable for this request; falling back to BM25: {exc}")
                return []
            query_array = self._l2_normalize(np.asarray([query_vector], dtype=np.float32))
            self._cache_put(
                self._dense_query_cache,
                dense_cache_key,
                query_array.copy(),
                DENSE_QUERY_CACHE_SIZE,
            )
        else:
            query_array = cached_query_array
        if dense_state is not None:
            dense_state.update({"available": True, "query_array": query_array})
        if allowed_doc_ids is not None:
            if not allowed_doc_ids:
                return []
            allowed_vectors = self.dense_vectors[np.asarray(allowed_doc_ids, dtype=np.int64)]
            scores = allowed_vectors @ query_array[0]
            top_idx = np.argsort(scores)[::-1][:top_n]
            return [int(allowed_doc_ids[int(i)]) for i in top_idx]
        _, indices = self.dense_index.search(query_array, top_n)
        return [int(idx) for idx in indices[0] if idx >= 0]

    def _rrf_merge(self, ranked_lists: list[list[int]], top_n: int, k: int = 60) -> list[int]:
        """Reciprocal Rank Fusion：融合 BM25 与 dense 的排名而不依赖分数同尺度。

        BM25 分数、向量相似度和不同产品子集的分布不可直接相加；RRF 只看名次，能稳定提升互补召回。
        """
        scores: dict[int, float] = {}
        first_seen: dict[int, tuple[int, int]] = {}
        for list_idx, ranked in enumerate(ranked_lists):
            for rank, doc_id in enumerate(ranked):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
                first_seen.setdefault(doc_id, (list_idx, rank))
        ordered = sorted(
            scores.items(),
            key=lambda item: (-item[1], first_seen[item[0]][0], first_seen[item[0]][1]),
        )
        return [doc_id for doc_id, _score in ordered[:top_n]]

    def _rerank_candidates(self, query: str, candidate_ids: list[int], top_n: int) -> list[int]:
        """对 RRF 候选做 cross-encoder rerank，失败时可解释地回退原排序。

        rerank 是精排层，不改变召回池；线上偶发 5xx/超时时记录 fallback，避免单个上游错误中断整批提交。
        """
        if not candidate_ids:
            return []
        if not self.rerank_enabled:
            return candidate_ids[:top_n]

        rerank_cache_key = (
            self._index_generation,
            getattr(self.rerank_client, "model", DEFAULT_RERANK_MODEL),
            self._normalize_cache_text(query),
            tuple(candidate_ids),
            int(top_n),
        )
        cached_rerank = self._cache_get(
            self._rerank_cache,
            rerank_cache_key,
            "rerank",
        )
        if cached_rerank is not None:
            return list(cached_rerank)

        documents = [build_rerank_text(self.retrieval_chunks[doc_id]) for doc_id in candidate_ids]
        rerank_elapsed = None
        try:
            t0 = time.time()
            ranked = self.rerank_client.rerank(query=query, documents=documents, top_n=min(top_n, len(documents)))
            rerank_elapsed = time.time() - t0
            if RERANK_TIMING_LOG_PATH:
                payload = {
                    "ts": time.time(),
                    "qid": getattr(_RERANK_CONTEXT, "qid", None) or os.getenv("CURRENT_QID"),
                    "query": query,
                    "top_n": top_n,
                    "candidate_count": len(candidate_ids),
                    "document_count": len(documents),
                    "elapsed": rerank_elapsed,
                }
                with _RERANK_FALLBACK_LOCK:
                    RERANK_TIMING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with RERANK_TIMING_LOG_PATH.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except RerankError as exc:
            print(f"rerank 不可用，回退到 RRF 排序: {exc}")
            if RERANK_FALLBACK_LOG_PATH:
                payload = {
                    "ts": time.time(),
                    "qid": getattr(_RERANK_CONTEXT, "qid", None) or os.getenv("CURRENT_QID"),
                    "query": query,
                    "top_n": top_n,
                    "candidate_count": len(candidate_ids),
                    "error": str(exc),
                }
                with _RERANK_FALLBACK_LOCK:
                    RERANK_FALLBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with RERANK_FALLBACK_LOG_PATH.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return candidate_ids[:top_n]

        reranked_ids: list[int] = []
        for item in ranked:
            if 0 <= item.index < len(candidate_ids):
                reranked_ids.append(candidate_ids[item.index])

        if len(reranked_ids) < top_n:
            seen = set(reranked_ids)
            reranked_ids.extend(doc_id for doc_id in candidate_ids if doc_id not in seen)

        reranked_ids = reranked_ids[:top_n]
        self._cache_put(
            self._rerank_cache,
            rerank_cache_key,
            list(reranked_ids),
            RERANK_CACHE_SIZE,
        )
        return reranked_ids

    def _apply_rerank_threshold(self, doc_ids: list[int], query: str) -> list[int]:
        """兼容旧调用点：固定阈值已停用，当前直接原样返回。"""
        return doc_ids

    def _result_dedup_key(self, doc_id: int) -> str:
        chunk = self.retrieval_chunks[doc_id]
        product = (chunk.get("product") or "").strip().lower()
        parent = str(chunk.get("parent_section_id") or "")
        return f"{product}::{parent}"

    def _embed_batch(self, batch_id: int, texts: list[str], model: str) -> tuple[int, list[list[float]]]:
        vectors = self.client.embed_texts(texts, model)
        return batch_id, vectors

    def _embed_corpus(self, texts: Iterable[str], model: str, batch_size: int) -> np.ndarray:
        texts = list(texts)

        # 切片展开：长文本（主要是含整表 caption_aux 的 chunk）按行切成 ≤380 字符的片，
        # 每片单独 embedding，最后按 owner 做 mean-pooling 还原为 1 chunk 1 向量。
        # 这样长 info_table 表格完整进 dense 向量（零删尾），且每片都在 ubatch 内不触发 500。
        seg_texts: list[str] = []
        owners: list[int] = []
        for i, text in enumerate(texts):
            segs = split_text_for_embedding(text) or [text]
            for s in segs:
                seg_texts.append(s)
                owners.append(i)

        batches: list[tuple[int, list[str]]] = []
        current_batch: list[str] = []
        for seg in seg_texts:
            current_batch.append(seg)
            if len(current_batch) >= batch_size:
                batches.append((len(batches), current_batch))
                current_batch = []
        if current_batch:
            batches.append((len(batches), current_batch))

        ordered_vectors: list[list[list[float]] | None] = [None] * len(batches)

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = {
                executor.submit(self._embed_batch, batch_id, batch_texts, model): batch_id
                for batch_id, batch_texts in batches
            }
            with tqdm(total=len(batches), desc="Embedding", unit="batch") as pbar:
                for future in as_completed(futures):
                    batch_id, vectors = future.result()
                    ordered_vectors[batch_id] = vectors
                    pbar.update(1)

        seg_vectors: list[list[float]] = []
        for batch_vectors in ordered_vectors:
            if batch_vectors is None:
                raise RuntimeError(f"{model} 存在未完成 batch，索引构建中断。")
            seg_vectors.extend(batch_vectors)

        seg_arr = np.asarray(seg_vectors, dtype=np.float32)
        dim = seg_arr.shape[1]
        out = np.zeros((len(texts), dim), dtype=np.float32)
        cnt = np.zeros(len(texts), dtype=np.float32)
        for owner, vec in zip(owners, seg_arr):
            out[owner] += vec
            cnt[owner] += 1.0
        cnt = np.clip(cnt, 1.0, None)
        out /= cnt[:, None]  # mean-pooling；L2 归一化在 build_index 后续统一做
        return out

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return vectors / norms


def format_results(results: list[SearchResult]) -> str:
    lines = []
    for i, item in enumerate(results, start=1):
        lines.append(f"[{i}] {item.product} / {item.heading}")
        lines.append(f"chunk_id={item.chunk_id} pics={item.pics}")
        lines.append(item.text[:280].replace("\n", " ") + ("..." if len(item.text) > 280 else ""))
        lines.append("")
    return "\n".join(lines).strip()

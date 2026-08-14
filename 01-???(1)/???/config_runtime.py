"""V6 初赛提交版默认运行配置。

该文件用于降低评审启动成本：`api_server.py` / `llm_router.py` 会在启动时
调用 `apply_default_env()`，把这里的配置写入尚未设置的环境变量。

如需替换上游 key、模型或端点，直接改本文件，或在启动前通过环境变量覆盖。
"""
from __future__ import annotations

import os


DEFAULT_ENV = {
    # /chat Bearer Token。smoke_test.sh 默认使用这个 token。
    "KAFU_API_TOKEN": "sk-datafountain-demo-0608",

    # 在线接口超时：公网/评审环境保守放宽，减少上游波动导致的误判。
    "CHAT_TIMEOUT_S": "300",
    # Keep a safety limit for a genuinely stuck upstream, but do not cut a
    # normal answer off at the old 20-second web deadline.
    "CHAT_GENERATION_TIMEOUT_S": "60",
    "CHAT_TIMEOUT_RESERVE_S": "0.75",
    # Visual pre-routing, intent classification, and grounded generation can
    # legitimately require several upstream model calls in one request.
    "CHAT_MULTIMODAL_TIMEOUT_S": "180",
    "CHAT_MAX_IMAGES": "3",
    "CHAT_MAX_IMAGE_BYTES": str(5 * 1024 * 1024),
    # 决赛题允许将公网图片/内容页链接直接写入 question。服务端仅在该
    # 开关开启时解析用户显式提供的 HTTP(S) 媒体，并使用 SSRF 防护与大小限制。
    "CHAT_AUTO_FETCH_QUESTION_MEDIA": "1",
    "CHAT_REMOTE_MEDIA_TIMEOUT_S": "15",
    # 有图片时先抽取规范产品候选、可见对象和检索意图，供产品路由使用；
    # 最终结论仍必须经手册检索证据验证。
    "CHAT_VISUAL_PREROUTE": "1",

    # 无 qid 在线场景的 service / tech 二分类路由器。使用与主回答链路
    # 协议一致的 GPT-5.5 Responses API，避免上游分类器失效后退化到正则。
    "CLASSIFIER_BASE_URL": "https://spatialai.vip",
    "CLASSIFIER_API_KEY": "replace-with-classifier-key",
    "CLASSIFIER_MODEL": "gpt-5.6-luna",
    "CLASSIFIER_WIRE_API": "responses",
    "CLASSIFIER_TIMEOUT_S": "12",
    "CLASSIFIER_MAX_TOKENS": "4",

    # 主回答模型：自建 OpenAI-compatible Responses API 端点。
    "SILICONFLOW_BASE_URL": "https://spatialai.vip",
    "SILICONFLOW_API_KEY": "replace-with-generation-key",
    "SILICONFLOW_MODEL": "gpt-5.6-terra",
    "SILICONFLOW_WIRE_API": "responses",
    "SILICONFLOW_ONLY": "1",
    "SILICONFLOW_MAX_CONCURRENCY": "3",
    "AGENT_MAX_TOKENS": "8192",
    "LLM_TIMEOUT_SECONDS": "30",
    # Only used after the evidence anomaly gate fires; main answer latency stays 30s.
    "EVIDENCE_SELECTOR_TIMEOUT_SECONDS": "75",
    # Never convert one slow model request into three serial 30-second calls.
    # A caller that needs retries must opt in explicitly.
    "LLM_TRANSIENT_RETRY_ATTEMPTS": "1",
    # Under load, return a fast busy error instead of waiting indefinitely for
    # a provider concurrency slot that may be held by an already-timed-out user.
    "LLM_ROUTE_QUEUE_TIMEOUT_SECONDS": "2",

    # 检索：默认使用硅基流动远程 embedding / rerank，评审不需要本地启动 8091/8090。
    "EMBEDDING_BASE_URL": "https://api.siliconflow.cn/v1",
    "EMBEDDING_API_KEY": "replace-with-embedding-key",
    "EMBEDDING_MODEL": "Pro/BAAI/bge-m3",
    "EMBEDDING_MAX_CONCURRENCY": "4",
    "RERANK_BASE_URL": "https://api.siliconflow.cn/v1",
    "RERANK_API_KEY": "replace-with-rerank-key",
    "RERANK_MODEL_ALIAS": "BAAI/bge-reranker-v2-m3",
    "RERANK_ENABLED": "1",

    # chunk 命中后返回完整 parent section。
    "RETURN_PARENT_SECTION": "1",
    # Keep all reranked related evidence available to the answer model.  Set
    # to "0" only for a deliberately constrained deployment.
    "DISABLE_GENERATION_EVIDENCE_BUDGET": "1",
}


def apply_default_env() -> None:
    """Apply delivery defaults without overriding variables already set by the caller."""
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)

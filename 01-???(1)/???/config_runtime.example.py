"""Safe configuration template for RAGv6.

Copy this file to ``config_runtime.py`` on a deployment machine and provide
real credentials through environment variables or a local secret store.
``config_runtime.py`` is intentionally excluded from Git.
"""
from __future__ import annotations

import os


DEFAULT_ENV = {
    "KAFU_API_TOKEN": "change-me",
    "CHAT_TIMEOUT_S": "300",
    "CHAT_MULTIMODAL_TIMEOUT_S": "60",
    "CHAT_MAX_IMAGES": "3",
    "CHAT_MAX_IMAGE_BYTES": str(5 * 1024 * 1024),
    "CHAT_AUTO_FETCH_QUESTION_MEDIA": "1",
    "CHAT_REMOTE_MEDIA_TIMEOUT_S": "15",
    "CHAT_VISUAL_PREROUTE": "1",
    "CLASSIFIER_BASE_URL": "https://your-provider.example/v1",
    "CLASSIFIER_API_KEY": "change-me",
    "CLASSIFIER_MODEL": "your-classifier-model",
    "CLASSIFIER_WIRE_API": "responses",
    "EMBEDDING_BASE_URL": "https://your-provider.example/v1",
    "EMBEDDING_API_KEY": "change-me",
    "EMBEDDING_MODEL": "Pro/BAAI/bge-m3",
    "RERANK_BASE_URL": "https://your-provider.example/v1",
    "RERANK_API_KEY": "change-me",
    "RERANK_MODEL_ALIAS": "BAAI/bge-reranker-v2-m3",
    "RERANK_ENABLED": "1",
    "SILICONFLOW_BASE_URL": "https://your-provider.example/v1",
    "SILICONFLOW_API_KEY": "change-me",
    "SILICONFLOW_MODEL": "your-generation-model",
    "SILICONFLOW_WIRE_API": "responses",
    "RETURN_PARENT_SECTION": "1",
}


def apply_default_env() -> None:
    """Apply defaults without overriding deployment environment variables."""
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)

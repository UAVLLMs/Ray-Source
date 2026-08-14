"""Optional Redis/Garnet cache for retrieval intermediates and final results."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


class SharedRetrievalCache:
    def __init__(self) -> None:
        self.enabled = os.getenv("RETRIEVAL_REDIS_CACHE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0").strip()
        self.namespace = os.getenv("RETRIEVAL_CACHE_NAMESPACE", "raysource:retrieval:v1").strip()
        self.ttl = max(30, int(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS", "3600")))
        self._lock = threading.Lock()
        self._client = None
        self._last_error = ""
        self.hits = 0
        self.misses = 0
        self.errors = 0

    def _redis(self):
        if not self.enabled:
            return None
        with self._lock:
            if self._client is None:
                import redis

                self._client = redis.Redis.from_url(
                    self.url,
                    socket_connect_timeout=0.25,
                    socket_timeout=0.35,
                    health_check_interval=15,
                    decode_responses=False,
                )
            return self._client

    def _key(self, kind: str, key: tuple) -> str:
        canonical = json.dumps(key, ensure_ascii=False, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{self.namespace}:{kind}:{digest}"

    @staticmethod
    def _encode(value: Any) -> bytes:
        if isinstance(value, np.ndarray):
            payload = {
                "type": "ndarray",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
            }
        elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], list):
            payload = {
                "type": "search",
                "results": [asdict(item) if is_dataclass(item) else item for item in value[0]],
                "filtered": value[1],
            }
        else:
            payload = {"type": "json", "value": value}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _decode(raw: bytes, *, result_type=None):
        payload = json.loads(raw.decode("utf-8"))
        if payload["type"] == "ndarray":
            data = base64.b64decode(payload["data"])
            return np.frombuffer(data, dtype=np.dtype(payload["dtype"])).reshape(payload["shape"]).copy()
        if payload["type"] == "search":
            results = [result_type(**item) if result_type else item for item in payload["results"]]
            return results, int(payload["filtered"])
        return payload["value"]

    def get(self, kind: str, key: tuple, *, result_type=None):
        client = self._redis()
        if client is None:
            return None
        try:
            raw = client.get(self._key(kind, key))
            if raw is None:
                self.misses += 1
                return None
            self.hits += 1
            return self._decode(raw, result_type=result_type)
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            self._last_error = type(exc).__name__
            with self._lock:
                self._client = None
            return None

    def put(self, kind: str, key: tuple, value: Any) -> None:
        client = self._redis()
        if client is None:
            return
        try:
            client.set(self._key(kind, key), self._encode(value), ex=self.ttl)
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            self._last_error = type(exc).__name__
            with self._lock:
                self._client = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": "redis" if self.enabled else "disabled",
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "last_error": self._last_error,
        }


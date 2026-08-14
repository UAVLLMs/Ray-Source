"""Anthropic 路由：默认走 highspeed，额度耗尽后自动切到 token plan。"""

from __future__ import annotations

import os
import json
import time
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace

from dotenv import load_dotenv
import httpx

load_dotenv()

try:
    from config_runtime import apply_default_env

    apply_default_env()
except Exception:
    pass


@dataclass(frozen=True)
class AnthropicRoute:
    """一个可用 LLM 上游路由。

    protocol 区分 Anthropic Messages API、OpenAI Chat Completions 与 Responses API；其余调用方统一拿到 Anthropic 形状的 response，降低 agent 主循环复杂度。
    """
    name: str
    base_url: str
    api_key: str
    model: str
    protocol: str = "anthropic"


@dataclass
class RouteRuntimeState:
    active: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    circuit: str = "closed"
    opened_until: float = 0.0
    half_open_attempts: int = 0
    rate_limited: int = 0
    timeouts: int = 0
    server_errors: int = 0
    network_errors: int = 0
    auth_errors: int = 0
    total_latency_seconds: float = 0.0
    last_latency_seconds: float = 0.0
    last_failure_kind: str = ""
    last_failure_at: float = 0.0


_DISABLED_ROUTES: set[str] = set()
_ROUTE_LOCK = threading.Lock()
_ROUTE_SEMAPHORES: dict[str, threading.BoundedSemaphore] = {}
_ROUTE_RUNTIME: dict[str, RouteRuntimeState] = {}
_ROUTE_SELECTION_CURSOR = 0
# Long-lived transport clients keep TCP/TLS connections alive between requests.
# They are keyed by endpoint + credential because routes can point at different
# compatible providers.  A single httpx.Client is thread-safe.
_HTTP_CLIENT_LOCK = threading.RLock()
_HTTP_CLIENTS: dict[tuple[str, str], httpx.Client] = {}
_OPENAI_CLIENTS: dict[tuple[str, str], object] = {}

_DEFAULT_ROUTE_CONCURRENCY = {
    "highspeed": int(os.getenv("HIGHSPEED_MAX_CONCURRENCY", "20")),
    "token-plan": int(os.getenv("TOKEN_PLAN_MAX_CONCURRENCY", "2")),
    "siliconflow": int(os.getenv("SILICONFLOW_MAX_CONCURRENCY", "15")),
    "spatial-a": int(os.getenv("SPATIAL_PRIMARY_MAX_CONCURRENCY", os.getenv("SILICONFLOW_MAX_CONCURRENCY", "3"))),
    "spatial-b": int(os.getenv("SPATIAL_SECONDARY_MAX_CONCURRENCY", os.getenv("SILICONFLOW_MAX_CONCURRENCY", "3"))),
}
_DEFAULT_LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
_TRANSIENT_RETRY_ATTEMPTS = max(1, int(os.getenv("LLM_TRANSIENT_RETRY_ATTEMPTS", "3")))
_DEFAULT_ROUTE_QUEUE_TIMEOUT = max(0.0, float(os.getenv("LLM_ROUTE_QUEUE_TIMEOUT_SECONDS", "2")))
_CIRCUIT_FAILURE_THRESHOLD = max(1, int(os.getenv("LLM_CIRCUIT_FAILURE_THRESHOLD", "2")))
_CIRCUIT_NETWORK_OPEN_SECONDS = max(1.0, float(os.getenv("LLM_CIRCUIT_NETWORK_OPEN_SECONDS", "12")))
_CIRCUIT_RATE_LIMIT_OPEN_SECONDS = max(1.0, float(os.getenv("LLM_CIRCUIT_RATE_LIMIT_OPEN_SECONDS", "20")))
_CIRCUIT_AUTH_OPEN_SECONDS = max(1.0, float(os.getenv("LLM_CIRCUIT_AUTH_OPEN_SECONDS", "60")))
_REQUEST_REASONING_EFFORT: ContextVar[str] = ContextVar("request_reasoning_effort", default="")


class LLMRouteBusyError(RuntimeError):
    """The provider is saturated; callers must fail fast instead of queuing forever."""


def _route_client_key(route: AnthropicRoute) -> tuple[str, str]:
    return (route.base_url.rstrip("/"), route.api_key)


def _get_http_client(route: AnthropicRoute) -> httpx.Client:
    """Return the route's persistent keep-alive HTTP transport."""
    key = _route_client_key(route)
    with _HTTP_CLIENT_LOCK:
        client = _HTTP_CLIENTS.get(key)
        if client is None or client.is_closed:
            client = httpx.Client(
                limits=httpx.Limits(max_connections=40, max_keepalive_connections=20, keepalive_expiry=60.0),
                timeout=httpx.Timeout(_DEFAULT_LLM_TIMEOUT),
                headers={"User-Agent": "lanqun-rag/1.0"},
            )
            _HTTP_CLIENTS[key] = client
        return client


def _get_openai_client(route: AnthropicRoute):
    """Cache OpenAI SDK clients so DeepSeek reuses the same HTTP transport."""
    from openai import OpenAI

    key = _route_client_key(route)
    with _HTTP_CLIENT_LOCK:
        client = _OPENAI_CLIENTS.get(key)
        if client is None:
            client = OpenAI(
                base_url=route.base_url,
                api_key=route.api_key,
                http_client=_get_http_client(route),
            )
            _OPENAI_CLIENTS[key] = client
        return client


def close_persistent_clients() -> None:
    """Release pooled sockets during FastAPI shutdown (safe to call repeatedly)."""
    with _HTTP_CLIENT_LOCK:
        clients = list(_HTTP_CLIENTS.values())
        _HTTP_CLIENTS.clear()
        _OPENAI_CLIENTS.clear()
    for client in clients:
        try:
            client.close()
        except Exception:
            pass


def warmup_route_clients() -> list[str]:
    """Create long-lived provider clients before the first real request.

    This deliberately performs no billable inference call.  It initializes the
    shared HTTP transports and SDK wrappers so normal requests reuse the same
    pools from process start.
    """
    warmed: list[str] = []
    routes = list(get_routes())
    # Qwen VL is model-specific and therefore not part of get_routes(). Warm
    # its transport as well so the first image request does not initialize a
    # new HTTP/OpenAI client.
    try:
        qwen_routes = _routes_for_model(_norm(os.getenv("VISUAL_PREROUTE_MODEL")) or "qwen-vl-max")
    except Exception:
        qwen_routes = []
    known = {(route.name, route.base_url, route.api_key) for route in routes}
    for route in qwen_routes:
        key = (route.name, route.base_url, route.api_key)
        if key not in known:
            routes.append(route)
            known.add(key)
    for route in routes:
        _get_http_client(route)
        if route.protocol in {"openai", "responses"}:
            _get_openai_client(route)
        warmed.append(route.name)
    return warmed


def set_request_reasoning_effort(value: str):
    """Set per-request reasoning effort and return a token for restoration."""
    normalized = _norm(value).lower()
    if normalized not in {"none", "low", "medium", "high"}:
        normalized = "medium"
    return _REQUEST_REASONING_EFFORT.set(normalized)


def _norm(text: str | None) -> str:
    return (text or "").strip()


def _build_route(prefix: str, default_name: str, default_model: str) -> AnthropicRoute | None:
    default_base_url = ""
    default_api_key = ""
    if prefix == "SILICONFLOW":
        default_model = "gpt-5.5-openai-compact"
    base_url = _norm(os.getenv(f"{prefix}_BASE_URL", default_base_url))
    api_key = _norm(os.getenv(f"{prefix}_API_KEY", default_api_key))
    model = _norm(os.getenv(f"{prefix}_MODEL")) or default_model
    if not base_url or not api_key or not model:
        return None
    return AnthropicRoute(
        name=default_name,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )


def _build_openai_route(prefix: str, default_name: str, default_model: str) -> AnthropicRoute | None:
    route = _build_route(prefix, default_name, default_model)
    if route is None:
        return None
    return AnthropicRoute(
        name=route.name,
        base_url=route.base_url,
        api_key=route.api_key,
        model=route.model,
        protocol=_norm(os.getenv(f"{prefix}_WIRE_API")) or "openai",
    )


def _env_flag(name: str) -> bool:
    return _norm(os.getenv(name)).lower() in {"1", "true", "yes", "on"}


def _build_spatial_secondary(primary: AnthropicRoute | None) -> AnthropicRoute | None:
    """Build the second Spatial channel from server-only environment values."""
    if primary is None:
        return None
    api_key = _norm(os.getenv("SPATIAL_SECONDARY_API_KEY"))
    if not api_key:
        return None
    return AnthropicRoute(
        name="spatial-b",
        base_url=_norm(os.getenv("SPATIAL_SECONDARY_BASE_URL")) or primary.base_url,
        api_key=api_key,
        model=_norm(os.getenv("SPATIAL_SECONDARY_MODEL")) or primary.model,
        protocol=_norm(os.getenv("SPATIAL_SECONDARY_WIRE_API")) or primary.protocol,
    )


def get_routes() -> list[AnthropicRoute]:
    """返回按优先级排序的可用路由。"""
    routes: list[AnthropicRoute] = []

    siliconflow = _build_openai_route(
        prefix="SILICONFLOW",
        default_name="spatial-a",
        default_model="Qwen/Qwen3.6-27B",
    )
    spatial_secondary = _build_spatial_secondary(siliconflow)
    openai_route = _build_openai_route(
        prefix="OPENAI",
        default_name="openai",
        default_model="gpt-5",
    )
    if siliconflow is not None and _env_flag("SILICONFLOW_ONLY"):
        return [route for route in (siliconflow, spatial_secondary) if route is not None]
    if siliconflow is not None and not _env_flag("DISABLE_SILICONFLOW"):
        routes.append(siliconflow)
        if spatial_secondary is not None:
            routes.append(spatial_secondary)

    if not _env_flag("DISABLE_HIGHSPEED"):
        highspeed = _build_route(
            prefix="HIGHSPEED",
            default_name="highspeed",
            default_model="MiniMax-M2.7-highspeed",
        )
        if highspeed is not None:
            routes.append(highspeed)

    token_plan = _build_route(
        prefix="ANTHROPIC",
        default_name="token-plan",
        default_model="MiniMax-M2.7",
    )
    if token_plan is not None:
        same_as_existing = any(
            r.base_url == token_plan.base_url
            and r.api_key == token_plan.api_key
            and r.model == token_plan.model
            for r in routes
        )
        if not same_as_existing:
            routes.append(token_plan)

    if openai_route is not None and openai_route not in routes:
        routes.append(openai_route)

    return routes


def get_default_model() -> str:
    routes = get_routes()
    if routes:
        return routes[0].model
    return "gpt-5.5"


def _runtime_state(route_name: str) -> RouteRuntimeState:
    return _ROUTE_RUNTIME.setdefault(route_name, RouteRuntimeState())


def _exception_status_code(exc: Exception) -> int | None:
    raw = getattr(exc, "status_code", None)
    if raw is None:
        raw = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _failure_kind(exc: Exception) -> str:
    if isinstance(exc, LLMRouteBusyError):
        return "busy"
    status_code = _exception_status_code(exc)
    text = str(exc).lower()
    if status_code == 429 or "rate limit" in text or "too many requests" in text:
        return "rate_limit"
    if status_code in {401, 403} or "unauthorized" in text or "invalid api key" in text:
        return "auth"
    if status_code == 402 or any(marker in text for marker in (
        "insufficient_quota", "quota", "credit", "余额", "额度", "配额"
    )):
        return "quota"
    if status_code is not None and status_code >= 500:
        return "server"
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)) or any(marker in text for marker in (
        "readtimeout", "read timeout", "timeout", "timed out"
    )):
        return "timeout"
    if isinstance(exc, httpx.TransportError) or any(marker in text for marker in (
        "connection error", "apiconnectionerror", "remoteprotocolerror", "protocol error",
        "connection aborted", "connection reset", "server disconnected", "temporary failure",
    )):
        return "network"
    return "request"


def _should_fallback(exc: Exception) -> bool:
    """Fail over only when another credential/transport can plausibly succeed."""
    return _failure_kind(exc) in {
        "busy", "rate_limit", "auth", "quota", "server", "timeout", "network"
    }


def _should_retry_same_route(exc: Exception) -> bool:
    """判断是否在同一路由内重试；仅覆盖网络抖动、超时、连接断开等瞬时错误。"""
    return _failure_kind(exc) in {"server", "timeout", "network"}


def _ordered_available_routes(routes: list[AnthropicRoute]) -> list[AnthropicRoute]:
    """Least-loaded ordering with round-robin tie breaking and circuit filtering."""
    global _ROUTE_SELECTION_CURSOR
    now = time.monotonic()
    with _ROUTE_LOCK:
        configured = [route for route in routes if route.name not in _DISABLED_ROUTES]
        available: list[AnthropicRoute] = []
        for route in configured:
            state = _runtime_state(route.name)
            if state.circuit == "open":
                if now < state.opened_until:
                    continue
                if state.active:
                    continue
                state.circuit = "half_open"
                state.half_open_attempts += 1
            if state.circuit == "half_open" and state.active:
                continue
            available.append(route)
        if not available:
            return []
        cursor = _ROUTE_SELECTION_CURSOR % len(available)
        _ROUTE_SELECTION_CURSOR += 1
        tie_order = {
            route.name: (index - cursor) % len(available)
            for index, route in enumerate(available)
        }
        return sorted(
            available,
            key=lambda route: (
                _runtime_state(route.name).active / get_route_concurrency(route.name),
                _runtime_state(route.name).active,
                _runtime_state(route.name).consecutive_failures,
                tie_order[route.name],
            ),
        )


def _begin_route(route: AnthropicRoute) -> bool:
    now = time.monotonic()
    with _ROUTE_LOCK:
        state = _runtime_state(route.name)
        if state.circuit == "open":
            if now < state.opened_until or state.active:
                return False
            state.circuit = "half_open"
            state.half_open_attempts += 1
        if state.circuit == "half_open" and state.active:
            return False
        state.active += 1
        return True


def _route_succeeded(route: AnthropicRoute, elapsed: float) -> None:
    with _ROUTE_LOCK:
        state = _runtime_state(route.name)
        state.successes += 1
        state.consecutive_failures = 0
        state.circuit = "closed"
        state.opened_until = 0.0
        state.last_failure_kind = ""
        state.last_latency_seconds = max(0.0, elapsed)
        state.total_latency_seconds += max(0.0, elapsed)


def _route_failed(route: AnthropicRoute, exc: Exception, elapsed: float) -> None:
    kind = _failure_kind(exc)
    now = time.monotonic()
    with _ROUTE_LOCK:
        state = _runtime_state(route.name)
        state.failures += 1
        state.consecutive_failures += 1
        state.last_failure_kind = kind
        state.last_failure_at = time.time()
        state.last_latency_seconds = max(0.0, elapsed)
        state.total_latency_seconds += max(0.0, elapsed)
        if kind == "rate_limit":
            state.rate_limited += 1
        elif kind == "timeout":
            state.timeouts += 1
        elif kind == "server":
            state.server_errors += 1
        elif kind == "network":
            state.network_errors += 1
        elif kind in {"auth", "quota"}:
            state.auth_errors += 1

        open_seconds = 0.0
        if kind == "rate_limit":
            open_seconds = _CIRCUIT_RATE_LIMIT_OPEN_SECONDS
        elif kind in {"auth", "quota"}:
            open_seconds = _CIRCUIT_AUTH_OPEN_SECONDS
        elif kind in {"server", "timeout", "network"} and (
            state.circuit == "half_open"
            or state.consecutive_failures >= _CIRCUIT_FAILURE_THRESHOLD
        ):
            open_seconds = _CIRCUIT_NETWORK_OPEN_SECONDS
        if open_seconds:
            state.circuit = "open"
            state.opened_until = now + open_seconds


def _end_route(route: AnthropicRoute) -> None:
    with _ROUTE_LOCK:
        state = _runtime_state(route.name)
        state.active = max(0, state.active - 1)


def route_runtime_snapshot() -> list[dict[str, object]]:
    """Return anonymous operational metrics; credentials and endpoints are omitted."""
    routes = get_routes()
    now = time.monotonic()
    with _ROUTE_LOCK:
        result = []
        for route in routes:
            state = _runtime_state(route.name)
            result.append({
                "channel": route.name,
                "model": route.model,
                "protocol": route.protocol,
                "concurrency_limit": get_route_concurrency(route.name),
                "active": state.active,
                "circuit": state.circuit,
                "open_remaining_seconds": round(max(0.0, state.opened_until - now), 3),
                "successes": state.successes,
                "failures": state.failures,
                "consecutive_failures": state.consecutive_failures,
                "rate_limited": state.rate_limited,
                "timeouts": state.timeouts,
                "server_errors": state.server_errors,
                "network_errors": state.network_errors,
                "auth_errors": state.auth_errors,
                "average_latency_seconds": round(
                    state.total_latency_seconds / max(1, state.successes + state.failures), 3
                ),
                "last_latency_seconds": round(state.last_latency_seconds, 3),
                "last_failure_kind": state.last_failure_kind,
                "last_failure_at": state.last_failure_at or None,
            })
        return result


def describe_routes() -> list[dict[str, object]]:
    """Public-safe route description with no credentials or endpoint URL."""
    return route_runtime_snapshot()


def _active_routes() -> list[AnthropicRoute]:
    return _ordered_available_routes(get_routes())


def _routes_for_model(model: str | None) -> list[AnthropicRoute]:
    """Keep DeepSeek isolated from the GPT route and use it for text-only models."""
    requested_model = _norm(model).lower()
    if requested_model.startswith("qwen"):
        qwen = _build_openai_route(
            prefix="QWEN",
            default_name="qwen-vl",
            default_model=_norm(model),
        )
        if qwen is None:
            raise RuntimeError("Qwen VL is not configured on this server")
        return _ordered_available_routes([qwen])
    if requested_model.startswith("deepseek-"):
        deepseek = _build_openai_route(
            prefix="DEEPSEEK",
            default_name="deepseek",
            default_model=_norm(model),
        )
        if deepseek is None:
            raise RuntimeError("DeepSeek is not configured on this server")
        return _ordered_available_routes([deepseek])
    if requested_model:
        matching_routes = [
            route for route in get_routes()
            if _norm(route.model).lower() == requested_model
        ]
        if matching_routes:
            return _ordered_available_routes(matching_routes)
    return _active_routes()


def get_route_concurrency(route_name: str) -> int:
    limit = _DEFAULT_ROUTE_CONCURRENCY.get(route_name, 1)
    return max(1, int(limit))


def _get_route_semaphore(route_name: str) -> threading.BoundedSemaphore:
    with _ROUTE_LOCK:
        sem = _ROUTE_SEMAPHORES.get(route_name)
        if sem is None:
            sem = threading.BoundedSemaphore(get_route_concurrency(route_name))
            _ROUTE_SEMAPHORES[route_name] = sem
        return sem


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "tool_result":
                    parts.append(str(item.get("content", "")))
                elif item.get("type") in {"image_url", "image"}:
                    parts.append("[用户上传图片]")
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _convert_multimodal_item_for_openai(item) -> dict | None:
    """把 Anthropic 风格 content block 转为 OpenAI-compatible 多模态 block。"""
    if not isinstance(item, dict):
        return {"type": "text", "text": _content_to_text(item)}
    item_type = item.get("type")
    if item_type == "text":
        return {"type": "text", "text": str(item.get("text", ""))}
    if item_type == "image_url":
        image_url = item.get("image_url")
        if isinstance(image_url, dict) and image_url.get("url"):
            return {"type": "image_url", "image_url": {"url": str(image_url["url"])}}
        if isinstance(image_url, str):
            return {"type": "image_url", "image_url": {"url": image_url}}
    return None


def _convert_messages_for_openai(system: str, messages: list) -> list[dict]:
    """把 agent 内部 Anthropic Messages 形状转换成 OpenAI Chat Completions 形状。

    需要同时处理三类差异：system 独立字段转 system message；assistant tool_use 转 tool_calls；user tool_result 转 tool 消息。若用户 content 含 image_url，则保留多模态数组而不是拍平成纯文本。
    """
    converted: list[dict] = [{"role": "system", "content": system}]
    pending_tool_ids: set[str] = set()

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    text_parts.append(getattr(block, "text", "") or "")
                elif block_type == "tool_use":
                    tool_id = getattr(block, "id", "") or f"call_{len(pending_tool_ids) + 1}"
                    pending_tool_ids.add(tool_id)
                    tool_calls.append({
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": getattr(block, "name", ""),
                            "arguments": json.dumps(
                                getattr(block, "input", {}) or {},
                                ensure_ascii=False,
                            ),
                        },
                    })
            converted_message: dict = {
                "role": "assistant",
                "content": "\n".join(p for p in text_parts if p) or None,
            }
            if tool_calls:
                converted_message["tool_calls"] = tool_calls
            converted.append(converted_message)
            continue

        if role == "user" and isinstance(content, list):
            user_parts: list[str] = []
            multimodal_parts: list[dict] = []
            has_multimodal_input = False
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    tool_id = str(item.get("tool_use_id", ""))
                    result_text = str(item.get("content", ""))
                    if tool_id in pending_tool_ids:
                        converted.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result_text,
                        })
                        pending_tool_ids.discard(tool_id)
                    else:
                        user_parts.append(result_text)
                        multimodal_parts.append({"type": "text", "text": result_text})
                else:
                    converted_item = _convert_multimodal_item_for_openai(item)
                    if converted_item is not None:
                        multimodal_parts.append(converted_item)
                        if converted_item.get("type") != "text":
                            has_multimodal_input = True
                    user_parts.append(_content_to_text(item))
            if multimodal_parts:
                if has_multimodal_input:
                    converted.append({"role": "user", "content": multimodal_parts})
                else:
                    converted.append({"role": "user", "content": "\n".join(p for p in user_parts if p)})
            continue

        converted.append({"role": role, "content": _content_to_text(content)})

    return converted


def _convert_tools_for_openai(tools: list | None) -> list[dict] | None:
    if tools is None:
        return None
    converted = []
    for tool in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return converted


def _convert_messages_for_responses(messages: list) -> list[dict]:
    """Convert the agent's Anthropic-shaped history to Responses API input items."""
    converted: list[dict] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    text_parts.append(getattr(block, "text", "") or "")
                elif block_type == "tool_use":
                    converted.append({
                        "type": "function_call",
                        "call_id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "arguments": json.dumps(getattr(block, "input", {}) or {}, ensure_ascii=False),
                    })
            if any(text_parts):
                converted.append({"role": "assistant", "content": "\n".join(text_parts)})
            continue
        if role == "user" and isinstance(content, list):
            input_parts: list[dict] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    converted.append({
                        "type": "function_call_output",
                        "call_id": str(item.get("tool_use_id", "")),
                        "output": str(item.get("content", "")),
                    })
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else image_url
                    if url:
                        input_parts.append({"type": "input_image", "image_url": str(url)})
                else:
                    text = _content_to_text(item)
                    if text:
                        input_parts.append({"type": "input_text", "text": text})
            if input_parts:
                converted.append({"role": "user", "content": input_parts})
            continue
        converted.append({"role": role, "content": _content_to_text(content)})
    return converted


def _openai_response_to_anthropic_shape(response) -> SimpleNamespace:
    """把 OpenAI response 包装成 response.content=[text/tool_use] 的 Anthropic 兼容形状。"""
    if not hasattr(response, 'choices'):
        print(f"[llm] API 返回非预期类型: type={type(response).__name__} repr={repr(response)[:300]}")
        raise AttributeError(f"API response has no 'choices' attribute, got {type(response).__name__}: {str(response)[:200]}")
    message = response.choices[0].message
    blocks = []
    content = getattr(message, "content", None)
    if content:
        blocks.append(SimpleNamespace(type="text", text=content))

    tool_calls = getattr(message, "tool_calls", None) or []
    for tool_call in tool_calls:
        function = tool_call.function
        raw_args = function.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except Exception:
            args = {}
        blocks.append(SimpleNamespace(
            type="tool_use",
            id=tool_call.id,
            name=function.name,
            input=args,
        ))
    return SimpleNamespace(content=blocks)


def _responses_response_to_anthropic_shape(response) -> SimpleNamespace:
    """把 OpenAI Responses API 输出包装成 agent 使用的 Anthropic 兼容形状。"""
    blocks = []
    for item in getattr(response, "output", None) or []:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            for content in getattr(item, "content", None) or []:
                if getattr(content, "type", None) == "output_text":
                    text = getattr(content, "text", "") or ""
                    if text:
                        blocks.append(SimpleNamespace(type="text", text=text))
        elif item_type == "function_call":
            raw_args = getattr(item, "arguments", "{}") or "{}"
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}
            blocks.append(SimpleNamespace(
                type="tool_use",
                id=getattr(item, "call_id", None) or getattr(item, "id", ""),
                name=getattr(item, "name", ""),
                input=args,
            ))
    return SimpleNamespace(content=blocks)


def _create_responses_message(
    *,
    route: AnthropicRoute,
    system: str,
    messages: list,
    max_tokens: int,
    tools: list | None,
    model: str | None,
    timeout: int | float | None,
):
    kwargs = {
        "model": model or route.model,
        "instructions": system,
        "input": _convert_messages_for_responses(messages),
        "max_output_tokens": max_tokens,
        "store": False,
    }
    effort = _REQUEST_REASONING_EFFORT.get() or _norm(os.getenv("OPENAI_REASONING_EFFORT"))
    if effort:
        kwargs["reasoning"] = {"effort": effort}
    openai_tools = _convert_tools_for_openai(tools)
    if openai_tools is not None:
        kwargs["tools"] = [tool["function"] | {"type": "function"} for tool in openai_tools]
        kwargs["tool_choice"] = "auto"
    request_timeout = timeout if timeout is not None else _DEFAULT_LLM_TIMEOUT
    response = _get_http_client(route).post(
        f"{route.base_url.rstrip('/')}/responses",
        json=kwargs,
        headers={"Authorization": f"Bearer {route.api_key}"},
        timeout=request_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    output = []
    for item in payload.get("output", []):
        content = [SimpleNamespace(**part) for part in item.get("content", [])]
        output.append(SimpleNamespace(**(item | {"content": content})))
    return _responses_response_to_anthropic_shape(SimpleNamespace(output=output))


def _create_responses_message_streaming(
    *,
    route: AnthropicRoute,
    system: str,
    messages: list,
    max_tokens: int,
    tools: list | None,
    model: str | None,
    timeout: int | float | None,
    on_delta=None,
):
    kwargs = {
        "model": model or route.model,
        "instructions": system,
        "input": _convert_messages_for_responses(messages),
        "max_output_tokens": max_tokens,
        "store": False,
        "stream": True,
    }
    effort = _REQUEST_REASONING_EFFORT.get() or _norm(os.getenv("OPENAI_REASONING_EFFORT"))
    if effort:
        kwargs["reasoning"] = {"effort": effort}
    openai_tools = _convert_tools_for_openai(tools)
    if openai_tools is not None:
        kwargs["tools"] = [tool["function"] | {"type": "function"} for tool in openai_tools]
        kwargs["tool_choice"] = "auto"
    request_timeout = timeout if timeout is not None else _DEFAULT_LLM_TIMEOUT
    completed_response: dict | None = None
    output_items: list[dict] = []
    ttft: float | None = None
    t0 = time.time()
    with _get_http_client(route).stream(
        "POST",
        f"{route.base_url.rstrip('/')}/responses",
        json=kwargs,
        headers={"Authorization": f"Bearer {route.api_key}", "Accept": "text/event-stream"},
        timeout=request_timeout,
    ) as raw_response:
        raw_response.raise_for_status()
        for line in raw_response.iter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                delta = event.get("delta") or ""
                if delta:
                    if ttft is None:
                        ttft = time.time() - t0
                    if on_delta is not None:
                        on_delta(delta)
            elif event_type == "response.output_item.done" and isinstance(event.get("item"), dict):
                output_items.append(event["item"])
            elif event_type == "response.completed":
                completed_response = event.get("response")
            elif event_type in {"response.failed", "error"}:
                raise RuntimeError(str(event.get("error") or event))
    payload = completed_response or {"output": output_items}
    output = []
    for item in payload.get("output", []) or output_items:
        content = [SimpleNamespace(**part) for part in item.get("content", [])]
        output.append(SimpleNamespace(**(item | {"content": content})))
    return _responses_response_to_anthropic_shape(SimpleNamespace(output=output)), ttft


def _thinking_extra_body(
    base_url: str | None,
    *,
    model: str | None = None,
    route_name: str | None = None,
) -> dict:
    """按端点选 thinking 开关写法。
    - DeepSeek（base_url 含 deepseek）：官方开关是 {"thinking": {"type": "enabled"/"disabled"}}，
      不吃 budget_tokens / enable_thinking；强度只接受 reasoning_effort=high/max（此处不强加）。
    - 其它 OpenAI 兼容端点：沿用旧逻辑（thinking={type,budget_tokens} 开 / enable_thinking=False 关）。
    由环境变量 LLM_ENABLE_THINKING 决定开关。"""
    is_deepseek = bool(
        "deepseek" in (base_url or "").lower()
        or (route_name or "").strip().lower() == "deepseek"
        or (model or "").strip().lower().startswith("deepseek-")
    )
    request_effort = _REQUEST_REASONING_EFFORT.get().strip().lower()
    thinking_enabled = (
        request_effort == "high"
        if is_deepseek and request_effort
        else _env_flag("LLM_ENABLE_THINKING")
    )
    if thinking_enabled:
        if is_deepseek:
            return {"thinking": {"type": "enabled"}}
        budget = int(os.getenv("LLM_THINKING_BUDGET_TOKENS", "4096"))
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    if is_deepseek:
        return {"thinking": {"type": "disabled"}}
    if "0-0.pro" in (base_url or "").lower():
        return {"enable_thinking": False, "thinking": {"type": "disabled"}}
    return {"enable_thinking": False}


def _create_openai_message(
    *,
    route: AnthropicRoute,
    system: str,
    messages: list,
    max_tokens: int,
    tools: list | None,
    model: str | None,
    timeout: int | float | None,
):
    client = _get_openai_client(route).with_options(
        timeout=timeout if timeout is not None else _DEFAULT_LLM_TIMEOUT,
        max_retries=0,
    )
    kwargs = {
        "model": model or route.model,
        "messages": _convert_messages_for_openai(system, messages),
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    kwargs["extra_body"] = _thinking_extra_body(
        route.base_url,
        model=model or route.model,
        route_name=route.name,
    )
    openai_tools = _convert_tools_for_openai(tools)
    if openai_tools is not None:
        kwargs["tools"] = openai_tools
        kwargs["tool_choice"] = "auto"
    return _openai_response_to_anthropic_shape(client.chat.completions.create(**kwargs))


def _create_message_streaming_on_route(
    *,
    route: AnthropicRoute,
    system: str,
    messages: list,
    max_tokens: int,
    tools: list | None = None,
    model: str | None = None,
    timeout: int | float | None = None,
    on_delta=None,
):
    """Stream one already-selected route and return the common response shape.

    返回 (response, route_name, ttft)：
    - response：SimpleNamespace(content=[text/tool_use blocks])，与非流式同构，调用方逻辑无需改
    - ttft：首个**文本** token 的耗时（秒）；纯工具调用轮没有文本则为 None
    用法：每轮主循环都用它，第一个增量是 content→本轮是最终回答(ttft 有值)；是 tool_calls→本轮调工具(ttft=None)。
    on_delta(text)：可选回调，拿到每个文本增量（供 HTTP 端流式转发，本测量场景可不传）。
    Route admission, failover and circuit accounting are owned by the wrapper.
    """
    if route.protocol == "responses":
        response, ttft = _create_responses_message_streaming(
            route=route,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools,
            model=model,
            timeout=timeout,
            on_delta=on_delta,
        )
        return response, route.name, ttft
    client = _get_openai_client(route).with_options(
        timeout=timeout if timeout is not None else _DEFAULT_LLM_TIMEOUT,
    )
    kwargs = {
        "model": model or route.model,
        "messages": _convert_messages_for_openai(system, messages),
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "extra_body": _thinking_extra_body(
            route.base_url,
            model=model or route.model,
            route_name=route.name,
        ),
    }
    openai_tools = _convert_tools_for_openai(tools)
    if openai_tools is not None:
        kwargs["tools"] = openai_tools
        kwargs["tool_choice"] = "auto"

    text_parts: list[str] = []
    tool_acc: dict[int, dict] = {}
    ttft = None
    t0 = time.time()
    for chunk in client.chat.completions.create(**kwargs):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            if ttft is None:
                ttft = time.time() - t0
            text_parts.append(text)
            if on_delta is not None:
                on_delta(text)
        for tc in (getattr(delta, "tool_calls", None) or []):
            acc = tool_acc.setdefault(tc.index, {"id": None, "name": None, "args": ""})
            if tc.id:
                acc["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if fn.name:
                    acc["name"] = fn.name
                if fn.arguments:
                    acc["args"] += fn.arguments

    blocks = []
    content = "".join(text_parts)
    if content:
        blocks.append(SimpleNamespace(type="text", text=content))
    for idx in sorted(tool_acc):
        acc = tool_acc[idx]
        try:
            args = json.loads(acc["args"] or "{}")
        except Exception:
            args = {}
        blocks.append(SimpleNamespace(
            type="tool_use",
            id=acc["id"] or f"call_{idx}",
            name=acc["name"],
            input=args,
        ))
    return SimpleNamespace(content=blocks), route.name, ttft


def create_message_streaming(
    *,
    system: str,
    messages: list,
    max_tokens: int,
    tools: list | None = None,
    model: str | None = None,
    timeout: int | float | None = None,
    on_delta=None,
    queue_timeout: int | float | None = None,
):
    """Stream through the least-loaded healthy channel.

    A second channel is attempted only when the first fails before any user
    text was emitted. Once a delta has reached the caller, restarting on a
    second model would duplicate or splice answers, so the original error is
    propagated instead.
    """
    routes = _routes_for_model(model)
    if not routes:
        raise LLMRouteBusyError("all LLM route circuits are open")
    last_exc: Exception | None = None
    acquire_timeout = (
        _DEFAULT_ROUTE_QUEUE_TIMEOUT
        if queue_timeout is None
        else max(0.0, float(queue_timeout))
    )
    for index, route in enumerate(routes):
        route_sem = _get_route_semaphore(route.name)
        if not route_sem.acquire(timeout=acquire_timeout):
            last_exc = LLMRouteBusyError(
                f"LLM route {route.name} is busy (queue timeout {acquire_timeout:.1f}s)"
            )
            continue
        begun = _begin_route(route)
        if not begun:
            route_sem.release()
            last_exc = LLMRouteBusyError(f"LLM route {route.name} circuit is unavailable")
            continue
        emitted = False
        started = time.monotonic()

        def forward_delta(text: str) -> None:
            nonlocal emitted
            if text:
                emitted = True
            if on_delta is not None:
                on_delta(text)

        try:
            response, route_name, ttft = _create_message_streaming_on_route(
                route=route,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                tools=tools,
                model=model,
                timeout=timeout,
                on_delta=forward_delta,
            )
            _route_succeeded(route, time.monotonic() - started)
            return response, route_name, ttft
        except Exception as exc:
            last_exc = exc
            _route_failed(route, exc, time.monotonic() - started)
            has_next = index < len(routes) - 1
            if not emitted and has_next and _should_fallback(exc):
                continue
            raise
        finally:
            _end_route(route)
            route_sem.release()

    if last_exc is not None:
        raise last_exc
    raise LLMRouteBusyError("no healthy LLM route is currently available")


def create_message_with_fallback(
    *,
    system: str,
    messages: list,
    max_tokens: int,
    tools: list | None = None,
    model: str | None = None,
    timeout: int | float | None = None,
    retry_attempts: int | None = None,
    queue_timeout: int | float | None = None,
):
    """Use one healthy least-loaded route and fail over once when warranted."""
    routes = _routes_for_model(model)
    if not routes:
        raise LLMRouteBusyError("all LLM route circuits are open")
    last_exc: Exception | None = None
    # With a second healthy credential available, fail over after one failed
    # call instead of spending the user's latency budget retrying the same
    # upstream.  Callers can still explicitly request same-route retries, and a
    # single-route deployment retains the configured retry policy.
    default_attempts = 1 if len(routes) > 1 else _TRANSIENT_RETRY_ATTEMPTS
    attempt_limit = max(1, int(retry_attempts if retry_attempts is not None else default_attempts))

    for idx, route in enumerate(routes):
        route_sem = _get_route_semaphore(route.name)
        acquire_timeout = _DEFAULT_ROUTE_QUEUE_TIMEOUT if queue_timeout is None else max(0.0, float(queue_timeout))
        acquired = route_sem.acquire(timeout=acquire_timeout)
        if not acquired:
            last_exc = LLMRouteBusyError(
                f"LLM route {route.name} is busy (queue timeout {acquire_timeout:.1f}s)"
            )
            if idx < len(routes) - 1:
                continue
            raise last_exc
        begun = _begin_route(route)
        if not begun:
            route_sem.release()
            last_exc = LLMRouteBusyError(f"LLM route {route.name} circuit is unavailable")
            continue
        started = time.monotonic()
        try:
            if route.protocol in {"openai", "responses"}:
                for attempt in range(attempt_limit):
                    try:
                        create_fn = _create_responses_message if route.protocol == "responses" else _create_openai_message
                        response = create_fn(
                            route=route,
                            system=system,
                            messages=messages,
                            max_tokens=max_tokens,
                            tools=tools,
                            model=model,
                            timeout=timeout,
                        )
                        _route_succeeded(route, time.monotonic() - started)
                        return response, route
                    except Exception as exc:
                        if attempt < attempt_limit - 1 and _should_retry_same_route(exc):
                            print(f"[llm] {route.name} transient failure; retry {attempt + 1}/{attempt_limit}")
                            continue
                        raise

            import anthropic

            client = anthropic.Anthropic(
                base_url=route.base_url,
                api_key=route.api_key,
            )
            kwargs = {
                "model": model or route.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
            }
            if tools is not None:
                kwargs["tools"] = tools
            kwargs["timeout"] = timeout if timeout is not None else _DEFAULT_LLM_TIMEOUT
            # 默认关 thinking：多数模型走非 reasoning 模式更快；要开 thinking 设 LLM_ENABLE_THINKING=1
            if not _env_flag("LLM_ENABLE_THINKING"):
                kwargs["extra_body"] = {"enable_thinking": False}

            for attempt in range(attempt_limit):
                try:
                    response = client.messages.create(**kwargs)
                    _route_succeeded(route, time.monotonic() - started)
                    return response, route
                except Exception as exc:
                    if attempt < attempt_limit - 1 and _should_retry_same_route(exc):
                        print(f"[llm] {route.name} transient failure; retry {attempt + 1}/{attempt_limit}")
                        continue
                    raise
        except Exception as exc:
            last_exc = exc
            _route_failed(route, exc, time.monotonic() - started)
            has_next = idx < len(routes) - 1
            if has_next and _should_fallback(exc):
                print(f"[llm] {route.name} unavailable; fail over to {routes[idx + 1].name}")
                continue
            raise
        finally:
            _end_route(route)
            route_sem.release()

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("未配置可用的 LLM 路由")

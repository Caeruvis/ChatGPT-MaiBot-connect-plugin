"""Errors deliberately exclude upstream messages, request bodies and credentials."""

import re
from typing import Any

USAGE_URL = "https://chatgpt.com/settings/usage"

HINTS = {
    "unsupported_country_region_territory": (
        "OpenAI 判定发出请求的网络来自不受支持的国家或地区。请核对运行插件的主机及其网络出口。"
        "支持范围见 https://developers.openai.com/api/docs/supported-countries"
    ),
    "subscription_sharing_user_not_eligible": "账号、工作区或策略不允许使用此方案。",
    "subscription_sharing_usage_limit_exceeded": f"方案或应用用量受限，请在 {USAGE_URL} 检查并手动恢复。",
    "subscription_sharing_usage_unavailable": "用量服务暂时不可用，凭据已保留。",
    "subscription_sharing_user_unavailable": "账号服务暂时不可用，凭据已保留。",
    "subscription_sharing_unsupported_capability": "请求包含不支持的能力，请检查 error.param。",
    "subscription_sharing_route_not_supported": "当前授权不允许此路由。",
    "subscription_sharing_invalid_user": "身份验证失败；刷新后仍失败时检查账号授权。",
    "chatpass_v2_scope_not_authorized": "此授权缺少方案调用权限。",
    "chatpass_v2_invalid_authorization_context": "授权上下文无效。",
}


class PluginError(RuntimeError):
    """A safe, user-visible message."""


def safe_field(value: Any, secrets: tuple[str, ...] = ()) -> str | None:
    if not isinstance(value, str) or any(s and s in value for s in secrets):
        return None
    return value if re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,120}", value) else None


class OpenAIError(PluginError):
    def __init__(
        self,
        status: int,
        payload: Any = None,
        request_id: str | None = None,
        *,
        secrets: tuple[str, ...] = (),
    ) -> None:
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            self.shape = "error_object"
            code = error.get("code") or error.get("type")
            param = error.get("param")
        elif isinstance(error, str):
            self.shape, code, param = "oauth_error", error, None
        else:
            self.shape = "detail" if isinstance(payload, dict) and "detail" in payload else "other"
            code, param = None, None
        self.status = status
        self.code = safe_field(code, secrets) or "unknown_error"
        self.param = safe_field(param, secrets)
        self.request_id = safe_field(request_id, secrets) or "unknown"
        hint = HINTS.get(self.code, "检查授权与请求配置；原始错误正文已隐藏。")
        super().__init__(
            f"OpenAI HTTP {status}; code={self.code}; param={self.param or '-'}; "
            f"request_id={self.request_id}; shape={self.shape}。{hint}"
        )

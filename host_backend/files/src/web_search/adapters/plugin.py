"""通过明确配置的公开插件 API 提供统一搜索与正文读取。"""

from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple
import asyncio
import json

from src.web_search.config import WebServiceConfig
from src.web_search.models import PageResult, SearchHit, SearchResult, WebSearchError

PluginCaller = Callable[[WebServiceConfig, str, Dict[str, Any]], Awaitable[Dict[str, Any]]]
_active_calls: ContextVar[Tuple[str, ...]] = ContextVar("web_plugin_calls", default=())


async def call_plugin_api(config: WebServiceConfig, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Host 内部显式路由；不允许调用未公开、禁用或名称歧义的 API。"""
    from src.plugin_runtime.integration import get_plugin_runtime_manager

    runtime = get_plugin_runtime_manager()
    matches = []
    for supervisor in runtime.supervisors:
        entry = supervisor.api_registry.get_api(config.plugin_id, name, version=config.api_version)
        if entry is not None and entry.public and entry.enabled:
            matches.append((supervisor, entry))
    if len(matches) != 1:
        raise WebSearchError("插件联网 API 不可用，请检查插件启用状态、公开 API 名称和版本")
    supervisor, entry = matches[0]
    invoke_args = dict(args)
    if entry.dynamic:
        invoke_args.update(
            __maibot_api_name__=entry.name,
            __maibot_api_full_name__=entry.full_name,
            __maibot_api_version__=entry.version,
        )
    response = await supervisor.invoke_api(
        plugin_id=entry.plugin_id,
        component_name=entry.handler_name,
        args=invoke_args,
        timeout_ms=max(1000, int(config.timeout * 1000)),
    )
    payload = response.payload
    if response.error or not isinstance(payload, dict) or payload.get("success") is not True:
        raise WebSearchError("插件联网 API 调用失败，请检查插件配置、账号权限及用量；上游正文已隐藏")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise WebSearchError("插件联网 API 未返回结构化对象")
    return result


class PluginWebAdapter:
    def __init__(self, config: WebServiceConfig, caller: Optional[PluginCaller] = None) -> None:
        self.config = config
        self.caller = caller or call_plugin_api

    async def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        key = self.config.plugin_id
        active = _active_calls.get()
        if key in active:
            raise WebSearchError("检测到插件联网递归调用")
        token = _active_calls.set((*active, key))
        try:
            data = await self.caller(self.config, name, args)
            if not isinstance(data, dict) or len(json.dumps(data, ensure_ascii=False, allow_nan=False)) > 500000:
                raise WebSearchError("插件联网返回格式无效或超过传输上限")
            return data
        except (WebSearchError, asyncio.CancelledError, TimeoutError):
            raise
        except Exception:
            raise WebSearchError("插件联网调用失败；上游响应与内部异常已隐藏") from None
        finally:
            _active_calls.reset(token)

    async def search(self, query: str, max_results: int) -> SearchResult:
        data = await self.call(self.config.search_api, {"query": query, "max_results": max_results})
        if data.get("content_kind") != "search_results" or not isinstance(data.get("results"), list):
            raise WebSearchError("插件搜索 API 返回格式错误")
        hits = []
        for row in data["results"][:max_results]:
            if not isinstance(row, dict):
                raise WebSearchError("插件搜索结果格式错误")
            hits.append(
                SearchHit(
                    title=row["title"],
                    url=row["url"],
                    snippet=row["snippet"],
                    summary=row.get("summary", ""),
                    published_at=row.get("published_at"),
                )
            )
        return SearchResult(query=query, results=hits)

    async def fetch(self, url: str) -> PageResult:
        data = await self.call(self.config.fetch_api, {"url": url})
        if data.get("content_kind") != "page_text":
            raise WebSearchError("插件读取 API 必须返回真实网页正文 page_text")
        return PageResult(url=data["url"], title=data["title"], content=data["content"])

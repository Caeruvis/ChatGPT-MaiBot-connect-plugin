"""MaiBot Runner entry point. Runtime files belong exclusively to ctx.paths.data_dir."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from maibot_sdk import API, Command, Field, LLMProvider, MaiBotPlugin, PluginConfigBase

if __package__:
    from .chatgpt_maibot import APP_NAME, CLIENT_TYPE, VERSION
    from .chatgpt_maibot.auth import OpenAIPlanAuth
    from .chatgpt_maibot.deepseek_web import DeepSeekWebServer
    from .chatgpt_maibot.errors import PluginError
    from .chatgpt_maibot.page_fetch import PageFetcher
    from .chatgpt_maibot.provider import PlanProvider
    from .chatgpt_maibot.search_service import SearchService
    from .chatgpt_maibot.store import CredentialStore
else:
    from chatgpt_maibot import APP_NAME, CLIENT_TYPE, VERSION
    from chatgpt_maibot.auth import OpenAIPlanAuth
    from chatgpt_maibot.deepseek_web import DeepSeekWebServer
    from chatgpt_maibot.errors import PluginError
    from chatgpt_maibot.page_fetch import PageFetcher
    from chatgpt_maibot.provider import PlanProvider
    from chatgpt_maibot.search_service import SearchService
    from chatgpt_maibot.store import CredentialStore


class PluginSettings(PluginConfigBase):
    config_version: str = Field(default="0.1.0", description="配置结构版本")
    enabled: bool = Field(default=True, description="启用 Responses Provider")
    oauth_timeout_seconds: int = Field(default=300, ge=60, le=900, description="本机授权等待秒数")
    request_timeout_seconds: int = Field(default=240, ge=30, le=900, description="请求总超时秒数")
    admin_stream_ids: list[str] = Field(
        default_factory=list,
        description="可管理账号的私聊 stream_id（非 QQ 号）；空列表禁用全部聊天管理命令",
    )
    enable_web_search: bool = Field(
        default=False, description="启用 web_search；须自行验证账号与模型支持"
    )
    enable_tool_controls: bool = Field(
        default=False,
        description="允许模型参数 parallel_tool_calls/tool_choice；不授予聊天管理权限",
    )


class WebSettings(PluginConfigBase):
    enabled: bool = Field(default=False, description="启用统一联网后端的搜索与正文读取 API")
    model: str = Field(default="", description="专用搜索模型，填写当前账号 models 返回的 slug")
    fetch_timeout_seconds: int = Field(default=30, ge=1, le=120, description="网页读取总超时秒数")
    deepseek_port: int = Field(
        default=0, ge=0, le=65535, description="本机 DeepSeek 兼容接口端口，0 表示关闭"
    )
    deepseek_api_key: str = Field(
        default="", repr=False, description="DeepSeek 兼容接口的本机专用密钥"
    )


class OpenAIPlanConfig(PluginConfigBase):
    plugin: PluginSettings = Field(default_factory=PluginSettings)
    web: WebSettings = Field(default_factory=WebSettings)


class OpenAIPlanPlugin(MaiBotPlugin):
    config_model = OpenAIPlanConfig

    async def on_load(self) -> None:
        settings = self.config.plugin
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30, connect=15), follow_redirects=False
        )
        self.store = CredentialStore(self.ctx.paths.data_dir)
        self.auth = OpenAIPlanAuth(self.store, self._client, settings.oauth_timeout_seconds)
        self.provider = PlanProvider(
            self.auth, self._client, timeout_seconds=settings.request_timeout_seconds
        )
        self.search_service = SearchService(self.provider)
        self.page_fetcher = PageFetcher()
        self._deepseek_server = DeepSeekWebServer(self.search, self.fetch)
        self._reports: set[asyncio.Task] = set()
        try:
            await self.auth.initialize()
            await self._configure_deepseek_web()
        except BaseException:
            await self._client.aclose()
            raise
        self.ctx.logger.info("%s %s 已加载；聊天管理默认关闭", APP_NAME, VERSION)

    async def on_unload(self) -> None:
        if hasattr(self, "_deepseek_server"):
            await self._deepseek_server.configure(0)
        if hasattr(self, "auth"):
            await self.auth.cancel_login()
        reports = list(getattr(self, "_reports", set()))
        for task in reports:
            task.cancel()
        await asyncio.gather(*reports, return_exceptions=True)
        if hasattr(self, "_client"):
            await self._client.aclose()

    async def on_config_update(
        self, scope: str, config_data: dict[str, object], version: str
    ) -> None:
        del config_data, version
        if scope == "self" and hasattr(self, "auth"):
            self.auth.timeout_seconds = self.config.plugin.oauth_timeout_seconds
            self.provider.timeout_seconds = self.config.plugin.request_timeout_seconds
            await self._configure_deepseek_web()
            # Cancel pending authorizations when the administrative access policy changes.
            await self.auth.cancel_login()

    async def _configure_deepseek_web(self) -> None:
        settings = self.config.web
        port = settings.deepseek_port if settings.enabled and self.config.plugin.enabled else 0
        await self._deepseek_server.configure(port, settings.deepseek_api_key)

    def _admin_stream(self, kwargs: dict[str, Any]) -> str:
        stream = kwargs.get("stream_id")
        if not isinstance(stream, str) or not stream:
            raise PluginError(
                "聊天管理未授权。Host 未提供有效的 stream_id；请检查命令上下文传递，"
                "或使用本机认证助手。"
            )
        if stream not in self.config.plugin.admin_stream_ids:
            self.ctx.logger.warning("聊天管理未授权；当前命令 stream_id=%r", stream)
            raise PluginError(
                f"聊天管理未授权。当前命令 stream_id={stream!r}。"
                "确认此命令来自管理员私聊后，将该 ID 加入 [plugin].admin_stream_ids 并重新加载配置。"
                "enable_tool_controls 不授予管理权限；也可使用本机认证助手。"
            )
        return stream

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        return (
            str(exc)
            if isinstance(exc, PluginError)
            else "操作失败；未输出可能包含敏感信息的内部异常。"
        )

    async def _send(self, text: str, stream: str) -> tuple[bool, str, int]:
        await self.ctx.send.text(text, stream)
        return True, "已发送结果", 2

    @Command(
        "openai_siwc_login",
        description="Continue with ChatGPT（管理员本机授权）",
        pattern=r"^/openai_siwc_login$",
    )
    async def login(self, **kwargs: Any) -> tuple[bool, str, int]:
        try:
            stream = self._admin_stream(kwargs)
            attempt = await self.auth.begin_login()
            try:
                await self.ctx.send.text(
                    "Continue with ChatGPT\n在运行 MaiBot 的同一台电脑打开：\n" + attempt.url,
                    stream,
                )
            except BaseException:
                await self.auth.cancel_login()
                raise

            async def report():
                try:
                    account = await attempt.task
                    message = (
                        "授权完成。"
                        if account["plan_enabled"]
                        else "仅完成身份登录，未获得方案调用权限。"
                    )
                    await self.ctx.send.text(message, stream)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # Never log traceback locals or upstream OAuth error messages.
                    self.ctx.logger.warning("OpenAI 授权未完成，已隐藏敏感诊断。")
                    try:
                        await self.ctx.send.text(self._safe_error(exc), stream)
                    except Exception:
                        self.ctx.logger.warning("授权结果未能发送；请通过本机助手检查状态。")

            task = asyncio.create_task(report())
            self._reports.add(task)
            task.add_done_callback(self._reports.discard)
            return True, "授权流程已启动", 2
        except Exception as exc:
            return False, self._safe_error(exc), 2

    @Command(
        "openai_siwc_models", description="列出当前账号可见模型", pattern=r"^/openai_siwc_models$"
    )
    async def models(self, **kwargs: Any) -> tuple[bool, str, int]:
        try:
            stream = self._admin_stream(kwargs)
            models = await self.provider.models()
            text = "当前账号可见模型：\n" + "\n".join(
                f"{item['display_name']}: {item['slug']}" for item in models
            )
            return await self._send(text if models else "账号没有可见模型。", stream)
        except Exception as exc:
            return False, self._safe_error(exc), 2

    @Command("openai_siwc_status", description="显示脱敏账号状态", pattern=r"^/openai_siwc_status$")
    async def status(self, **kwargs: Any) -> tuple[bool, str, int]:
        try:
            stream = self._admin_stream(kwargs)
            accounts = await self.auth.accounts()
            active = accounts["active_client_id"]
            rows = [
                f"{item['client_id']}{'（当前）' if item['client_id'] == active else ''}: "
                f"已连接={item['connected']}, 方案权限={item['plan_enabled']}"
                for item in accounts["accounts"]
            ]
            return await self._send("\n".join(rows) or "尚未注册账号。", stream)
        except Exception as exc:
            return False, self._safe_error(exc), 2

    @Command(
        "openai_siwc_logout",
        description="撤销当前会话并清除本地令牌",
        pattern=r"^/openai_siwc_logout$",
    )
    async def logout(self, **kwargs: Any) -> tuple[bool, str, int]:
        try:
            stream = self._admin_stream(kwargs)
            confirmed = await self.auth.logout()
            return await self._send(
                "已撤销会话并清除本地令牌。"
                if confirmed
                else "本地令牌已清除；远程撤销未确认，请在 ChatGPT Settings → Usage 断开应用。",
                stream,
            )
        except Exception as exc:
            return False, self._safe_error(exc), 2

    def _require_web(self) -> WebSettings:
        if not self.config.plugin.enabled or not self.config.web.enabled:
            raise PluginError("桥接插件统一联网后端未启用，请开启 [web].enabled。")
        return self.config.web

    @API(
        "search",
        description="GPT 内置联网搜索，返回带来源的统一结果",
        version="1",
        public=True,
        timeout_ms=300000,
    )
    async def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        settings = self._require_web()
        return await self.search_service.search(settings.model, query, max_results)

    @API(
        "fetch",
        description="读取公开网页真实正文，不使用模型生成全文",
        version="1",
        public=True,
        timeout_ms=130000,
    )
    async def fetch(self, url: str) -> dict[str, Any]:
        settings = self._require_web()
        return await self.page_fetcher.fetch(url, timeout_seconds=settings.fetch_timeout_seconds)

    @LLMProvider(
        CLIENT_TYPE,
        name="ChatGPT Plan",
        description="Sign in with ChatGPT Responses Provider",
        version=VERSION,
    )
    async def handle_llm_provider(self, operation: str, request: dict[str, Any]) -> dict[str, Any]:
        if not self.config.plugin.enabled:
            raise PluginError("ChatGPT Plan Provider 已禁用。")
        if operation != "response":
            raise PluginError(
                "此授权仅支持 response；embedding、audio_transcription 等任务需使用其他 Provider。"
            )
        return await self.provider.response(
            request,
            enable_web_search=self.config.plugin.enable_web_search,
            enable_tool_controls=self.config.plugin.enable_tool_controls,
        )


def create_plugin() -> OpenAIPlanPlugin:
    return OpenAIPlanPlugin()

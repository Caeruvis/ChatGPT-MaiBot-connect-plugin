"""Local authorization helper; intentionally performs no inference or tool execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import webbrowser
from pathlib import Path

import httpx

from . import VERSION
from .auth import OpenAIPlanAuth
from .errors import PluginError
from .page_fetch import PageFetcher
from .provider import PlanProvider
from .search_service import SearchService
from .store import CredentialStore


def parser():
    root = argparse.ArgumentParser(description="ChatGPT-MaiBot 本机认证助手（无需 API Key）")
    root.add_argument("--data-dir", type=Path, required=True, help="MaiBot 为此插件分配的数据目录")
    sub = root.add_subparsers(dest="command", required=True)
    login = sub.add_parser("login", help="Continue with ChatGPT")
    login.add_argument("--new", action="store_true", help="增加其他账号或工作区")
    login.add_argument("--client-id", help="重新授权指定的已注册账号")
    login.add_argument("--consent", action="store_true", help="明确重新请求方案调用权限")
    login.add_argument("--no-browser", action="store_true", help="仅在本机终端显示授权链接")
    login.add_argument(
        "--timeout", type=int, default=300, choices=range(60, 901), metavar="60..900"
    )
    sub.add_parser("init", help="初始化本机 host ID")
    sub.add_parser("accounts", help="显示不含令牌或邮箱的账号列表")
    sub.add_parser("status", help="显示不含令牌或邮箱的账号列表")
    sub.add_parser("models", help="实时读取当前账号模型目录")
    test = sub.add_parser("test", help="发送一次固定的最小测试请求，不读取聊天历史")
    test.add_argument("--model", required=True, help="models 命令返回的账号可见 slug")
    test.add_argument(
        "--web-search", action="store_true", help="测试允许联网搜索；不强制模型调用搜索"
    )
    search = sub.add_parser("search", help="测试统一联网接口，执行一次真实搜索")
    search.add_argument("--model", required=True, help="账号可见的模型 slug")
    search.add_argument("--query", required=True, help="仅发送这条查询，不读取聊天历史")
    search.add_argument("--max-results", type=int, default=5, choices=range(1, 21))
    fetch = sub.add_parser("fetch", help="提取公开网页正文，不调用模型")
    fetch.add_argument("--url", required=True)
    switch = sub.add_parser("switch", help="选择已有账号")
    switch.add_argument("client_id")
    logout = sub.add_parser("logout", help="撤销会话并清除本地令牌，保留注册信息")
    logout.add_argument("--client-id")
    sub.add_parser("resume", help="检查 ChatGPT Usage 后手动解除用量暂停")
    return root


async def run(args) -> int:
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30, connect=15), follow_redirects=False
    ) as client:
        auth = OpenAIPlanAuth(CredentialStore(args.data_dir), client, getattr(args, "timeout", 300))
        provider = PlanProvider(auth, client)
        try:
            await auth.initialize()
            if args.command == "init":
                print("本机 host ID 已初始化；请勿从其他主机覆盖 host_id.txt。")
            elif args.command == "login":
                if args.new and args.client_id:
                    raise PluginError("--new 与 --client-id 不能同时使用。")
                attempt = await auth.begin_login(
                    new=args.new, client_id=args.client_id, consent=args.consent
                )
                print(
                    "Continue with ChatGPT：仅在运行此助手的同一台电脑浏览器完成授权。", flush=True
                )
                opened = False
                if not args.no_browser:
                    opened = await asyncio.to_thread(webbrowser.open, attempt.url)
                if not opened:
                    print(
                        "请在本机浏览器打开以下链接；不要转发到群聊或日志：\n" + attempt.url,
                        flush=True,
                    )
                result = await attempt.task
                print(
                    "授权完成。"
                    if result["plan_enabled"]
                    else "已登录，但未获方案调用权限；推理保持禁用。"
                )
            elif args.command in {"accounts", "status"}:
                print(json.dumps(await auth.accounts(), ensure_ascii=False, indent=2))
            elif args.command == "models":
                print(json.dumps(await provider.models(), ensure_ascii=False, indent=2))
            elif args.command == "test":
                prompt = (
                    "请使用联网搜索查阅 Python 官方网站，给出当前最新稳定版的版本号和官方链接。"
                    if args.web_search
                    else "请只回复 CONNECT_OK。"
                )
                print(
                    f"插件 {VERSION}；正在发送一条固定测试请求；联网搜索={args.web_search}。",
                    flush=True,
                )
                result = await provider.response(
                    {
                        "model_info": {"model_identifier": args.model},
                        "message_list": [{"role": "user", "content": prompt}],
                    },
                    enable_web_search=args.web_search,
                )
                print(
                    json.dumps(
                        {
                            "plugin_version": VERSION,
                            "model": args.model,
                            "diagnostics": result["raw_data"],
                            "content": result["content"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                if args.web_search and not result["raw_data"]["web_search_completed"]:
                    print("本次未观察到完成的 web_search_call，尚不能确认联网搜索成功。")
            elif args.command == "search":
                result = await SearchService(provider).search(
                    args.model, args.query, args.max_results
                )
                print(json.dumps(result, ensure_ascii=False, indent=2))
            elif args.command == "fetch":
                result = await PageFetcher().fetch(args.url)
                print(json.dumps(result, ensure_ascii=False, indent=2))
            elif args.command == "switch":
                await auth.switch(args.client_id)
                print("已切换账号；下一次请求使用该账号的模型目录。")
            elif args.command == "logout":
                confirmed = await auth.logout(args.client_id)
                print(
                    "远程撤销已确认；本地令牌已清除。"
                    if confirmed
                    else "本地令牌已清除；远程撤销未确认，请在 https://chatgpt.com/settings/usage 断开应用。"
                )
            elif args.command == "resume":
                await provider.resume()
                print("已解除用量暂停；如果额度仍受限，下次请求会再次暂停。")
            return 0
        finally:
            await auth.cancel_login()


def main() -> int:
    args = parser().parse_args()
    try:
        return asyncio.run(run(args))
    except PluginError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130
    except Exception:
        print("操作失败，内部异常已隐藏以保护凭据；请检查目录权限与运行环境。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

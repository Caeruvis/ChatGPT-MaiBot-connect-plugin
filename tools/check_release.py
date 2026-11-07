"""Local consistency checks. This is not MaiBot's Host ManifestValidator."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def check(*, release: bool = False) -> list[str]:
    errors = []
    manifest = json.loads((ROOT / "_manifest.json").read_text())
    if manifest.get("manifest_version") != 2:
        errors.append("manifest_version 必须为 2")
    for source in [ROOT / "plugin.py", *(ROOT / "chatgpt_maibot").glob("*.py")]:
        text = source.read_text()
        tree = ast.parse(text, feature_version=(3, 12))
        compile(tree, str(source), "exec")
        for node in ast.walk(tree):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            if any(name == "src" or name.startswith("src.") for name in modules):
                errors.append(f"禁止导入 Host 内部模块：{source.name}")
        if "backend-api" in text or ".codex/auth.json" in text:
            errors.append(f"禁止的凭据或私有路由：{source.name}")
    sys.path.insert(0, str(ROOT))
    from plugin import create_plugin

    instance = create_plugin()
    declared = [
        {k: p.get(k) for k in ("client_type", "name", "description", "version")}
        for p in instance.get_llm_providers()
    ]
    if declared != manifest.get("llm_providers"):
        errors.append("Manifest 与 SDK Provider 声明不一致")
    instance.normalize_plugin_config({})
    instance.get_webui_config_schema()
    if manifest.get("capabilities") != ["send.text"]:
        errors.append("能力声明应恰好为 send.text")
    requirements = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            requirement = Requirement(line)
            requirements[canonicalize_name(requirement.name)] = str(requirement.specifier)
    dependencies = {
        canonicalize_name(item["name"]): str(
            Requirement(item["name"] + item["version_spec"]).specifier
        )
        for item in manifest["dependencies"]
        if item["type"] == "python_package"
    }
    if requirements != dependencies:
        errors.append("Manifest 与 requirements 依赖不一致")
    for path in (ROOT / "examples").glob("*.toml"):
        tomllib.loads(path.read_text())
    for path in ROOT.rglob("*"):
        if path.is_file() and path.name in {"accounts.json", "host_id.txt", "config.toml"}:
            errors.append(f"源码目录存在运行时凭据或配置：{path.relative_to(ROOT)}")
    if release:
        if manifest["author"]["name"] == "待填写":
            errors.append("发布前必须填写作者")
        if "example.invalid" in json.dumps(manifest):
            errors.append("发布前必须替换 example.invalid 元信息")
        # Keep the existing ID so upgrades can find the authorized account data.
        if manifest["id"] != "local.chatgpt-maibot-connect":
            errors.append("发布时必须保留现有插件 ID，以便已有实例继续读取授权数据")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    errors = check(release=args.release)
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print("本地一致性检查通过。" + ("" if args.release else "发布元信息仍需人工填写。"))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

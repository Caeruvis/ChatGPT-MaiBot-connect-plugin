"""Bundle the reviewed MaiBot source changes against an immutable upstream commit."""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
from pathlib import Path

from install_host_backend import digest

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "c6429547af711555e3d4d1b8d04b413572cb616c"
INSTALL_FILES = [
    "src/config/config.py",
    "src/web_search/config.py",
    "src/web_search/service.py",
    "src/web_search/adapters/plugin.py",
    "src/web_search/README.md",
    "dashboard/src/routes/config/webSearch/types.ts",
    "dashboard/src/routes/config/webSearch/WebSearchDialog.tsx",
]
REVIEW_FILES = [
    *INSTALL_FILES,
    "pytests/web_search/test_plugin_backend.py",
    "changelogs/changelog.md",
]


def build(source: Path) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if head != BASE_COMMIT:
        raise ValueError("主程序工作副本的基线提交不匹配。")
    bundle = ROOT / "host_backend"
    bundle.mkdir(exist_ok=True)
    records, patch = [], []
    for name in REVIEW_FILES:
        result = subprocess.run(
            ["git", "show", f"{BASE_COMMIT}:{name}"], cwd=source, capture_output=True, check=False
        )
        if result.returncode and name not in {
            "src/web_search/adapters/plugin.py",
            "pytests/web_search/test_plugin_backend.py",
        }:
            raise ValueError(f"无法读取基线文件：{name}")
        previous = result.stdout if result.returncode == 0 else None
        current = (source / name).read_bytes()
        patch.extend(
            difflib.unified_diff(
                previous.decode().splitlines(keepends=True) if previous is not None else [],
                current.decode().splitlines(keepends=True),
                fromfile=f"a/{name}" if previous is not None else "/dev/null",
                tofile=f"b/{name}",
            )
        )
        if name in INSTALL_FILES:
            target = bundle / "files" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(current)
            records.append(
                {"path": name, "base_sha256": digest(previous), "sha256": digest(current)}
            )
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "base_commit": BASE_COMMIT,
                "base_version": "1.3.19",
                "files": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (bundle / "unified-web-backend.patch").write_text("".join(patch))
    (bundle / "LICENSE").write_bytes((source / "LICENSE").read_bytes())
    (bundle / "README.md").write_text(
        "# MaiBot 统一联网后端配套补丁\n\n"
        f"来源：Mai-with-u/MaiBot，基线提交 `{BASE_COMMIT}`（1.3.19）。\n\n"
        "主程序补丁沿用上游 LICENSE，完整许可见本目录 LICENSE；桥接插件独立沿用其根目录 MIT 许可。\n\n"
        "files/ 为安装器使用的源码；unified-web-backend.patch 包含完整可审查改动及回归测试。\n"
        "安装与回滚见插件根目录 UNIFIED_WEB_BACKEND.md。\n"
    )
    print(f"已生成 {len(records)} 个安装文件及完整源码补丁。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    build(parser.parse_args().source.resolve())

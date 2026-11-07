"""Run actual MaiBot release Manifest/context contracts without booting the Host.

Only Host's logging module is replaced to avoid creating a runtime instance.
This does not test RPC, browser authorization, chat adapters or real inference.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import sys
import tempfile
import tomllib
import types
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host-source",
        type=Path,
        required=True,
        help="MaiBot source or unpacked wheel root",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.host_source.resolve()))
    with tempfile.TemporaryDirectory(prefix="maibot-contract-") as runtime:
        os.environ["MAIBOT_INSTANCE_DIR"] = runtime
        logger_stub = types.ModuleType("src.common.logger")
        logger_stub.get_logger = logging.getLogger
        sys.modules["src.common.logger"] = logger_stub

        from src.llm_models.payload_content.context_item import (
            ContextImagePart,
            ContextItemMeta,
            ContextTextPart,
            UserMessageItem,
        )
        from src.llm_models.payload_content.context_protocol import (
            ContextProtocolMode,
            validate_context_items,
        )
        from src.llm_models.payload_content.context_snapshot import (
            deserialize_context_item_snapshot,
            serialize_context_items_snapshot,
        )
        from src.llm_models.payload_content.resp_format import RespFormat, RespFormatType
        from src.plugin_runtime.runner.manifest_validator import ManifestValidator

        project = tomllib.loads((args.host_source / "pyproject.toml").read_text())["project"]
        host_version, sdk_version = project["version"], version("maibot-plugin-sdk")
        validator = ManifestValidator(
            host_version=host_version, sdk_version=sdk_version, log_errors=False
        )
        manifest = validator.load_from_plugin_path(ROOT)
        assert manifest is not None, validator.errors
        # Same package import strategy as Host PluginLoader, with no plugin-root sys.path entry.
        name = "verified_chatgpt_plugin"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "plugin.py", submodule_search_locations=[str(ROOT)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        instance = module.create_plugin()
        assert (
            instance.get_llm_providers()[0]["client_type"] == manifest.llm_providers[0].client_type
        )
        assert instance.normalize_plugin_config({})[0]["plugin"]["config_version"] == "0.1.0"

        mapping = importlib.import_module(name + ".chatgpt_maibot.mapping")
        source = UserMessageItem(
            meta=ContextItemMeta.create(),
            parts=(ContextTextPart("Describe"), ContextImagePart("png", "ZmFrZQ==")),
        )
        request = {
            "item_schema_version": 1,
            "context_items": serialize_context_items_snapshot([source]),
            "model_info": {"model_identifier": "test-model"},
            "response_format": RespFormat(RespFormatType.JSON_OBJ).to_dict(),
            "tool_options": [{"type": "function", "function": {"name": "weather.query"}}],
        }
        body, reverse = mapping.build_responses_body(request)
        assert body["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
        assert body["text"]["format"]["type"] == "json_object"
        wire = next(iter(reverse))
        parsed = mapping.parse_completed_response(
            {
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "Done"}]},
                    {
                        "type": "function_call",
                        "name": wire,
                        "namespace": "maibot",
                        "call_id": "test_call",
                        "arguments": '{"city":"Nanjing"}',
                    },
                ],
                "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            },
            reverse,
        )
        meta = {
            "item_id": "",
            "logical_turn_id": "contract-turn",
            "timestamp": datetime.now().isoformat(),
        }
        output = [
            deserialize_context_item_snapshot({**item, "meta": {**meta, "item_id": f"output-{i}"}})
            for i, item in enumerate(parsed["output_items"])
        ]
        validate_context_items(output, ContextProtocolMode.MODEL_OUTPUT)
        assert output[1].tool_call.materialize_args() == {"city": "Nanjing"}
        assert parsed["usage"]["prompt_tokens"] == 5
        print(
            json.dumps(
                {
                    "host": host_version,
                    "sdk": sdk_version,
                    "manifest": "passed",
                    "package_import": "passed",
                    "native_context_roundtrip": "passed",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

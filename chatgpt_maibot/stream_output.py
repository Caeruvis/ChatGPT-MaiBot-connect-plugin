"""Collect finished output items while keeping response.completed as the success gate."""

from __future__ import annotations

from typing import Any

from .errors import PluginError


class CompletedOutput:
    def __init__(self) -> None:
        self._response_id: str | None = None
        self._items: dict[int, dict[str, Any]] = {}
        self._unindexed: list[dict[str, Any]] = []
        self._announced: set[int] = set()
        self._item_indices: dict[str, int | None] = {}

    @staticmethod
    def _index(value: Any) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 4096:
            raise PluginError("流式输出的 output_index 无效。")
        return value

    def observe(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind in {
            "response.created",
            "response.in_progress",
            "response.completed",
            "response.failed",
            "response.incomplete",
        }:
            response = event.get("response")
            response_id = response.get("id") if isinstance(response, dict) else None
        else:
            response_id = event.get("response_id")
        if response_id is not None:
            if not isinstance(response_id, str) or not response_id:
                raise PluginError("流式输出的 response ID 无效。")
            if self._response_id is not None and response_id != self._response_id:
                raise PluginError("流式事件的 response ID 不一致，未返回任何输出。")
            self._response_id = response_id

        if kind == "response.output_item.added":
            if "output_index" in event:
                self._announced.add(self._index(event["output_index"]))
            return
        if kind != "response.output_item.done":
            return
        item = event.get("item")
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            raise PluginError("output_item.done 缺少有效输出项。")
        status = item.get("status")
        if status is not None and status != "completed":
            # A failed hosted search may be followed by a valid explanatory assistant message.
            if not (item["type"] == "web_search_call" and status == "failed"):
                raise PluginError("output_item.done 中的输出项尚未完成。")
        index = self._index(event["output_index"]) if "output_index" in event else None
        item_id = item.get("id")
        if item_id is not None:
            if not isinstance(item_id, str) or not item_id:
                raise PluginError("流式输出项 ID 无效。")
            if item_id in self._item_indices and self._item_indices[item_id] != index:
                raise PluginError("同一流式输出项出现了不同索引。")
            self._item_indices[item_id] = index
        if index is not None:
            previous = self._items.get(index)
            if previous is not None and previous != item:
                raise PluginError("同一输出索引出现了冲突的完成项。")
            self._items[index] = item
        elif item_id is not None and any(old.get("id") == item_id for old in self._unindexed):
            if not any(old == item for old in self._unindexed):
                raise PluginError("同一无索引输出项的内容发生冲突。")
        else:
            self._unindexed.append(item)
        if len(self._items) + len(self._unindexed) > 4096:
            raise PluginError("流式输出项数量超过本地上限。")

    def finish(self, response: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Called only for response.completed; never build an output item from text deltas."""
        if response.get("status") != "completed" or not isinstance(response.get("output"), list):
            raise PluginError("completed 事件中的响应快照无效。")
        if response.get("error") is not None or response.get("incomplete_details") is not None:
            raise PluginError("完成事件仍携带错误或未完成信息，未返回输出。")
        if response["output"]:
            return response, "completed_snapshot"
        if self._unindexed:
            if self._items or len(self._unindexed) != 1 or self._announced - {0}:
                raise PluginError("完成项缺少可确定的输出顺序。")
            output = list(self._unindexed)
        elif self._items:
            indices = sorted(self._items)
            if indices != list(range(len(indices))) or self._announced - self._items.keys():
                raise PluginError("流式输出项不完整，未返回部分输出。")
            output = [self._items[index] for index in indices]
        else:
            return response, "empty_completed_snapshot"
        return {**response, "output": output}, "output_item_done"

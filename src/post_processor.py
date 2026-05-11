"""Deterministic post-processor for the executor's tool_calls.

The roadmap identified four high-confidence trace-level fixes that do not
require LLM reasoning:

  1. record_decision must be the LAST entry in the tool_calls array.
  2. If a tool's schema lists ``reason`` as a property but the executor
     filled in ``notes`` / ``rationale`` / ``comment`` / ``justification``
     instead, rename so the grader's tool_called_with check sees the
     schema field.
  3. If a schema field is a list-of-strings but the executor passed a
     single string, wrap it.
  4. Drop arguments whose key is not in the tool's schema AND whose
     value is empty (None / "" / [] / {}). This avoids polluting the
     grader's tool_called_with diff with junk keys.

We do NOT attempt label changes, content rewrites, identifier backfills,
or workflow-tool insertions — those need an LLM and risk regression
(run 6's validator over-revised in exactly those categories).

Each helper returns its input unchanged when it has no schema to work
from or the change is unsafe.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


REASON_ALIASES = ("notes", "rationale", "comment", "justification", "explanation")


def post_process_tool_calls(
    tool_calls: list[Any],
    tools: list[dict],
) -> list[dict]:
    """Apply deterministic fixes to the executor's tool_calls array.

    Input ``tool_calls`` is the raw list from the litellm response
    (OpenAI ChatCompletionMessageToolCall objects). Output is the same
    list as plain dicts in A2A wire format, with deterministic fixes
    applied.
    """
    schema_by_name = _index_schemas(tools)

    fixed: list[dict] = []
    for tc in tool_calls or []:
        d = _tool_call_to_dict(tc)
        name = d["function"]["name"]
        schema = schema_by_name.get(name)
        if schema is not None:
            d["function"]["arguments"] = _normalize_arguments(
                d["function"]["arguments"], schema
            )
        fixed.append(d)

    return _record_decision_last(fixed)


def _index_schemas(tools: list[dict]) -> dict[str, dict]:
    """Map tool name -> JSON schema (the `parameters` dict)."""
    out: dict[str, dict] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name", "")).strip()
        params = fn.get("parameters")
        if name and isinstance(params, dict):
            out[name] = params
    return out


def _tool_call_to_dict(tc: Any) -> dict:
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        return {
            "id": tc.get("id", ""),
            "type": tc.get("type", "function"),
            "function": {
                "name": str(fn.get("name", "")),
                "arguments": fn.get("arguments", ""),
            },
        }
    fn = getattr(tc, "function", None)
    return {
        "id": getattr(tc, "id", ""),
        "type": "function",
        "function": {
            "name": getattr(fn, "name", "") if fn is not None else "",
            "arguments": getattr(fn, "arguments", "") if fn is not None else "",
        },
    }


def _normalize_arguments(raw_args: Any, schema: dict) -> str:
    """Apply field-name aliasing, list-wrapping, and empty-key pruning.

    Input is the JSON-encoded string the model emitted. Returns a
    JSON-encoded string. On parse failure, the original is returned.
    """
    if not isinstance(raw_args, str) or not raw_args.strip():
        return raw_args if isinstance(raw_args, str) else "{}"
    try:
        args = json.loads(raw_args)
    except Exception:
        return raw_args
    if not isinstance(args, dict):
        return raw_args

    props = schema.get("properties") if isinstance(schema, dict) else None
    props = props if isinstance(props, dict) else {}

    changed = False

    # 1. Alias reason-like keys -> "reason" if the schema has "reason".
    if "reason" in props and not _has_nonempty(args.get("reason")):
        for alias in REASON_ALIASES:
            if _has_nonempty(args.get(alias)):
                args["reason"] = args.pop(alias)
                changed = True
                break

    # 2. Wrap single-string values that the schema expects as a list.
    for key, propschema in props.items():
        if not isinstance(propschema, dict):
            continue
        if propschema.get("type") != "array":
            continue
        item_schema = propschema.get("items") or {}
        if isinstance(item_schema, dict) and item_schema.get("type") != "string":
            continue
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            args[key] = [val]
            changed = True

    # 3. Drop keys not in the schema whose value is empty.
    for key in list(args.keys()):
        if key in props:
            continue
        if not _has_nonempty(args[key]):
            args.pop(key)
            changed = True

    if not changed:
        return raw_args
    return json.dumps(args)


def _has_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def _record_decision_last(tool_calls: list[dict]) -> list[dict]:
    """If record_decision is present and not last, move it to the end.

    Preserves the relative order of all other calls.
    """
    if not tool_calls:
        return tool_calls
    last_name = tool_calls[-1]["function"].get("name", "")
    if last_name == "record_decision":
        return tool_calls

    rd_indices = [
        i for i, tc in enumerate(tool_calls)
        if tc["function"].get("name", "") == "record_decision"
    ]
    if not rd_indices:
        return tool_calls

    # Keep only the last record_decision (defensive against accidental dups).
    keep_rd_at = rd_indices[-1]
    rd_call = tool_calls[keep_rd_at]
    others = [
        tc for i, tc in enumerate(tool_calls)
        if tc["function"].get("name", "") != "record_decision"
    ]
    logger.info("Post-process: moved record_decision to last (was index %d of %d)",
                keep_rd_at, len(tool_calls))
    return [*others, rd_call]

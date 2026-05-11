"""FastAPI A2A purple-agent server for pi-bench.

Exposes the pi-bench bootstrap extension so policy/task context and tool
schemas are sent once per scenario and cached against a context_id. Each
turn calls OpenAI gpt-5-mini (or any model the operator overrides via the
`--model` flag / OPENAI_MODEL env var) with the full benchmark system
prompt assembled in `system_prompt.build_system_prompt`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from typing import Any

import litellm
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .post_processor import post_process_tool_calls
from .session_state import assess as assess_session, build_addendum
from .system_prompt import build_system_prompt

logger = logging.getLogger(__name__)

POLICY_BOOTSTRAP_EXTENSION = "urn:pi-bench:policy-bootstrap:v1"

app = FastAPI(title="pi-bench-purple-policygrounder")

_model: str = "gpt-5-mini"
_reasoning_effort: str | None = "high"
_seed: int | None = None
_card_url: str = ""
_host: str = "0.0.0.0"
_port: int = 8080
_sessions: dict[str, dict] = {}


def _build_agent_card() -> dict[str, Any]:
    """Return an A2A-spec-compliant agent card.

    Built via the official `a2a-sdk` pydantic models so the output matches
    exactly what amber/gateway parsers expect (correct camelCase aliases,
    required fields like `preferredTransport`, current `protocolVersion`).
    The Pi-Bench `urn:pi-bench:policy-bootstrap:v1` extension is declared
    in `capabilities.extensions` so the green grader knows to send the
    bootstrap payload at session start.
    """
    from a2a.types import (
        AgentCapabilities,
        AgentCard,
        AgentExtension,
        AgentSkill,
    )

    url = _card_url or f"http://{_host}:{_port}/"
    skill = AgentSkill(
        id="pibench_policy_compliance",
        name="Pi-Bench policy compliance",
        description=(
            "Reads the scenario policy and external tool inventory, "
            "gathers required state via tool calls, executes the "
            "policy-prescribed workflow, and records a canonical "
            "decision in {ALLOW, ALLOW-CONDITIONAL, DENY, ESCALATE}."
        ),
        tags=["pi-bench", "policy-compliance", "structured-execution"],
        examples=[
            "Process a refund request within the return window.",
            "Decide whether a wire transfer requires compliance escalation.",
            "Approve or deny an access-grant request against a policy doc.",
        ],
    )
    capabilities = AgentCapabilities(
        streaming=False,
        push_notifications=False,
        state_transition_history=False,
        extensions=[
            AgentExtension(
                uri=POLICY_BOOTSTRAP_EXTENSION,
                description="Pi-Bench one-shot policy + tool-inventory bootstrap",
                required=False,
            )
        ],
    )
    card = AgentCard(
        name="pi-bench-purple-policygrounder",
        description=(
            "Policy-grounded purple agent for the AgentBeats Pi-Bench "
            "benchmark. Reads policy + state through tools, executes the "
            "full prescribed workflow, and records exactly one canonical "
            "decision per scenario."
        ),
        url=url,
        version="0.1.0",
        skills=[skill],
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
    )
    return card.model_dump(mode="json", by_alias=True, exclude_none=True)


@app.get("/.well-known/agent.json")
async def agent_card() -> JSONResponse:
    return JSONResponse(_build_agent_card())


@app.get("/.well-known/agent-card.json")
async def agent_card_alias() -> JSONResponse:
    return JSONResponse(_build_agent_card())


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "model": _model})


@app.post("/")
async def message_send(request: Request) -> JSONResponse:
    body = await request.json()
    if body.get("method") != "message/send":
        return _jsonrpc_error(body.get("id"), -32601, f"Unknown method: {body.get('method')}")

    parts = (body.get("params") or {}).get("message", {}).get("parts") or []
    if not parts:
        return _jsonrpc_error(body.get("id"), -32602, "No message parts")
    data = parts[0].get("data", {}) or {}

    if data.get("bootstrap"):
        return _handle_bootstrap(body.get("id"), data)
    return await _handle_turn(body.get("id"), data)


def _handle_bootstrap(request_id: str | None, data: dict) -> JSONResponse:
    context_id = str(uuid.uuid4())
    benchmark_context = _as_list(data.get("benchmark_context"))
    tools = _as_list(data.get("tools"))
    _sessions[context_id] = {
        "tools": tools,
        "system_prompt": build_system_prompt(benchmark_context, tools),
        "run_id": data.get("run_id"),
        "domain": data.get("domain", ""),
    }
    logger.info(
        "Bootstrap: context_id=%s ctx_nodes=%d tools=%d",
        context_id, len(benchmark_context), len(tools),
    )
    return _jsonrpc_success(request_id, {
        "kind": "data",
        "data": {"bootstrapped": True, "context_id": context_id},
    })


async def _handle_turn(request_id: str | None, data: dict) -> JSONResponse:
    context_id = data.get("context_id")
    messages = _as_list(data.get("messages"))

    session = None
    if context_id:
        session = _sessions.get(str(context_id))
        if session is None:
            return _jsonrpc_error(request_id, -32004, f"Unknown context_id: {context_id}")
        tools = session["tools"]
        system_prompt = session["system_prompt"]
    else:
        benchmark_context = _as_list(data.get("benchmark_context"))
        tools = _as_list(data.get("tools"))
        system_prompt = build_system_prompt(benchmark_context, tools)

    visible = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
    session_assessment = assess_session(visible)
    addendum = build_addendum(session_assessment)
    if addendum:
        model_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": addendum},
            *visible,
        ]
    else:
        model_messages = [{"role": "system", "content": system_prompt}, *visible]

    kwargs: dict[str, Any] = {
        "model": _model,
        "messages": model_messages,
        "drop_params": True,
        "num_retries": 3,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if _seed is not None:
        kwargs["seed"] = _seed
    if _reasoning_effort:
        kwargs["reasoning_effort"] = _reasoning_effort

    try:
        response = await asyncio.to_thread(litellm.completion, **kwargs)
    except Exception as exc:
        logger.exception("litellm.completion failed")
        return _jsonrpc_error(request_id, -32000, str(exc))

    choice_message = response.choices[0].message
    return _jsonrpc_success(request_id, _format_response(choice_message, tools))


def _format_response(choice_message: Any, tools: list[dict]) -> dict:
    tool_calls_raw = getattr(choice_message, "tool_calls", None)
    content = getattr(choice_message, "content", None)
    if tool_calls_raw:
        tc_list = post_process_tool_calls(list(tool_calls_raw), tools)
        data: dict[str, Any] = {"tool_calls": tc_list}
        if content:
            data["content"] = content
        return {"kind": "data", "data": data}
    if content:
        return {"kind": "data", "data": {"content": content}}
    return {"kind": "data", "data": {"content": "###STOP###"}}


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _jsonrpc_success(request_id: str | None, part: dict) -> JSONResponse:
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": request_id or str(uuid.uuid4()),
        "result": {
            "status": {
                "message": {
                    "role": "agent",
                    "parts": [part],
                },
            },
        },
    })


def _jsonrpc_error(request_id: str | None, code: int, message: str) -> JSONResponse:
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": request_id or str(uuid.uuid4()),
        "error": {"code": code, "message": message},
    })


def main() -> None:
    global _model, _reasoning_effort, _seed, _card_url, _host, _port

    parser = argparse.ArgumentParser(description="pi-bench purple agent (policy-grounder)")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument("--reasoning-effort", default=os.environ.get("REASONING_EFFORT", "high"),
                        choices=["minimal", "low", "medium", "high", ""])
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--card-url", default=os.environ.get("CARD_URL", ""))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    _model = args.model
    _reasoning_effort = args.reasoning_effort or None
    _seed = args.seed
    _card_url = args.card_url
    _host = args.host
    _port = args.port

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info("Starting purple agent: model=%s effort=%s host=%s port=%d",
                _model, _reasoning_effort, args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

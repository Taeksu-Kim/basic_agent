"""Offline conformance check for LLM client adapters (no model, no network).

Each provider's real SDK path runs (any-llm -> openai / anthropic SDK -> HTTP), but
the HTTP layer is an ``httpx.MockTransport`` that records the request and answers
with a scripted provider-shaped response. So this checks what *would be sent* and how
responses/errors are mapped -- not model quality.

    python -m agent.examples.llm_adapter_check

Checks per provider: schema delivery (native vs prompt), extra body fields, token
usage, context-overflow mapping, and an end-to-end ReAct + plan_execute run.

Mock response/error shapes are modelled on each provider's documented format; the
vLLM overflow message is the one the real server returned on 2026-11-01.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from agent.core.contract import envelope
from agent.core.hitl import arun_graph
from agent.core.llm import AnyLLMClient, ContextOverflowError
from agent.core.tools import ToolRegistry
from agent.patterns import registry as patterns

OVERFLOW = {
    "openai_compatible": (400, {"error": {"message": "This model's maximum context length is 8192 tokens. "
                                          "However, you requested 1024 output tokens and your prompt contains "
                                          "at least 7169 input tokens", "type": "BadRequestError", "code": 400}}),
    "openai": (400, {"error": {"message": "This model's maximum context length is 128000 tokens.",
                               "type": "invalid_request_error", "code": "context_length_exceeded"}}),
    "anthropic": (400, {"type": "error", "error": {"type": "invalid_request_error",
                                                   "message": "prompt is too long: 210000 tokens > 200000 maximum"}}),
}


class Backend:
    """Scripted provider: replies with queued model outputs, records request bodies."""

    def __init__(self, provider: str, outputs: list[str], overflow: bool = False) -> None:
        self.provider, self.outputs, self.overflow = provider, list(outputs), overflow
        self.bodies: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.bodies.append(body)
        if self.overflow:
            status, err = OVERFLOW[self.provider]
            return httpx.Response(status, json=err)
        text = self.outputs.pop(0) if self.outputs else '{"action": "finish", "final": "(script exhausted)"}'
        if self.provider == "anthropic":
            return httpx.Response(200, json={
                "id": "msg", "type": "message", "role": "assistant", "model": "m",
                "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 3}})
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}})


def make_client(provider: str, backend: Backend, **kw: Any) -> AnyLLMClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(backend.handler))
    if provider == "openai_compatible":
        return AnyLLMClient("openai_compatible", "local", api_base="http://vllm.invalid/v1",
                            api_key="none", client_kwargs={"http_client": http}, **kw)
    return AnyLLMClient(provider, "m", api_key="sk-fake", client_kwargs={"http_client": http}, **kw)


SCHEMA = {"anyOf": [
    {"type": "object", "properties": {"action": {"type": "string", "const": "tool"}, "tool": {"type": "string"},
                                      "args": {"type": "object"}}, "required": ["action", "tool", "args"]},
    {"type": "object", "properties": {"action": {"type": "string", "const": "finish"},
                                      "final": {"type": "string"}}, "required": ["action", "final"]}]}

rows: list[tuple[str, str, bool, str]] = []


def check(provider: str, name: str, ok: bool, detail: str) -> None:
    rows.append((provider, name, ok, detail))


def run_provider(provider: str) -> None:
    # 1) schema delivery
    b = Backend(provider, ['{"action": "finish", "final": "x"}'])
    c = make_client(provider, b, extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                    if provider == "openai_compatible" else None)
    c.complete("SYS", "USER", schema=SCHEMA)
    body = b.bodies[0]
    sent = json.dumps(body, ensure_ascii=False)
    if c.structured == "native":
        rf = body.get("response_format") or {}
        kept = "anyOf" in json.dumps(rf) and '"args": {"type": "object"}' in json.dumps(rf)
        check(provider, "schema (native)", rf.get("type") == "json_schema" and kept,
              f"response_format json_schema, anyOf·free-form args kept={kept}, strict={rf.get('json_schema', {}).get('strict')}")
    else:
        system_text = body.get("system") if provider == "anthropic" else json.dumps(body.get("messages"))
        ok = ("output_config" not in body and "response_format" not in body
              and "anyOf" in json.dumps(system_text, ensure_ascii=False))
        check(provider, "schema (prompt)", ok, "no native format; schema in system prompt" if ok else sent[:120])
    # 2) extra body fields
    if provider == "openai_compatible":
        check(provider, "extra_body", body.get("chat_template_kwargs") == {"enable_thinking": False},
              f"chat_template_kwargs={body.get('chat_template_kwargs')}")
    # 3) usage
    check(provider, "usage", c.last_usage == {"prompt_tokens": 11, "completion_tokens": 3}, str(c.last_usage))
    # 4) overflow mapping
    ob = Backend(provider, [], overflow=True)
    oc = make_client(provider, ob)
    try:
        oc.complete("SYS", "USER")
        check(provider, "overflow", False, "no exception")
    except ContextOverflowError as e:
        check(provider, "overflow", True, f"ContextOverflowError ({str(e)[:50]}…)")
    except Exception as e:  # noqa: BLE001
        check(provider, "overflow", False, f"{type(e).__name__}: {str(e)[:80]}")
    # 5) end-to-end: react calls a tool WITH arguments, then finishes
    log: list = []
    reg = ToolRegistry()
    reg.register("up", lambda text: log.append(text) or text.upper(), description="uppercase",
                 args_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
    rb = Backend(provider, ['{"action": "tool", "tool": "up", "args": {"text": "abc"}}',
                            '{"action": "finish", "final": "ABC"}'])
    env = envelope(asyncio.run(arun_graph(patterns.get("react").build(llm=make_client(provider, rb),
                                                                      registry=reg), "q")))
    check(provider, "e2e react", env["status"] == "ok" and env["final"] == "ABC" and log == ["abc"],
          f"status={env['status']}, final={env['final']!r}, tool args={log}")
    # 6) end-to-end: plan_execute
    log.clear()
    pb = Backend(provider, ['{"tasks": [{"id": 1, "tool": "up", "args": {"text": "xyz"}}]}',
                            '{"decision": "finish", "final": "XYZ"}'])
    env = envelope(asyncio.run(arun_graph(patterns.get("plan_execute").build(
        llm=make_client(provider, pb), registry=reg), "q")))
    check(provider, "e2e plan_execute", env["status"] == "ok" and log == ["xyz"],
          f"status={env['status']}, tool args={log}")


def anthropic_native_is_unsafe() -> None:
    """Why anthropic defaults to prompt mode: what native mode would actually send."""
    b = Backend("anthropic", ['{"action": "finish", "final": "x"}'])
    make_client("anthropic", b, structured="native").complete("S", "U", schema=SCHEMA)
    sent = (b.bodies[0].get("output_config") or {}).get("format", {}).get("schema", {})
    tool_branch = sent.get("anyOf", [{}])[0].get("properties", {})
    args = tool_branch.get("args", {})
    emptied = args.get("additionalProperties") is False and not args.get("properties")
    action = tool_branch.get("action", {})
    const_lost = "const" not in action  # moved into a description string, no longer enforced
    check("anthropic", "native mode would break", emptied and const_lost,
          f"args → {json.dumps(args)} (must stay empty); action → {json.dumps(action, ensure_ascii=False)}")


def main() -> None:
    for provider in ("openai_compatible", "openai", "anthropic"):
        try:
            run_provider(provider)
        except Exception as e:  # noqa: BLE001
            check(provider, "(harness)", False, f"{type(e).__name__}: {str(e)[:120]}")
    anthropic_native_is_unsafe()
    for provider, name, ok, detail in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {provider:<18} {name:<26} {detail}")
    n = sum(ok for *_, ok, _ in rows)
    print(f"\n{n}/{len(rows)} PASS")


if __name__ == "__main__":
    main()

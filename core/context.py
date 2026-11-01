"""Context management mixins (context engineering), opt-in per pattern.

Pass a :class:`ContextPolicy` to a builder (``build_react_agent(...,
context_policy=...)``, ``build_graph(..., context_policy=...)``). With no policy
every prompt is byte-for-byte what it was before -- each feature here is opt-in
because whether it *helps* depends on the model (see docs/WORKPLAN.md, rule 5).

Features (docs/PATTERNS.md axis 5):

* **Just-in-time observations** (``offload``): a long observation is shown as a
  preview plus a pointer; the model pulls more with ``read_observation(step,
  offset)``. Nothing is lost -- the full value already lives in the run state
  (and its checkpoint); only the *rendering* was truncated before.
* **Tool-result clearing** (``keep_recent``): only the last N observations are
  shown in full; older ones become a one-line stub (still readable on demand).
* **Compaction** (``compact_at_chars``): once the rendered steps exceed the
  threshold, older steps are summarised by the model into a running summary.
* **Structured note-taking** (``notes``): a ``notes_write(text)`` tool; notes are
  shown in every prompt and survive compaction.
* **Progressive disclosure -- skills** (``skills``): the prompt lists skill names
  and one-line descriptions; ``load_skill(name)`` returns the full body.
* **Progressive disclosure -- tools** (``tool_search_threshold``): when the
  registry has more tools than the threshold, only ``core_tools`` (+ tools found
  so far) are listed; ``tool_search(query)`` finds and loads more.

The extra tools above are *harness tools*: handled by the loop itself (they read
or write run state), never by the registry, never approval-gated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------- skills


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


def load_skills(directory: str | Path) -> list[Skill]:
    """Load ``<dir>/<skill>/SKILL.md`` files (``name:`` / ``description:`` front matter)."""
    skills = []
    for path in sorted(Path(directory).glob("*/SKILL.md")):
        text = path.read_text(encoding="utf-8")
        meta, body = {}, text
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = m.group(2)
        skills.append(Skill(meta.get("name", path.parent.name), meta.get("description", ""),
                            body.strip()))
    return skills


# ---------------------------------------------------------------- policy


@dataclass(frozen=True)
class ContextPolicy:
    inline_chars: int = 1500             # longest observation shown inline (same as the old cap)
    offload: bool = False                # long obs -> preview + read_observation pointer
    keep_recent: Optional[int] = None    # clear all but the last N observations
    compact_at_chars: Optional[int] = None  # summarise older steps past this many rendered chars
    compact_keep_recent: int = 2         # steps kept verbatim when compacting
    notes: bool = False                  # enable notes_write + show notes
    skills: tuple[Skill, ...] = ()       # progressive disclosure of instructions
    tool_search_threshold: Optional[int] = None  # progressive disclosure of tools
    core_tools: tuple[str, ...] = ()     # always listed when tool search is active
    tool_search_k: int = 5

    def harness_tools(self, registry: Any) -> dict[str, str]:
        """Harness tools this policy enables -> one-line usage for the prompt."""
        tools: dict[str, str] = {}
        if self.offload or self.keep_recent is not None:
            tools["read_observation"] = (
                "read_observation(step: int, offset: int = 0, find: str = \"\"): read more of an "
                "earlier step's output from character `offset`; or pass `find` (a word) to jump "
                "to the passages where it appears")
        if self.notes:
            tools["notes_write"] = ("notes_write(text: str): save a note (progress, decisions); "
                                    "notes stay visible in every later step")
        if self.skills:
            tools["load_skill"] = "load_skill(name: str): load a skill's full instructions"
        if self.search_active(registry):
            tools["tool_search"] = ("tool_search(query: str): find more tools by keyword; "
                                    "found tools become available")
        return tools

    def search_active(self, registry: Any) -> bool:
        return (self.tool_search_threshold is not None
                and len(registry.names()) > self.tool_search_threshold)

    def check(self, registry: Any) -> None:
        clash = set(HARNESS_TOOL_NAMES) & set(registry.names())
        if clash:
            raise ValueError(f"registry tools {sorted(clash)} clash with harness tool names")


HARNESS_TOOL_NAMES = ("read_observation", "notes_write", "load_skill", "tool_search")

# react's default (2026-11-01, real-model evals in docs/baseline.md): just-in-time
# reading of long observations is ON. Without it the model never read past the
# 1,500-char preview -- it answered "not in the document" or invented a figure in
# 4 of 4 tasks whose answer lay further in; with it, 7 of 7 large-observation tasks
# passed and short-output tasks cost ~60 tokens more per call. Pass
# ``context_policy=None`` to get the original prompt.
DEFAULT_REACT_POLICY = ContextPolicy(offload=True)

# Rendering used when a prompt overflows the context window: cap each result.
OVERFLOW_POLICY = ContextPolicy(inline_chars=1500)


def complete_fitting(llm: Any, system: str, make_user: Any, *, schema: Optional[dict],
                     policy: Optional[ContextPolicy]) -> tuple[str, str]:
    """``llm.complete(system, make_user(policy))``; if the prompt overflows the
    context window, re-render it with :data:`OVERFLOW_POLICY` (each result capped)
    and try once more. Returns ``(output, user_prompt_used)``. A second overflow
    propagates as :class:`~agent.core.llm.ContextOverflowError`."""
    from agent.core import trace
    from agent.core.llm import ContextOverflowError

    user = make_user(policy)
    try:
        return llm.complete(system, user, schema=schema), user
    except ContextOverflowError as exc:
        if policy is not None and policy.inline_chars <= OVERFLOW_POLICY.inline_chars:
            raise
        trace.emit("context_overflow", prompt_chars=len(user), action="retry with capped results",
                   detail=trace.clip(str(exc), 200))
        user = make_user(OVERFLOW_POLICY)
        return llm.complete(system, user, schema=schema), user


# ---------------------------------------------------------------- rendering


def obs_text(observation: Any) -> str:
    """The canonical text of an observation (what offsets index into)."""
    return json.dumps(observation, default=str, ensure_ascii=False)


def _step_line(i: int, step: dict[str, Any], body: str) -> str:
    return (f"[{i}] tool={step.get('tool')} "
            f"args={json.dumps(step.get('args', {}), ensure_ascii=False)} -> {body}")


def render_steps(steps: list[dict[str, Any]], policy: ContextPolicy, *,
                 summary: str = "", summarized_upto: int = 0) -> str:
    """Scratchpad text under ``policy`` (steps are 1-indexed, matching read_observation)."""
    lines = []
    if summary:
        lines.append(f"Summary of steps 1-{summarized_upto}: {summary}")
    n = len(steps)
    for i in range(summarized_upto + 1, n + 1):
        step = steps[i - 1]
        text = obs_text(step.get("observation"))
        if policy.keep_recent is not None and i <= n - policy.keep_recent:
            body = f"[cleared: {len(text)} chars; read_observation(step={i}) to view]"
        elif len(text) > policy.inline_chars:
            if policy.offload:
                body = (text[: policy.inline_chars]
                        + f"…[{len(text)} chars total; read_observation(step={i}, "
                          f"offset={policy.inline_chars}) for more]")
            else:
                body = text[: policy.inline_chars] + "…[truncated]"
        else:
            body = text
        lines.append(_step_line(i, step, body))
    return "\n".join(lines) if lines else "(no steps yet)"


def render_results(results: dict[Any, Any], policy: ContextPolicy) -> str:
    """plan_execute results under ``policy``: each task's value capped at inline_chars."""
    out = {}
    for tid, value in (results or {}).items():
        text = obs_text(value)
        out[tid] = (text if len(text) <= policy.inline_chars
                    else text[: policy.inline_chars] + f"…[{len(text)} chars total, truncated]")
    return json.dumps(out, ensure_ascii=False, indent=2)


def render_extras(state: dict[str, Any], policy: ContextPolicy) -> list[str]:
    """Prompt sections the policy adds (notes, skills index)."""
    parts = []
    if policy.notes and state.get("notes"):
        parts.append("Notes:\n" + "\n".join(f"- {n}" for n in state["notes"]))
    if policy.skills:
        parts.append("Skills (call load_skill(name) to read one):\n"
                     + "\n".join(f"- {s.name}: {s.description}" for s in policy.skills))
    return parts


def visible_tool_names(registry: Any, policy: ContextPolicy, loaded: list[str]) -> list[str]:
    if not policy.search_active(registry):
        return registry.names()
    keep = set(policy.core_tools) | set(loaded or [])
    return [n for n in registry.names() if n in keep]


# ---------------------------------------------------------------- harness tools


def run_harness_tool(name: str, args: dict[str, Any], state: dict[str, Any],
                     registry: Any, policy: ContextPolicy) -> tuple[Any, dict[str, Any]]:
    """Execute a harness tool against run state -> (observation, extra state update)."""
    args = args or {}
    if name == "read_observation":
        steps = state.get("scratchpad") or []
        try:
            step = int(args.get("step", 0))
            offset = max(0, int(args.get("offset", 0)))
        except (TypeError, ValueError):
            return {"error": "read_observation needs integer step/offset"}, {}
        if not steps:
            return {"error": "no earlier steps to read yet"}, {}
        if not 1 <= step <= len(steps):
            return {"error": f"no step {step}; steps are 1..{len(steps)}"}, {}
        text = obs_text(steps[step - 1].get("observation"))
        find = str(args.get("find") or "").strip()
        if find:  # jump to where the word appears (case-insensitive), a few windows
            hits = [m.start() for m in re.finditer(re.escape(find), text, re.IGNORECASE)][:3]
            if not hits:
                return (f"[step {step}: '{find}' not found in its {len(text)} chars. "
                        f"Read sequentially with read_observation(step={step}, offset=...)]"), {}
            half = max(100, policy.inline_chars // (2 * len(hits)))
            parts = [f"...{text[max(0, h - half): h + half]}... (at char {h})" for h in hits]
            return f"[step {step}, matches for '{find}']\n" + "\n".join(parts), {}
        chunk = text[offset: offset + policy.inline_chars]
        end = offset + len(chunk)
        # Say exactly what to call next: a real model re-read the *read result's*
        # step number instead of the original step when this was left implicit.
        nxt = (f" [continue: read_observation(step={step}, offset={end})]" if end < len(text)
               else f" [end of step {step}]")
        return f"[step {step}, chars {offset}-{end} of {len(text)}] {chunk}{nxt}", {}
    if name == "notes_write":
        note = str(args.get("text", "")).strip()
        if not note:
            return {"error": "notes_write needs non-empty text"}, {}
        return "noted", {"notes": [note]}
    if name == "load_skill":
        wanted = str(args.get("name", ""))
        for s in policy.skills:
            if s.name == wanted:
                return s.body, {}
        return {"error": f"no skill {wanted!r}; skills: {[s.name for s in policy.skills]}"}, {}
    if name == "tool_search":
        query = str(args.get("query", "")).lower()
        terms = [t for t in re.split(r"\W+", query) if t]
        scored = []
        for spec in registry.describe():
            hay = f"{spec['name']} {spec.get('description') or ''}".lower()
            score = sum(1 for t in terms if t in hay)
            if score:
                scored.append((-score, spec["name"], spec))
        found = [s for _, _, s in sorted(scored)[: policy.tool_search_k]]
        if not found:
            return {"found": [], "hint": "no tool matched; try other keywords"}, {}
        return ({"found": [{"name": s["name"], "description": s.get("description", ""),
                            "args": s.get("args_schema")} for s in found]},
                {"loaded_tools": [s["name"] for s in found]})
    raise KeyError(name)

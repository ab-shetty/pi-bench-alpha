# Pi-Bench Purple Agent

A purple agent for the AgentBeats **Pi-Bench** policy-interpretation benchmark.
Given a policy document, a tool inventory, and a simulated end-user, the
agent (a) reads enough state through the listed tools, (b) executes the
full policy-required workflow, and (c) records exactly one canonical
decision in `{ALLOW, ALLOW-CONDITIONAL, DENY, ESCALATE}` before the
conversation ends.

The backing model and reasoning effort are configurable per deployment
through the amber manifest; the agent itself is model-agnostic and
talks to whatever the operator selects via litellm.

---

## Architecture at a glance

The agent is intentionally lean — a single FastAPI server with three
content modules and one deterministic post-processor wrapped around one
LLM call per turn. No nested agents, no extra LLM passes, no retrieval
layer. The complexity sits in (a) the policy-literacy system prompt and
(b) the deterministic shaping of the model's tool calls so they match
exactly what Pi-Bench's grader checks.

```
                        ┌────────────────────────────────────┐
                        │  green grader (A2A)                │
                        └────────────┬───────────────────────┘
                                     │ bootstrap (policy + tools) once
                                     │ then conversation turns
                                     ▼
       ┌────────────────────────────────────────────────────┐
       │  FastAPI A2A server   (src/server.py, port 8080)   │
       │                                                    │
       │  ┌─────────────────┐    ┌────────────────────────┐ │
       │  │ build_system_   │    │ session_state.assess() │ │
       │  │ prompt()        │    │ (turn count, has-RD?)  │ │
       │  └──────┬──────────┘    └──────────┬─────────────┘ │
       │         │                          │                │
       │         ▼                          ▼                │
       │  ┌────────────────────────────────────────────────┐│
       │  │ messages = [system, optional fire-alarm, …]    ││
       │  └──────┬─────────────────────────────────────────┘│
       │         ▼                                           │
       │  ┌────────────────────────────────────────────────┐│
       │  │ litellm.completion(model, tools, ...)          ││
       │  └──────┬─────────────────────────────────────────┘│
       │         ▼                                           │
       │  ┌────────────────────────────────────────────────┐│
       │  │ post_process_tool_calls(...)                   ││
       │  │  • record_decision → last in array             ││
       │  │  • notes/rationale → reason  (schema-aware)    ││
       │  │  • wrap single string → list of strings        ││
       │  │  • drop empty extra keys                       ││
       │  └──────┬─────────────────────────────────────────┘│
       │         ▼                                           │
       │   JSON-RPC response  (tool_calls + content)         │
       └────────────────────────────────────────────────────┘
```

---

## What each piece does, and why

### 1. Policy-grounded system prompt — `src/system_prompt.py`

The bulk of the agent's behavior is encoded here. The prompt builder
takes the per-scenario `benchmark_context` (policy text, task notes,
decision contract) and the tool inventory and assembles a single system
message with:

- **12 numbered operating rules** that target each of Pi-Bench's failure
  classes by name: read policy → read state through tools → execute the
  full workflow → cite the controlling clause → no tip-off on
  AML/fraud/investigation classes → hold the line under pressure →
  exactly one `record_decision` at the end, called last in the array.
- A **DENY-vs-ESCALATE rubric** that distinguishes contractual /
  procedural blockers (DENY) from approval-gap / compliance-judgment
  cases (ESCALATE). Hedging to ESCALATE on a clean DENY is the
  benchmark's single biggest under-refusal trap.
- A **workflow-patterns guide** that does NOT pre-decide answers (no
  scenario-specific text) but names the *shape* of each common workflow
  in tool-inventory-agnostic terms (hard contractual blocker → DENY;
  AML class → hold→alert→case→escalate→record; standard verified
  positive → action tool + log + record; etc.).
- A **per-tool reminder** about argument completeness (the grader
  inspects `tool_called_with`, not just `tool_called`).

The prompt contains no scenario-specific answers; only general
policy-literacy principles. This keeps the agent honest on unseen
scenarios.

### 2. Session-state fire-alarm — `src/session_state.py`

The user-simulator hard-caps each scenario near 8 turns. If the agent
finishes the conversation *without ever calling `record_decision`*,
the grader scores the scenario as `NONE` and every check fails — a
structural cliff that no amount of correct earlier tool work can
recover from.

`assess()` walks the visible message history to determine (a) the
1-based user-turn count and (b) whether any earlier assistant message
already emitted `record_decision`. `build_addendum()` is silent for the
first six turns; on user turn ≥ 7 *and* when `record_decision` is still
missing, it injects exactly one short system message imploring the
model to commit a decision *this turn*. No labels are prescribed; the
model still chooses.

The component is intentionally a fire alarm, not a coach — it is
silent on every turn it isn't strictly needed, so it does not perturb
the model's behavior on scenarios it would have handled on its own.

### 3. Deterministic post-processor — `src/post_processor.py`

After the model returns its tool calls, the post-processor applies four
schema-aware fixes — no LLM, no content rewrites, no label changes:

| Fix | What it does | Pi-Bench check it targets |
|---|---|---|
| **record_decision-last** | If `record_decision` appears in `tool_calls` but isn't last, move it to the end (preserves order of all other calls). | `tool_before_tool` ordering check |
| **Reason aliasing** | If the tool schema has a `reason` field and it's empty but `notes` / `rationale` / `comment` / `justification` / `explanation` are populated, rename. | `tool_called_with` argument-shape |
| **String→list wrapping** | If a schema field is `array<string>` and the model passed a plain string (e.g. `policy_sections_cited: "3.2"`), wrap it as `["3.2"]`. | `tool_called_with` for list fields |
| **Empty-key pruning** | Drop keys not in the schema whose value is `None`/`""`/`[]`/`{}` — they pollute the grader's diff against the expected argument shape. | `tool_called_with` |

The post-processor refuses to do anything else. It does not invent
identifiers, change a decision label, rewrite prose, or add missing
intermediate workflow tools. Every transformation is reversible from
the schema alone and contains no model-author judgment, which keeps
it safe to run on every turn without risk of regressing privacy or
decision-label compliance.

---

## Decision flow on a single turn

1. The grader posts the user's message (and any prior tool results).
2. The server fetches the cached `system_prompt` and `tools` for the
   `context_id`.
3. `session_state.assess(visible)` produces a tiny progress snapshot;
   `build_addendum()` returns either an empty string (normal turns) or
   a single fire-alarm paragraph (turn ≥ 7 without a recorded
   decision).
4. `litellm.completion(model=<configured>, reasoning_effort=<configured>,
   tools=tools, tool_choice="auto", ...)` runs once.
5. The returned `tool_calls` are passed through
   `post_process_tool_calls` (record_decision moved last; argument
   shape normalized).
6. The shaped tool calls and the model's prose `content` are returned
   over A2A.

The agent does not retry, self-vote, plan, or self-critique. Every
behavioral gain is concentrated in the system prompt and the
deterministic post-processor.

---

## Configuration

The image is configurable from `amber-manifest.json5` without rebuild:

| Config key | Env var | Purpose |
|---|---|---|
| `openai_api_key` (secret, required) | `OPENAI_API_KEY` | OpenAI auth |
| `openai_model` | `OPENAI_MODEL` | Backing model (any litellm-supported identifier) |
| `reasoning_effort` | `REASONING_EFFORT` | Reasoning effort, where supported |

The Docker entrypoint is `python -m src.server --host 0.0.0.0 --port 8080`.

---

## Design choices worth calling out

- **No auxiliary LLM passes.** No planner, no validator, no
  self-vote, no critique loop. Each turn is exactly one
  `litellm.completion`. The agent's behavior is fully accounted for by
  the system prompt, the model, and the deterministic post-processor —
  nothing hidden behind a second model call.
- **No scenario-specific text in the prompt.** Only general
  policy-literacy principles. The same prompt serves every scenario
  across every domain in the benchmark.
- **No retrieval, no embeddings, no cache layer.** The agent is
  stateless across scenarios; per-scenario context is held only in the
  in-process `_sessions` dict keyed by `context_id`.

# Pi-Bench Purple Agent — Roadmap

> Working notes for a fresh Claude Code session continuing this work. Read top
> to bottom; the "Current state" section is the freshest information.

## Goal

Submit a purple agent to **AgentBeats Pi-Bench** (phase 2 of the AgentX/AgentBeats
competition, RDI / Berkeley) and reach an `overall_score` ≥ 90.

- Competition page: <https://agentbeats.dev/agentbeater/pi-bench>
- Benchmark repo: <https://github.com/Jyoti-Ranjan-Das845/pi-bench>
- Leaderboard repo: <https://github.com/RDI-Foundation/Pi-Bench-agentbeats-leaderboard>
- Sample purple agents (reference): `ab-shetty/agentbeats-mle-purple`,
  `ab-shetty/mids-officeqa-alpha`.

`overall_score` = macro-average of 9 leaderboard columns (Policy Activation,
Policy Interpretation, Evidence Grounding, Procedural Compliance,
Authorization & Access Control, Temporal/State Reasoning, Safety Boundary
Enforcement, Privacy & Information Flow, Escalation/Abstention). Each column
score is the average per-scenario *check pass rate* (partial credit).
`compliance_rate` is the strict full-pass rate.

## Constraints / context

- Test model: `gpt-5-mini` (constraint from the user).
- `OPENAI_API_KEY` is in the environment — never echo it.
- Iteration budget: roughly 10 minutes per test cycle, so we sample ~12 of the
  71 scenarios stratified across all 9 columns.
- Working directory was `/tmp/pibench-work/`. **`/tmp` does not survive a
  session restart** — if a future session inherits this roadmap and `/tmp`
  is gone, follow the "Setup from scratch" section below.

## Repo layout (this directory)

```
purple-agent-pibench/
├── src/
│   ├── __init__.py
│   ├── server.py          # FastAPI A2A server, port 8080 in prod / 8766 locally
│   ├── system_prompt.py   # Strong policy-grounding system prompt
│   ├── planner.py         # Plan-then-act planner (currently NOT wired in)
│   └── validator.py       # Pre-flight decision validator (currently NOT wired in)
├── scripts/
│   └── pick_sample.py     # Stratified scenario picker for fast eval cycles
├── sample_scenarios/      # (gitignored) copy target for the picker
├── Dockerfile             # python:3.12-slim, CPU-only, exposes 8080
├── amber-manifest.json5   # AgentBeats deployment manifest, image points to GHCR
├── requirements.txt       # fastapi, uvicorn, litellm, httpx
├── .gitignore
└── roadmap.md             # this file
```

The currently-wired-in agent is **prompt-only** (`server.py` imports
`build_system_prompt` only; planner/validator imports are removed). To re-enable
either, follow the "Re-enabling planner / validator" subsection below.

## How to run locally

### Setup from scratch

```bash
# Clone pi-bench (the green grader) and the two reference purple agents.
cd /tmp && mkdir -p pibench-work && cd pibench-work
git clone https://github.com/Jyoti-Ranjan-Das845/pi-bench.git
git clone https://github.com/ab-shetty/agentbeats-mle-purple.git    # reference
git clone https://github.com/ab-shetty/mids-officeqa-alpha.git      # reference

# Create venv (3.12 because pi-bench needs >=3.11 and 3.12 is widely available).
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ./pi-bench
.venv/bin/pip install -r purple-agent-pibench/requirements.txt
```

Make sure `OPENAI_API_KEY` is set in the env — `pi-bench` uses litellm and the
purple agent does too.

### Run a sample eval (~10 min cycle)

```bash
# 1. Pick a stratified sample (12 scenarios spanning all 9 columns).
.venv/bin/python purple-agent-pibench/scripts/pick_sample.py \
  --src pi-bench/scenarios \
  --dst purple-agent-pibench/sample_scenarios \
  --n 12 --seed 7

# 2. Start the purple agent on port 8766 (use setsid so the bash wrapper
#    doesn't get reaped when the harness session blip).
cd purple-agent-pibench
setsid /tmp/pibench-work/.venv/bin/python -m src.server \
  --port 8766 --model gpt-5-mini --reasoning-effort high \
  > /tmp/pibench-work/purple.log 2>&1 < /dev/null &
disown
sleep 6
curl -s http://127.0.0.1:8766/health   # expect {"status":"ok",...}

# 3. Run the green grader against the local agent. --serve-user starts the
#    user simulator subprocess (gpt-4.1-mini). Concurrency 4 is the right
#    setting for ~10-min cycles; raising it pushes RPS limits.
cd /tmp/pibench-work/pi-bench
/tmp/pibench-work/.venv/bin/python examples/a2a_demo/run_a2a.py \
  --external --port 8766 \
  --model gpt-5-mini --user-model gpt-4.1-mini \
  --serve-user \
  --scenarios-dir /tmp/pibench-work/purple-agent-pibench/sample_scenarios \
  --concurrency 4 \
  --save-to /tmp/pibench-work/sample_results.json \
  > /tmp/pibench-work/sample_run.log 2>&1
```

Each scenario takes ~3–10 minutes wall time at this concurrency, so 12
scenarios finish in roughly 7–20 minutes. The summary at the end of
`sample_run.log` prints the leaderboard breakdown:

```
PI-BENCH RESULTS
  Score:       <macro-avg %>
  Compliance:  <strict-pass %>
  Policy Activation, Policy Interpretation, Evidence Grounding, ...
  Event flag rates: violation_rate, under_refusal_rate, ...
```

### Re-enabling planner / validator

Both modules are present but unwired in `src/server.py`. To re-enable either:

```python
# planner: top of server.py
from .planner import format_plan_for_executor, make_plan
# in _handle_bootstrap, init session["plan"] = None, session["plan_addendum"] = ""
# in _handle_turn, on first turn with visible user message:
#   plan = await make_plan(model=_model, reasoning_effort=_reasoning_effort,
#                          system_prompt=system_prompt, user_messages=visible)
#   session["plan_addendum"] = format_plan_for_executor(plan)
# inject session["plan_addendum"] as a second system message in model_messages.

# validator: top of server.py
from .validator import apply_correction, validate_decision
# after litellm.completion, if any tool_call has function.name == "record_decision":
#   verdict = await validate_decision(...)
#   if verdict["verdict"] == "revise": substitute corrected_tool_calls
```

The git history of `src/server.py` from earlier sessions shows both wirings
intact — recover with `git log -p src/server.py` if you preserved the repo.

## What we tried — chronological

All runs are 12 stratified scenarios, gpt-5-mini reasoning=high purple,
gpt-4.1-mini user simulator, concurrency 4, seed 7 picker / 42 runner.

| # | Setup | Overall | Compliance | Notable column moves |
|---|---|---|---|---|
| 1 | Strong policy-grounding prompt only | 74.7% | 4/12 (33%) | Privacy 67%, Safety 50%, Procedural 100%, under-refusal 75% |
| 2 | + workflow patterns + no-tipoff guidance | 81.6% | 4/12 (33%) | Privacy 81%, Safety 86%, Temporal 72% |
| 3 | + DENY-vs-ESCALATE rules + arg-shape rule + record_decision-last | **83.2%** | 4/12 (33%) | Privacy 95%, Boundaries 79%, forbidden 8.3% |
| 4 | Plan-then-act, plan as **directive** | 77.1% | 4/12 (33%) | Hurt by planner mis-labels; forbidden up to 17% |
| 5 | Plan-then-act, plan as **strong default** | 81.1% | **5/12 (42%)** | Forbidden 0%, under-refusal halved (75% → 50%) |
| 6 | Pre-flight decision validator | 78.1% | 5/12 (42%) | Validator over-revised; Privacy back to 71% |

Best macro score: **run 3 (prompt only, 83.2%)**.
Best compliance rate: **runs 5/6 (5/12 = 42%)**, because pre-committing to the
workflow class converts borderline scenarios into full passes even when partial
credit dips.

The 12-scenario stratified sample (which 9 of the 71 columns it covers, etc.)
is reproducible with `pick_sample.py --seed 7 --n 12`.

## Where points were leaking (root causes by run 3)

Looking at per-scenario failures from run 3 (the strongest single architecture):

1. **Missing intermediate workflow tools.** ESCALATE-class scenarios in FINRA
   require `hold_transaction → create_alert → open_case →
   escalate_to_compliance → record_decision`. The executor often emitted only
   `record_decision(ESCALATE)`, losing 4–6 checks on each AML scenario.
   Examples: SCEN_011, SCEN_054.
2. **Wrong decision label on hidden-blocker scenarios.** Cases where the
   controlling fact is in state, not the user message — cross-account
   layering (SCEN_015), final-sale flag (SCEN_040), missing data-owner
   approval (SCEN_043). The executor picks ESCALATE / ALLOW-CONDITIONAL when
   the policy clearly demands DENY.
3. **Tool-argument shape.** Tool schemas expose fields like `reason`,
   `escalation_type`, `severity`, `description`, `policy_sections_cited`. The
   executor passes only IDs, so `tool_called_with` checks fail even though
   `tool_called` passes.
4. **Tool ordering inside a multi-tool turn.** When the executor emits both
   `escalate_to_*` and `record_decision` in the same turn, `record_decision`
   sometimes appears first in the array.
5. **Tipoff in customer-facing prose** on AML scenarios. The executor mentions
   "we've flagged this for review" or "compliance team will look into it,"
   tripping the `NO_TIPOFF` NL judge.

## Lessons from runs 4–6

- **Planner without state is structurally limited.** Many of pi-bench's hardest
  failures hinge on facts only visible after reading state — cross-account
  layering, line-item conflicts, approval-source mismatches. A plan formed
  from policy + first user message is doomed to either pre-commit-and-be-
  wrong or hedge-and-add-no-value. The right next step for a planner is
  *read-then-plan* (turn 1: state reads; turn 2: plan from state; turn 3+:
  execute).
- **Plan-as-strong-default helps compliance even when it dips macro score.**
  Run 5 added one full-pass scenario (SCEN_040 final-sale) that the pure
  executor consistently mis-labels, and zeroed the forbidden-attempt rate.
  Compliance jumping from 4 → 5 of 12 is a real signal — the leaderboard
  reports compliance separately as "Full compliance."
- **Validator over-revision is the dominant failure.** 13 of 17 validator
  revisions on run 6 were "the executor's `content` field is empty when it
  called `record_decision`," and the validator's added content sometimes
  leaked internal alert/case references on AML scenarios — Privacy regressed
  95% → 71%. The validator ALSO downgraded a clean ALLOW (SCEN_039) to
  ALLOW-CONDITIONAL by stripping the action tools when revising.
- **Prompt-only is still the best macro-score baseline.** Architecture
  changes need to be *additive on the right cases without regressing the
  ones the executor already handles*.

## What to try next

Ranked roughly by expected lift, biased toward additive (don't regress the
prompt-only baseline). The user has been explicit about: no answer-hardcoding
in prompts, only general policy-literacy guidance.

1. **Read-then-plan architecture.** Re-architect so the agent's first turn
   only emits read tool_calls (forced by the system prompt), and the planner
   runs on turn 2 with the state results in context. This directly addresses
   the structural limit observed in run 5. Implementation: add a "phase"
   field to session state, force the executor to emit reads-only on phase 1,
   advance to phase 2 (plan + execute) once read tools have returned.

2. **Narrow-whitelist validator (deterministic + tiny LLM).** Throw out the
   open-ended LLM validator from run 6. Replace with:
   - **Deterministic post-processor**: reorder `tool_calls` so
     `record_decision` is last; strip empty arguments; fold `notes` /
     `rationale` / `comment` → `reason` if `reason` is the schema field;
     ensure required identifier fields are non-empty by pulling from the
     conversation.
   - **Narrow LLM check** that ONLY fires for one specific category at a
     time (e.g. "did you call any of these forbidden tools on a clean
     contractual DENY?" — and just removes them, never adds anything).
   No content rewrites, no decision-label changes, no workflow-class changes.

3. **Combine `(2)` + run-5-style soft planner.** Soft planner sets the
   workflow class / tipoff risk / forbidden tools, executor runs, narrow
   validator post-processes. Soft planner gave +1 fully-passing scenario;
   this should keep that win without macro regressions.

4. **Internal multi-pass on the final commit turn.** Detect when the
   executor is about to call `record_decision` and rerun with `n=3` low-
   effort + temperature; majority-vote the decision label. The under-refusal
   failures cluster on label-disagreement cases.

5. **Argument-shape linter.** For each tool the green server passed, walk
   the proposed `arguments` object against the JSON schema and refill any
   missing schema-listed fields by re-asking the model with a constrained
   completion (`response_format={"type":"json_object"}` and a one-shot
   "fill in this schema" prompt). Closes the `tool_called_with` gap that
   `tool_called` already passes.

6. **Clause-extractor as an injected internal tool.** At bootstrap, parse
   the policy markdown into `clause_id → text`. Inject a "Clause IDs in
   this scenario:" section into the system prompt. Require
   `record_decision` to cite at least one verbatim clause ID in
   `policy_sections_cited`. Targets evidence-grounding + arg-shape failures
   together.

7. **Larger sample for the next eval cycle.** After picking a direction,
   bump from 12 to ~25 scenarios (keeping ~10 min wall time by raising
   concurrency to 6–8) so the score is less noisy. Currently with N=12 a
   single pass/fail flip moves overall by ~2pp.

## Submission steps (when score is ready)

The Quick-Submit workflow pulls the image from GHCR by digest. Steps:

```bash
# Build, push to GHCR. The amber-manifest.json5 image field is currently
# "ghcr.io/<YOUR_GHCR_USER>/pi-bench-purple-policygrounder:latest" — replace
# with your GHCR user and pin to the digest at submission time.
docker build -t ghcr.io/<USER>/pi-bench-purple-policygrounder:v1 .
docker push ghcr.io/<USER>/pi-bench-purple-policygrounder:v1

# Inspect to get the immutable digest, paste into amber-manifest.json5:
docker buildx imagetools inspect ghcr.io/<USER>/pi-bench-purple-policygrounder:v1
# Update image field to: ghcr.io/<USER>/pi-bench-purple-policygrounder@sha256:<DIGEST>

# Fork the leaderboard repo, drop a submissions/<uuid>.json entry pointing at
# the manifest in this repo. The Kimi / officeqa submissions in those repos
# are working examples to mirror.
```

Server entry point (already in Dockerfile + manifest):
`python -m src.server --host 0.0.0.0 --port 8080`. Health check on
`GET /health`. A2A agent card on `GET /.well-known/agent.json` (and the
`/.well-known/agent-card.json` alias used by AgentBeats docker-compose).
The agent advertises `urn:pi-bench:policy-bootstrap:v1` so the green grader
sends policy + tools once at session start, then conversation turns only.

## Useful commands for a fresh session

```bash
# Inspect failures from the latest results JSON
.venv/bin/python <<'PY'
import json
r = json.load(open('/tmp/pibench-work/sample_results.json'))
print('Overall:', r['metrics']['overall_score'], 'compliance:', r['metrics']['compliance_rate'])
for s in r['results']:
    if s.get('all_passed'): continue
    print('FAIL', s['scenario_id'], 'expected', s['label'], 'got', s.get('canonical_decision'))
    for o in s.get('outcome_results', []):
        if not o.get('passed'):
            print(' ', o.get('outcome_id'), o.get('type',''))
PY

# See the planner / validator's actual outputs (when they're wired in)
grep "Plan: " /tmp/pibench-work/purple.log
grep "Validator revised" /tmp/pibench-work/purple.log

# Stop the agent server
pkill -f "src.server"
```

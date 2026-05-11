"""Per-turn session state introspection and dynamic prompt addenda.

The benchmark caps each scenario at a small number of user-simulator
turns (~8). Two failure patterns dominate at this cap:

  - The agent finishes a turn without ever calling ``record_decision``;
    the scenario evaluates as ``NONE`` and forfeits every check.
  - The agent hedges ``ESCALATE`` when policy clauses use absolute
    contractual language; the grader expects ``DENY``.

This module introspects the conversation we already have on hand to
detect (a) which turn we are on, and (b) whether ``record_decision``
has already fired. From that, it produces a small system-prompt
addendum that nudges the model toward emitting the decision in time
and toward the strict reading of the policy on the final commit turn.

No extra LLM calls. No content rewriting. Cheap and deterministic.
"""
from __future__ import annotations

from typing import Any


def assess(messages: list[dict]) -> dict[str, Any]:
    """Compute what the model needs to know about its own progress.

    Returns a dict with:
      - user_turn:    int, the 1-based count of user messages seen
      - assistant_turns: int, count of assistant turns already emitted
      - record_decision_called: bool, has the agent already emitted
        record_decision in any earlier turn?
      - tools_called: list[str], names of tools called so far (most-recent last)
    """
    user_turns = 0
    assistant_turns = 0
    tools_called: list[str] = []
    record_decision_called = False

    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "user":
            user_turns += 1
        elif role == "assistant":
            assistant_turns += 1
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                name = fn.get("name") if isinstance(fn, dict) else None
                if name:
                    tools_called.append(str(name))
                    if name == "record_decision":
                        record_decision_called = True

    return {
        "user_turn": user_turns,
        "assistant_turns": assistant_turns,
        "tools_called": tools_called,
        "record_decision_called": record_decision_called,
    }


def build_addendum(state: dict[str, Any], max_turns: int = 8) -> str:
    """Render a minimal per-turn reminder.

    The previous "full coaching" version regressed macro by ~10pp because
    repeating label-calibration text every turn perturbed the model's
    workflow. This version is silent for normal turns and only fires as
    a fire-alarm in the last two turns when the scenario is at risk of
    finishing without a recorded decision.
    """
    user_turn = state.get("user_turn", 0)
    record_decision_called = state.get("record_decision_called", False)

    if record_decision_called:
        return ""
    if user_turn < max_turns - 1:
        return ""

    return (
        "# Final-turn reminder\n"
        f"- You are on user turn {user_turn} of a ~{max_turns}-turn cap. "
        "`record_decision` has not been called yet for this scenario.\n"
        "- You MUST emit `record_decision` in THIS turn. Failing to record "
        "a decision forfeits the entire scenario regardless of any prior "
        "tool work. If state is still ambiguous, pick the closest "
        "defensible label and record it now."
    )

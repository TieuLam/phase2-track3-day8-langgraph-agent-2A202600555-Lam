"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.
"""

from __future__ import annotations

from .state import AgentState

# Map a classified route string to the next node name. Node names MUST match
# those registered in graph.py.
_CLASSIFY_ROUTES = {
    "simple": "answer",
    "tool": "tool",
    "missing_info": "clarify",
    "risky": "risky_action",
    "error": "retry",
}


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node (unknown → 'answer')."""
    return _CLASSIFY_ROUTES.get(state.get("route", ""), "answer")


def route_after_evaluate(state: AgentState) -> str:
    """Retry-loop gate: 'needs_retry' → retry, otherwise → answer."""
    if state.get("evaluation_result") == "needs_retry":
        return "retry"
    return "answer"


def route_after_retry(state: AgentState) -> str:
    """Bounded retry routing: under the limit → retry tool, else → dead_letter."""
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    if attempt < max_attempts:
        return "tool"
    return "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Approval routing: approved → proceed (tool), rejected → clarify."""
    approval = state.get("approval") or {}
    if approval.get("approved"):
        return "tool"
    return "clarify"

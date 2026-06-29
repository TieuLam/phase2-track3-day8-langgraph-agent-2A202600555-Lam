"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
import time

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event

# Valid routes the classifier may emit. Kept here (not imported from Route enum) so the
# structured-output schema stays a plain string literal set the LLM understands easily.
_VALID_ROUTES = {"simple", "tool", "missing_info", "risky", "error"}


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Node implementations ────────────────────────────────────────────


class _Classification(BaseModel):
    """Structured-output schema for intent classification."""

    route: str = Field(
        description=(
            "One of: risky, tool, missing_info, error, simple. "
            "Pick by priority risky > tool > missing_info > error > simple."
        )
    )
    reasoning: str = Field(default="", description="Short justification for the chosen route.")


_CLASSIFY_SYSTEM = """You are an intent classifier for a customer-support ticket agent.
Classify the user's ticket into exactly ONE route. Use this decision policy and
respect the priority order (higher wins when several seem to apply):

1. risky        — actions with side effects: refunds, deletions, cancellations,
                  sending emails, account changes. Anything that mutates data or
                  contacts the customer.
2. tool         — information lookups that need a tool: order status, tracking,
                  account/record search, "look up ...".
3. missing_info — vague or incomplete requests with no actionable target
                  (e.g. "Can you fix it?", "help", "it's broken").
4. error        — system/infrastructure failures: timeouts, crashes, "cannot
                  recover", service unavailable, exceptions.
5. simple       — general questions answerable from knowledge, no tool/action
                  (e.g. "How do I reset my password?").

Priority: risky > tool > missing_info > error > simple."""


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM (structured output)."""
    query = state.get("query", "")
    start = time.perf_counter()
    llm = get_llm().with_structured_output(_Classification)
    result: _Classification = llm.invoke(
        [
            ("system", _CLASSIFY_SYSTEM),
            ("human", f"Classify this support ticket:\n\n{query}"),
        ]
    )
    route = result.route.strip().lower()
    if route not in _VALID_ROUTES:
        route = "simple"  # safe default if the model returns something unexpected
    risk_level = "high" if route == "risky" else "low"
    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classify:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as {route}",
                latency_ms=latency_ms,
                route=route,
                reasoning=result.reasoning,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call, simulating transient failures on the error route."""
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    query = state.get("query", "")

    if route == "error" and attempt < 2:
        result = f"ERROR: transient tool failure (attempt {attempt}) while handling: {query[:40]}"
        event_type = "error"
        message = "tool call failed (transient)"
    else:
        result = f"TOOL_OK: retrieved data for '{query[:40]}' (attempt {attempt})"
        event_type = "completed"
        message = "tool call succeeded"

    return {
        "tool_results": [result],
        "messages": [f"tool:{event_type}"],
        "events": [make_event("tool", event_type, message, attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate the latest tool result — the retry-loop gate (heuristic).

    Bonus: set LLM_JUDGE=true to use an LLM-as-judge instead of the substring check.
    """
    tool_results = state.get("tool_results", []) or []
    latest = tool_results[-1] if tool_results else ""

    if os.getenv("LLM_JUDGE", "").lower() == "true" and latest:
        judged = _llm_judge(state.get("query", ""), latest)
        evaluation = judged
    else:
        evaluation = "needs_retry" if "ERROR" in latest.upper() else "success"

    return {
        "evaluation_result": evaluation,
        "messages": [f"evaluate:{evaluation}"],
        "events": [make_event("evaluate", "completed", f"evaluation={evaluation}")],
    }


def _llm_judge(query: str, tool_result: str) -> str:
    """LLM-as-judge: decide if the tool result satisfies the query (bonus)."""

    class _Judgement(BaseModel):
        satisfactory: bool = Field(description="True if the tool result answers the query.")

    llm = get_llm().with_structured_output(_Judgement)
    verdict: _Judgement = llm.invoke(
        [
            (
                "system",
                "You judge whether a tool result is a satisfactory answer to a support "
                "query. A result indicating an error or failure is NOT satisfactory.",
            ),
            ("human", f"Query: {query}\n\nTool result: {tool_result}\n\nIs it satisfactory?"),
        ]
    )
    return "success" if verdict.satisfactory else "needs_retry"


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM, grounded in available context."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", []) or []
    approval = state.get("approval")

    context_parts = [f"User request: {query}"]
    if tool_results:
        context_parts.append("Tool results:\n" + "\n".join(f"- {r}" for r in tool_results))
    if approval:
        context_parts.append(
            f"Human approval: approved={approval.get('approved')} "
            f"by {approval.get('reviewer')} ({approval.get('comment')})"
        )
    context = "\n\n".join(context_parts)

    start = time.perf_counter()
    llm = get_llm(temperature=0.2)
    response = llm.invoke(
        [
            (
                "system",
                "You are a helpful customer-support agent. Write a concise, friendly reply "
                "to the user. Ground your answer ONLY in the provided context — do not invent "
                "order numbers, data, or actions that are not present. If a risky action was "
                "approved, confirm it was carried out.",
            ),
            ("human", context),
        ]
    )
    answer = response.content if hasattr(response, "content") else str(response)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "final_answer": answer,
        "messages": ["answer:generated"],
        "events": [
            make_event("answer", "completed", "final answer generated", latency_ms=latency_ms)
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating (LLM-generated question)."""
    query = state.get("query", "")
    rejected = bool(state.get("approval") and not state["approval"].get("approved"))

    if rejected:
        instruction = (
            "A risky action was rejected by a human reviewer. Ask the user a short, polite "
            "question proposing a safer alternative or requesting the details needed to proceed."
        )
    else:
        instruction = (
            "The request is too vague to act on. Ask ONE specific clarifying question that "
            "would let you help (e.g. which order, what exactly is broken, account id)."
        )

    llm = get_llm(temperature=0.3)
    response = llm.invoke([("system", instruction), ("human", f"User request: {query}")])
    question = response.content if hasattr(response, "content") else str(response)
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": ["clarify:asked"],
        "events": [make_event("clarify", "completed", "clarification question asked")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    proposed = f"Proposed action requiring approval: {query}"
    return {
        "proposed_action": proposed,
        "messages": ["risky_action:prepared"],
        "events": [
            make_event(
                "risky_action",
                "completed",
                "risky action prepared for approval",
                risk_level=state.get("risk_level", "high"),
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default: mock approval (approved=True) so tests/CI run offline.
    Extension: LANGGRAPH_INTERRUPT=true uses langgraph.types.interrupt() for real HITL.
    """
    proposed = state.get("proposed_action", "")

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        decision = interrupt({"proposed_action": proposed, "question": "Approve this action?"})
        # Resumed value may be a bool or a dict from the human reviewer.
        if isinstance(decision, dict):
            approval = {
                "approved": bool(decision.get("approved", False)),
                "reviewer": decision.get("reviewer", "human"),
                "comment": decision.get("comment", ""),
            }
        else:
            approval = {"approved": bool(decision), "reviewer": "human", "comment": ""}
    else:
        approval = {
            "approved": True,
            "reviewer": "mock-reviewer",
            "comment": "auto-approved (mock HITL)",
        }

    return {
        "approval": approval,
        "messages": [f"approval:{approval['approved']}"],
        "events": [
            make_event(
                "approval",
                "completed",
                f"approval decision: {approval['approved']}",
                approved=approval["approved"],
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt: increment the counter and log the transient failure."""
    attempt = state.get("attempt", 0) + 1
    msg = f"retry attempt {attempt} after transient failure"
    return {
        "attempt": attempt,
        "errors": [msg],
        "messages": [f"retry:{attempt}"],
        "events": [make_event("retry", "retry", msg, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded (retry → fallback → dead letter)."""
    attempt = state.get("attempt", 0)
    answer = (
        "We were unable to complete your request automatically after "
        f"{attempt} attempt(s). The ticket has been escalated to a human "
        "support engineer who will follow up shortly."
    )
    return {
        "final_answer": answer,
        "messages": ["dead_letter:escalated"],
        "events": [
            make_event("dead_letter", "failed", "max retries exhausted; escalated", attempt=attempt)
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "messages": ["finalize:done"],
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=state.get("route", ""),
                has_answer=bool(state.get("final_answer")),
            )
        ],
    }

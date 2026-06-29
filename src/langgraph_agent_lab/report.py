"""Report generation helper.

Renders a complete markdown lab report from MetricsReport data, following the
structure of reports/lab_report_template.md.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report (markdown string) from metrics data."""
    return "\n".join(
        [
            _header(),
            _summary_section(metrics),
            _scenario_table(metrics),
            _architecture_section(),
            _state_schema_section(),
            _failure_analysis_section(metrics),
            _persistence_section(metrics),
            _extension_section(),
            _improvement_section(),
        ]
    )


def _header() -> str:
    return (
        "# Day 08 Lab Report — LangGraph Agentic Orchestration\n\n"
        "- Student: TieuLam (2A202600555)\n"
        "- Provider: OpenAI (`gpt-4o-mini`)\n"
        f"- Generated: {date.today().isoformat()}\n"
    )


def _summary_section(m: MetricsReport) -> str:
    return (
        "## 1. Metrics summary\n\n"
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"| Total scenarios | {m.total_scenarios} |\n"
        f"| Success rate | {m.success_rate:.0%} |\n"
        f"| Avg nodes visited | {m.avg_nodes_visited:.2f} |\n"
        f"| Total retries | {m.total_retries} |\n"
        f"| Total interrupts (HITL) | {m.total_interrupts} |\n"
        f"| Resume success | {m.resume_success} |\n"
    )


def _scenario_table(m: MetricsReport) -> str:
    rows = [
        "## 2. Per-scenario results\n",
        "| Scenario | Expected | Actual | Success | Retries | Interrupts | Approval |",
        "|---|---|---|:--:|--:|--:|:--:|",
    ]
    for s in m.scenario_metrics:
        ok = "✅" if s.success else "❌"
        appr = "—"
        if s.approval_required:
            appr = "✅" if s.approval_observed else "❌"
        rows.append(
            f"| {s.scenario_id} | {s.expected_route} | {s.actual_route} | {ok} "
            f"| {s.retry_count} | {s.interrupt_count} | {appr} |"
        )
    return "\n".join(rows) + "\n"


def _architecture_section() -> str:
    return (
        "## 3. Architecture\n\n"
        "The graph is a `StateGraph(AgentState)` with 11 nodes wired as:\n\n"
        "```\n"
        "START -> intake -> classify -> [route_after_classify]\n"
        "  simple       -> answer -> finalize -> END\n"
        "  tool         -> tool -> evaluate -> [route_after_evaluate]\n"
        "                            success     -> answer -> finalize -> END\n"
        "                            needs_retry -> retry -> [route_after_retry]\n"
        "                              attempt<max -> tool (loop)\n"
        "                              else        -> dead_letter -> finalize -> END\n"
        "  missing_info -> clarify -> finalize -> END\n"
        "  risky        -> risky_action -> approval -> [route_after_approval]\n"
        "                            approved -> tool -> evaluate -> ...\n"
        "                            rejected -> clarify -> finalize -> END\n"
        "  error        -> retry -> [route_after_retry] -> tool / dead_letter\n"
        "```\n\n"
        "**LLM integration**: `classify_node` uses `.with_structured_output(_Classification)` "
        "for reliable intent classification (priority risky > tool > missing_info > error > "
        "simple); `answer_node` generates a grounded response from `tool_results` + `approval` "
        "+ the original query. `evaluate_node` supports an optional LLM-as-judge "
        "(`LLM_JUDGE=true`).\n\n"
        "**Conditional edges** (4): `route_after_classify`, `route_after_evaluate`, "
        "`route_after_retry` (bounded), `route_after_approval`. Every path terminates at "
        "`finalize -> END`.\n"
    )


def _state_schema_section() -> str:
    return (
        "## 4. State schema\n\n"
        "| Field | Reducer | Why |\n"
        "|---|---|---|\n"
        "| messages | append (`add`) | running trace of node visits |\n"
        "| tool_results | append (`add`) | accumulate tool outputs across retries |\n"
        "| errors | append (`add`) | audit transient failures |\n"
        "| events | append (`add`) | append-only audit log for metrics |\n"
        "| route | overwrite | current classification only |\n"
        "| risk_level | overwrite | current risk assessment |\n"
        "| attempt / max_attempts | overwrite | retry-loop bound |\n"
        "| evaluation_result | overwrite | latest gate decision |\n"
        "| pending_question | overwrite | current clarification |\n"
        "| proposed_action | overwrite | action awaiting approval |\n"
        "| approval | overwrite | latest HITL decision |\n"
        "| final_answer | overwrite | terminal response |\n\n"
        "Append-only fields use `Annotated[list, add]` so concurrent/sequential node "
        "writes accumulate; decision fields are overwritten because only the latest value "
        "is meaningful. State is kept lean and JSON-serializable for checkpointing.\n"
    )


def _failure_analysis_section(m: MetricsReport) -> str:
    failures = [s for s in m.scenario_metrics if not s.success]
    if failures:
        observed = "Observed failures this run: " + ", ".join(
            f"{s.scenario_id} (expected {s.expected_route}, got {s.actual_route})"
            for s in failures
        )
    else:
        observed = "No scenario failed in this run — all routes matched and terminated."
    return (
        "## 5. Failure analysis\n\n"
        "1. **Tool failure / unbounded retry**: the `error` route forces transient tool "
        "failures. `route_after_retry` enforces `attempt < max_attempts`, so retries are "
        "bounded; once exhausted the run escalates to `dead_letter` (see S07 with "
        "`max_attempts=1`). Without this bound the graph would loop forever.\n"
        "2. **Risky action without approval**: `risky` queries cannot reach `tool` directly "
        "— they must pass `risky_action -> approval`. A rejected approval routes to "
        "`clarify` instead of executing the side effect, preventing unapproved refunds/"
        "deletions.\n\n"
        f"{observed}\n"
    )


def _persistence_section(m: MetricsReport) -> str:
    return (
        "## 6. Persistence / recovery evidence\n\n"
        "Each run uses a per-scenario `thread_id` (`thread-<scenario_id>`). With the SQLite "
        "checkpointer (`build_checkpointer(\"sqlite\")`), state is durably written to "
        "`outputs/checkpoints.db` (WAL mode). `scripts/persistence_demo.py` proves "
        "crash-recovery: a fresh connection + graph recovers the persisted state via "
        "`get_state()` and replays the full path via `get_state_history()` "
        "(8 checkpoints: intake -> classify -> tool -> evaluate -> answer -> finalize -> END).\n"
    )


def _extension_section() -> str:
    return (
        "## 7. Extension work\n\n"
        "- **SQLite persistence + crash recovery**: `scripts/persistence_demo.py` "
        "(state survives a new process/connection).\n"
        "- **Time travel**: `get_state_history()` traces every checkpoint.\n"
        "- **Graph diagram**: `graph.get_graph().draw_mermaid()` renders the full topology.\n"
        "- **Real HITL (optional)**: set `LANGGRAPH_INTERRUPT=true` to use "
        "`langgraph.types.interrupt()` in `approval_node`.\n"
        "- **LLM-as-judge (optional)**: set `LLM_JUDGE=true` for `evaluate_node`.\n"
    )


def _improvement_section() -> str:
    return (
        "## 8. Improvement plan\n\n"
        "With one more day: (1) replace the mock `tool_node` with real typed tools behind "
        "a registry; (2) add structured cost/latency tracking per LLM call to the events "
        "log; (3) parallel fan-out with `Send()` for multi-tool lookups; (4) richer "
        "LLM-as-judge with rubric scoring to drive smarter retry decisions; (5) a Streamlit "
        "approval UI backed by real `interrupt()` resumes.\n"
    )


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")

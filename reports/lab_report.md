# Day 08 Lab Report — LangGraph Agentic Orchestration

- Student: TieuLam (2A202600555)
- Provider: OpenAI (`gpt-4o-mini`)
- Generated: 2026-06-29

## 1. Metrics summary

| Metric | Value |
|---|---:|
| Total scenarios | 7 |
| Success rate | 100% |
| Avg nodes visited | 6.43 |
| Total retries | 3 |
| Total interrupts (HITL) | 2 |
| Resume success | False |

## 2. Per-scenario results

| Scenario | Expected | Actual | Success | Retries | Interrupts | Approval |
|---|---|---|:--:|--:|--:|:--:|
| S01_simple | simple | simple | ✅ | 0 | 0 | — |
| S02_tool | tool | tool | ✅ | 0 | 0 | — |
| S03_missing | missing_info | missing_info | ✅ | 0 | 0 | — |
| S04_risky | risky | risky | ✅ | 0 | 1 | ✅ |
| S05_error | error | error | ✅ | 2 | 0 | — |
| S06_delete | risky | risky | ✅ | 0 | 1 | ✅ |
| S07_dead_letter | error | error | ✅ | 1 | 0 | — |

## 3. Architecture

The graph is a `StateGraph(AgentState)` with 11 nodes wired as:

```
START -> intake -> classify -> [route_after_classify]
  simple       -> answer -> finalize -> END
  tool         -> tool -> evaluate -> [route_after_evaluate]
                            success     -> answer -> finalize -> END
                            needs_retry -> retry -> [route_after_retry]
                              attempt<max -> tool (loop)
                              else        -> dead_letter -> finalize -> END
  missing_info -> clarify -> finalize -> END
  risky        -> risky_action -> approval -> [route_after_approval]
                            approved -> tool -> evaluate -> ...
                            rejected -> clarify -> finalize -> END
  error        -> retry -> [route_after_retry] -> tool / dead_letter
```

**LLM integration**: `classify_node` uses `.with_structured_output(_Classification)` for reliable intent classification (priority risky > tool > missing_info > error > simple); `answer_node` generates a grounded response from `tool_results` + `approval` + the original query. `evaluate_node` supports an optional LLM-as-judge (`LLM_JUDGE=true`).

**Conditional edges** (4): `route_after_classify`, `route_after_evaluate`, `route_after_retry` (bounded), `route_after_approval`. Every path terminates at `finalize -> END`.

## 4. State schema

| Field | Reducer | Why |
|---|---|---|
| messages | append (`add`) | running trace of node visits |
| tool_results | append (`add`) | accumulate tool outputs across retries |
| errors | append (`add`) | audit transient failures |
| events | append (`add`) | append-only audit log for metrics |
| route | overwrite | current classification only |
| risk_level | overwrite | current risk assessment |
| attempt / max_attempts | overwrite | retry-loop bound |
| evaluation_result | overwrite | latest gate decision |
| pending_question | overwrite | current clarification |
| proposed_action | overwrite | action awaiting approval |
| approval | overwrite | latest HITL decision |
| final_answer | overwrite | terminal response |

Append-only fields use `Annotated[list, add]` so concurrent/sequential node writes accumulate; decision fields are overwritten because only the latest value is meaningful. State is kept lean and JSON-serializable for checkpointing.

## 5. Failure analysis

1. **Tool failure / unbounded retry**: the `error` route forces transient tool failures. `route_after_retry` enforces `attempt < max_attempts`, so retries are bounded; once exhausted the run escalates to `dead_letter` (see S07 with `max_attempts=1`). Without this bound the graph would loop forever.
2. **Risky action without approval**: `risky` queries cannot reach `tool` directly — they must pass `risky_action -> approval`. A rejected approval routes to `clarify` instead of executing the side effect, preventing unapproved refunds/deletions.

No scenario failed in this run — all routes matched and terminated.

## 6. Persistence / recovery evidence

Each run uses a per-scenario `thread_id` (`thread-<scenario_id>`). With the SQLite checkpointer (`build_checkpointer("sqlite")`), state is durably written to `outputs/checkpoints.db` (WAL mode). `scripts/persistence_demo.py` proves crash-recovery: a fresh connection + graph recovers the persisted state via `get_state()` and replays the full path via `get_state_history()` (8 checkpoints: intake -> classify -> tool -> evaluate -> answer -> finalize -> END).

## 7. Extension work

- **SQLite persistence + crash recovery**: `scripts/persistence_demo.py` (state survives a new process/connection).
- **Time travel**: `get_state_history()` traces every checkpoint.
- **Graph diagram**: `graph.get_graph().draw_mermaid()` renders the full topology.
- **Real HITL (optional)**: set `LANGGRAPH_INTERRUPT=true` to use `langgraph.types.interrupt()` in `approval_node`.
- **LLM-as-judge (optional)**: set `LLM_JUDGE=true` for `evaluate_node`.

## 8. Improvement plan

With one more day: (1) replace the mock `tool_node` with real typed tools behind a registry; (2) add structured cost/latency tracking per LLM call to the events log; (3) parallel fan-out with `Send()` for multi-tool lookups; (4) richer LLM-as-judge with rubric scoring to drive smarter retry decisions; (5) a Streamlit approval UI backed by real `interrupt()` resumes.

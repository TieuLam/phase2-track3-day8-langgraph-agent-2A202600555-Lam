"""Persistence / crash-recovery evidence for the Day 08 lab.

Demonstrates that the SQLite checkpointer survives a *fresh process* by:

1. Running one scenario through the graph with a SQLite checkpointer and a known
   ``thread_id``. The final checkpoint is written to ``outputs/checkpoints.sqlite``.
2. Building a BRAND-NEW checkpointer + graph (new sqlite connection — simulating a
   restarted process) and reading the persisted state back with ``get_state`` and
   ``get_state_history``. The recovered state proves the run was durably checkpointed.

Run:  python scripts/persistence_demo.py
Requires: OPENAI_API_KEY (or another provider) in .env, langgraph-checkpoint-sqlite.
"""

from __future__ import annotations

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state

THREAD_ID = "demo-persistence-thread"


def main() -> None:
    scenario = Scenario(
        id="persist_demo",
        query="Please lookup order status for order 98765",
        expected_route=Route.TOOL,
    )
    config = {"configurable": {"thread_id": THREAD_ID}}

    # -- Phase A: run and checkpoint to disk ------------------------------
    saver_a = build_checkpointer("sqlite")
    graph_a = build_graph(checkpointer=saver_a)
    state = initial_state(scenario)
    state["thread_id"] = THREAD_ID
    result = graph_a.invoke(state, config=config)
    print("-- Run 1 (writes checkpoint) -----------------------------")
    print("route       :", result.get("route"))
    print("final_answer:", (result.get("final_answer") or "")[:80])
    print("events       :", len(result.get("events", [])))

    # -- Phase B: fresh process simulation — new connection, same DB ------
    saver_b = build_checkpointer("sqlite")
    graph_b = build_graph(checkpointer=saver_b)
    recovered = graph_b.get_state(config)
    print("\n-- Recovery (new connection, same thread_id) -------------")
    print("thread_id recovered:", recovered.config["configurable"]["thread_id"])
    print("route persisted    :", recovered.values.get("route"))
    print("answer persisted   :", bool(recovered.values.get("final_answer")))

    history = list(graph_b.get_state_history(config))
    print("\n-- State history (time travel) ---------------------------")
    print("checkpoints stored :", len(history))
    # Oldest → newest. ``snap.next`` is the node(s) pending after this checkpoint,
    # so it traces the path the graph took: intake → classify → tool → ...
    for snap in reversed(history):
        step = snap.metadata.get("step")
        pending = ", ".join(snap.next) if snap.next else "END"
        route = snap.values.get("route", "") if snap.values else ""
        print(f"  step {step:>2}: next=[{pending}] route={route!r}")

    assert recovered.values.get("route") == "tool", "state did not persist!"
    assert len(history) > 1, "expected multiple checkpoints"
    print("\nOK: SQLite checkpoint survived a fresh graph/connection.")


if __name__ == "__main__":
    main()

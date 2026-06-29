"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# Default on-disk location for the SQLite checkpoint database.
# Name matches the patterns already in .gitignore (checkpoints.db*).
DEFAULT_SQLITE_PATH = "outputs/checkpoints.db"


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer for the requested backend.

    - "none"   → no checkpointer (graph runs without persistence)
    - "memory" → in-process MemorySaver (default; survives a single run only)
    - "sqlite" → on-disk SqliteSaver (survives process restarts → crash-resume)

    For SQLite we follow the langgraph-checkpoint-sqlite 3.x API: construct
    ``SqliteSaver(conn=sqlite3.connect(...))`` directly (NOT ``from_conn_string``,
    which is a context manager and would close the connection on exit). WAL mode is
    enabled for safer concurrent reads while a run is in progress.
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = _sqlite_path(database_url)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the connection can be reused across the
        # worker threads LangGraph may use during a run.
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return SqliteSaver(conn=conn)
    if kind == "postgres":
        raise NotImplementedError(
            "TODO(student): implement Postgres checkpointer (optional extension)"
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")


def _sqlite_path(database_url: str | None) -> str:
    """Resolve a filesystem path for the SQLite DB from an optional URL/path."""
    if not database_url:
        return DEFAULT_SQLITE_PATH
    # Accept both plain paths and sqlite URLs like "sqlite:///outputs/cp.sqlite".
    if database_url.startswith("sqlite:///"):
        return database_url[len("sqlite:///") :]
    if database_url.startswith("sqlite://"):
        return database_url[len("sqlite://") :]
    return database_url

"""SQLite memory of *verified* intermediate results, reused by later steps and later runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from typing import Any


def memory_key(tool: str, args: dict[str, Any]) -> str:
    canonical = json.dumps({"tool": tool, "args": args}, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class MemoryStore:
    def __init__(self, path: str = ":memory:") -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS verified (
                       key TEXT PRIMARY KEY, tool TEXT NOT NULL, args TEXT NOT NULL, value TEXT NOT NULL,
                       display TEXT NOT NULL, method TEXT NOT NULL, detail TEXT NOT NULL,
                       created REAL NOT NULL, hits INTEGER NOT NULL DEFAULT 0)"""
            )
            self._db.commit()

    def get(self, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
        key = memory_key(tool, args)
        with self._lock:
            row = self._db.execute(
                "SELECT value, display, method, detail FROM verified WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            self._db.execute("UPDATE verified SET hits = hits + 1 WHERE key = ?", (key,))
            self._db.commit()
        return {"value": json.loads(row[0]), "display": row[1], "method": row[2], "detail": row[3]}

    def put(self, tool: str, args: dict[str, Any], value: Any, display: str, method: str, detail: str) -> None:
        key = memory_key(tool, args)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO verified (key, tool, args, value, display, method, detail, created, hits) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT hits FROM verified WHERE key = ?), 0))",
                (key, tool, json.dumps(args, sort_keys=True, default=str), json.dumps(value, default=str),
                 display, method, detail, time.time(), key),
            )
            self._db.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM verified").fetchone()[0])

    def entries(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT tool, args, display, method, hits FROM verified ORDER BY created DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {"tool": r[0], "args": json.loads(r[1]), "display": r[2], "method": r[3], "hits": r[4]} for r in rows
        ]

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM verified")
            self._db.commit()

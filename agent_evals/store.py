"""SQLite-backed run history: trace persistence, listing, and regression diffs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent_evals.models import RunResult, StepRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    suite TEXT NOT NULL,
    runner TEXT NOT NULL,
    config TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS task_results (
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    task_name TEXT NOT NULL,
    category TEXT NOT NULL,
    score REAL NOT NULL,
    passed INTEGER NOT NULL,
    final_answer TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, task_id)
);
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_name TEXT NOT NULL DEFAULT '',
    tool_args TEXT NOT NULL DEFAULT '{}',
    tool_result TEXT NOT NULL DEFAULT '',
    latency_ms REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_steps_run_task ON steps (run_id, task_id, seq);
"""


class RunStore:
    """Persist runs, per-task results, and full step traces."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def save_run(self, run: RunResult, config: dict | None = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO runs (id, started_at, suite, runner, config) VALUES (?,?,?,?,?)",
            (run.run_id, run.started_at, run.suite, run.runner, json.dumps(config or {})),
        )
        for tr in run.task_results:
            cur.execute(
                """INSERT OR REPLACE INTO task_results
                   (run_id, task_id, task_name, category, score, passed, final_answer, error)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    run.run_id,
                    tr.task_id,
                    tr.task_name,
                    tr.category,
                    tr.score,
                    int(tr.passed),
                    tr.final_answer,
                    tr.error,
                ),
            )
            cur.execute("DELETE FROM steps WHERE run_id=? AND task_id=?", (run.run_id, tr.task_id))
            for st in tr.steps:
                cur.execute(
                    """INSERT INTO steps
                       (run_id, task_id, seq, role, content, tool_name, tool_args, tool_result, latency_ms)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        run.run_id,
                        tr.task_id,
                        st.seq,
                        st.role,
                        st.content,
                        st.tool_name,
                        json.dumps(st.tool_args),
                        st.tool_result,
                        st.latency_ms,
                    ),
                )
        self.conn.commit()

    def list_runs(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT r.id, r.started_at, r.suite, r.runner,
                      COUNT(t.task_id) AS total,
                      SUM(t.passed) AS passed,
                      AVG(t.score) AS avg_score
               FROM runs r LEFT JOIN task_results t ON t.run_id = r.id
               GROUP BY r.id ORDER BY r.started_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run_summary(self, run_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_task_results(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM task_results WHERE run_id=? ORDER BY task_id", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trace(self, run_id: str, task_id: str) -> list[StepRecord]:
        rows = self.conn.execute(
            "SELECT * FROM steps WHERE run_id=? AND task_id=? ORDER BY seq",
            (run_id, task_id),
        ).fetchall()
        return [
            StepRecord(
                seq=r["seq"],
                role=r["role"],
                content=r["content"],
                tool_name=r["tool_name"],
                tool_args=json.loads(r["tool_args"] or "{}"),
                tool_result=r["tool_result"],
                latency_ms=r["latency_ms"],
            )
            for r in rows
        ]

    def export_jsonl(self, run_id: str, out_path: str | Path) -> int:
        """Write one JSON object per step for every task in the run."""
        out_path = Path(out_path)
        count = 0
        with open(out_path, "w") as fh:
            for tr in self.get_task_results(run_id):
                for st in self.get_trace(run_id, tr["task_id"]):
                    fh.write(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "task_id": tr["task_id"],
                                "seq": st.seq,
                                "role": st.role,
                                "content": st.content,
                                "tool_name": st.tool_name,
                                "tool_args": st.tool_args,
                                "tool_result": st.tool_result,
                                "latency_ms": round(st.latency_ms, 2),
                            }
                        )
                        + "\n"
                    )
                    count += 1
        return count

    def compare_runs(self, run_a: str, run_b: str) -> dict:
        """Diff two runs: per-task score deltas plus improved/regressed lists."""
        a = {r["task_id"]: r for r in self.get_task_results(run_a)}
        b = {r["task_id"]: r for r in self.get_task_results(run_b)}
        task_ids = sorted(set(a) | set(b))
        per_task = []
        improved, regressed, unchanged, added, removed = [], [], [], [], []
        for tid in task_ids:
            ra, rb = a.get(tid), b.get(tid)
            if ra is None:
                added.append(tid)
                continue
            if rb is None:
                removed.append(tid)
                continue
            delta = round(rb["score"] - ra["score"], 4)
            entry = {
                "task_id": tid,
                "task_name": rb["task_name"],
                "score_a": ra["score"],
                "score_b": rb["score"],
                "delta": delta,
                "passed_a": bool(ra["passed"]),
                "passed_b": bool(rb["passed"]),
            }
            per_task.append(entry)
            if delta > 0:
                improved.append(tid)
            elif delta < 0:
                regressed.append(tid)
            else:
                unchanged.append(tid)
        avg_a = round(sum(r["score"] for r in a.values()) / len(a), 4) if a else 0.0
        avg_b = round(sum(r["score"] for r in b.values()) / len(b), 4) if b else 0.0
        return {
            "run_a": run_a,
            "run_b": run_b,
            "per_task": per_task,
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
            "added": added,
            "removed": removed,
            "avg_a": avg_a,
            "avg_b": avg_b,
            "avg_delta": round(avg_b - avg_a, 4),
        }

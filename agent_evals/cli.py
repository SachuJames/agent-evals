"""Command line interface: run suites, inspect history, diff runs, make reports."""

from __future__ import annotations

import json
from pathlib import Path

import click

from agent_evals.loop import run_suite
from agent_evals.runners import HttpRunner, MockRunner
from agent_evals.scorers import StubJudge
from agent_evals.store import RunStore
from agent_evals.tasks import builtin_tasks, load_suite

DEFAULT_DB = Path(".agent-evals") / "runs.db"


def _store(db: str | None) -> RunStore:
    return RunStore(db or str(DEFAULT_DB))


def _make_runner(name: str, mock_mode: str):
    if name == "mock":
        return MockRunner(mode=mock_mode)
    if name == "http":
        return HttpRunner()
    raise click.BadParameter(f"unknown runner: {name!r} (choose mock or http)")


@click.group()
def main() -> None:
    """agent-evals: score AI agents on YAML task suites."""


@main.command("run")
@click.option("--suite", default="builtin", help="Suite name or path to a task directory.")
@click.option("--runner", default="mock", help="Runner: mock or http.")
@click.option("--mock-mode", default="perfect", help="MockRunner mode: perfect or flaky.")
@click.option("--db", default=None, help="SQLite db path (default .agent-evals/runs.db).")
@click.option("--report", "report_path", default=None, help="Write an HTML report to this path.")
@click.option("--judge", default="stub", help="LLM judge: stub (offline) or none.")
@click.option("--task", "only", multiple=True, help="Run only these task ids.")
def run_cmd(suite, runner, mock_mode, db, report_path, judge, only):
    """Run a task suite and persist the run."""
    tasks = builtin_tasks() if suite == "builtin" else load_suite(suite)
    if only:
        wanted = set(only)
        tasks = [t for t in tasks if t.id in wanted]
        missing = wanted - {t.id for t in tasks}
        if missing:
            raise click.BadParameter(f"unknown task ids: {', '.join(sorted(missing))}")
    agent = _make_runner(runner, mock_mode)
    judge_obj = StubJudge() if judge == "stub" else None
    click.echo(f"Running {len(tasks)} tasks with runner '{agent.name}'...")
    run = run_suite(tasks, agent, suite_name=suite, judge=judge_obj)

    store = _store(db)
    store.save_run(run)
    click.echo(f"run id: {run.run_id}")
    for tr in run.task_results:
        mark = "PASS" if tr.passed else "FAIL"
        click.echo(f"  [{mark}] {tr.task_id:28s} score={tr.score:.2f} steps={len(tr.steps)}")
        if tr.error:
            click.echo(f"         error: {tr.error}")
    click.echo(f"Result: {run.passed}/{run.total} passed, avg score {run.avg_score:.2f}")

    if report_path:
        summary = store.get_run_summary(run.run_id) or {}
        task_rows = store.get_task_results(run.run_id)
        traces = {t["task_id"]: store.get_trace(run.run_id, t["task_id"]) for t in task_rows}
        from agent_evals.report import render_report

        Path(report_path).write_text(render_report(summary, task_rows, traces))
        click.echo(f"HTML report written to {report_path}")
    store.close()


@main.command("list-tasks")
@click.option("--suite", default="builtin")
def list_tasks(suite):
    """List tasks in a suite."""
    tasks = builtin_tasks() if suite == "builtin" else load_suite(suite)
    for t in tasks:
        click.echo(f"{t.id:28s} [{t.category:20s}] {t.name}")


@main.command("runs")
@click.option("--db", default=None)
def runs_cmd(db):
    """List recorded runs, newest first."""
    store = _store(db)
    rows = store.list_runs()
    if not rows:
        click.echo("no runs recorded yet")
    for r in rows:
        click.echo(
            f"{r['id']}  suite={r['suite']:10s} runner={r['runner']:12s} "
            f"{r['passed']}/{r['total']} passed  avg={r['avg_score']:.2f}"
        )
    store.close()


@main.command("show")
@click.argument("run_id")
@click.option("--db", default=None)
@click.option("--trace", "task_id", default=None, help="Print the full trace for a task.")
@click.option("--jsonl", "jsonl_path", default=None, help="Export traces to JSONL.")
def show_cmd(run_id, db, task_id, jsonl_path):
    """Show per-task scores for a run, optionally with a task trace."""
    store = _store(db)
    summary = store.get_run_summary(run_id)
    if summary is None:
        raise click.BadParameter(f"unknown run id: {run_id!r}")
    click.echo(
        f"run {summary['id']}  suite={summary['suite']} runner={summary['runner']} "
        f"started={summary['started_at']}"
    )
    for tr in store.get_task_results(run_id):
        mark = "PASS" if tr["passed"] else "FAIL"
        click.echo(f"  [{mark}] {tr['task_id']:28s} score={tr['score']:.2f}")
        if tr["error"]:
            click.echo(f"         error: {tr['error']}")
    if task_id:
        for st in store.get_trace(run_id, task_id):
            click.echo(f"--- step {st.seq} [{st.role}] ({st.latency_ms:.1f} ms)")
            if st.tool_name:
                click.echo(f"tool {st.tool_name} args={json.dumps(st.tool_args)}")
                click.echo(f"result: {st.tool_result}")
            else:
                click.echo(st.content)
    if jsonl_path:
        n = store.export_jsonl(run_id, jsonl_path)
        click.echo(f"exported {n} steps to {jsonl_path}")
    store.close()


@main.command("diff")
@click.argument("run_a")
@click.argument("run_b")
@click.option("--db", default=None)
def diff_cmd(run_a, run_b, db):
    """Compare two runs: what improved, regressed, or stayed the same."""
    store = _store(db)
    if store.get_run_summary(run_a) is None:
        raise click.BadParameter(f"unknown run id: {run_a!r}")
    if store.get_run_summary(run_b) is None:
        raise click.BadParameter(f"unknown run id: {run_b!r}")
    d = store.compare_runs(run_a, run_b)
    click.echo(f"avg score: {d['avg_a']:.2f} -> {d['avg_b']:.2f} (delta {d['avg_delta']:+.2f})")
    for pt in d["per_task"]:
        arrow = "^" if pt["delta"] > 0 else ("v" if pt["delta"] < 0 else "=")
        click.echo(f"  [{arrow}] {pt['task_id']:28s} {pt['score_a']:.2f} -> {pt['score_b']:.2f}")
    if d["regressed"]:
        click.echo(f"regressed: {', '.join(d['regressed'])}")
    if d["improved"]:
        click.echo(f"improved: {', '.join(d['improved'])}")
    store.close()


@main.command("report")
@click.argument("run_id")
@click.option("-o", "--output", required=True, help="Output HTML path.")
@click.option("--db", default=None)
@click.option("--vs", "vs_run", default=None, help="Compare against this run id in the report.")
def report_cmd(run_id, output, db, vs_run):
    """Generate a single-file HTML report for a run."""
    from agent_evals.report import render_report

    store = _store(db)
    summary = store.get_run_summary(run_id)
    if summary is None:
        raise click.BadParameter(f"unknown run id: {run_id!r}")
    task_rows = store.get_task_results(run_id)
    traces = {t["task_id"]: store.get_trace(run_id, t["task_id"]) for t in task_rows}
    diff = store.compare_runs(vs_run, run_id) if vs_run else None
    Path(output).write_text(render_report(summary, task_rows, traces, diff))
    click.echo(f"report written to {output}")
    store.close()


if __name__ == "__main__":
    main()

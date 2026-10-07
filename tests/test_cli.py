"""End-to-end CLI tests using click's CliRunner."""

import sqlite3

from click.testing import CliRunner

from agent_evals.cli import main


def _db_tasks(runner, db):
    result = runner.invoke(main, ["run", "--suite", "builtin", "--db", str(db)])
    assert result.exit_code == 0, result.output
    return result.output


def test_cli_run_end_to_end(tmp_path):
    runner = CliRunner()
    db = tmp_path / "runs.db"
    out = _db_tasks(runner, db)
    assert "10/10 passed" in out
    assert "run id:" in out
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM task_results").fetchone()[0] == 10
    assert con.execute("SELECT COUNT(*) FROM steps").fetchone()[0] > 10
    con.close()


def test_cli_list_tasks():
    runner = CliRunner()
    result = runner.invoke(main, ["list-tasks"])
    assert result.exit_code == 0
    assert "math_pythagoras" in result.output


def test_cli_runs_and_show(tmp_path):
    runner = CliRunner()
    db = tmp_path / "runs.db"
    out = _db_tasks(runner, db)
    run_id = [line for line in out.splitlines() if line.startswith("run id:")][0].split()[-1]

    result = runner.invoke(main, ["runs", "--db", str(db)])
    assert result.exit_code == 0
    assert run_id in result.output

    result = runner.invoke(main, ["show", run_id, "--db", str(db)])
    assert result.exit_code == 0
    assert "math_compound_interest" in result.output

    result = runner.invoke(main, ["show", run_id, "--db", str(db), "--trace", "chained_calc"])
    assert result.exit_code == 0
    assert "calculator" in result.output


def test_cli_show_unknown_run(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["show", "nope", "--db", str(tmp_path / "runs.db")])
    assert result.exit_code != 0


def test_cli_diff(tmp_path):
    runner = CliRunner()
    db = str(tmp_path / "runs.db")
    out1 = runner.invoke(main, ["run", "--suite", "builtin", "--db", db]).output
    out2 = runner.invoke(
        main, ["run", "--suite", "builtin", "--db", db, "--mock-mode", "flaky"]
    ).output
    run_a = [line for line in out1.splitlines() if line.startswith("run id:")][0].split()[-1]
    run_b = [line for line in out2.splitlines() if line.startswith("run id:")][0].split()[-1]

    result = runner.invoke(main, ["diff", run_a, run_b, "--db", db])
    assert result.exit_code == 0, result.output
    assert "regressed: format_three_bullets, math_pythagoras" in result.output
    assert "delta -0.20" in result.output


def test_cli_report(tmp_path):
    runner = CliRunner()
    db = str(tmp_path / "runs.db")
    out = runner.invoke(main, ["run", "--suite", "builtin", "--db", db]).output
    run_id = [line for line in out.splitlines() if line.startswith("run id:")][0].split()[-1]
    report = tmp_path / "report.html"
    result = runner.invoke(main, ["report", run_id, "-o", str(report), "--db", db])
    assert result.exit_code == 0, result.output
    html = report.read_text()
    assert "agent-evals run report" in html
    assert run_id in html
    assert "math_pythagoras" in html


def test_cli_run_only_subset(tmp_path):
    runner = CliRunner()
    db = str(tmp_path / "runs.db")
    result = runner.invoke(
        main, ["run", "--suite", "builtin", "--db", db, "--task", "math_pythagoras"]
    )
    assert result.exit_code == 0, result.output
    assert "1/1 passed" in result.output

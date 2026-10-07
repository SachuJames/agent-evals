"""Single-file HTML report: summary, per-task scores, diffs, expandable traces.

Everything is inlined (no external assets) so the report works offline.
"""

from __future__ import annotations

import html
from datetime import datetime

_CSS = """
body{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
margin:0;background:#0d1117;color:#e6edf3}
.wrap{max-width:1100px;margin:0 auto;padding:32px 24px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:32px 0 12px}
.sub{color:#8b949e;margin-bottom:24px}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 20px;min-width:150px}
.card .v{font-size:26px;font-weight:700}.card .l{color:#8b949e;font-size:12px;margin-top:4px}
table{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid #21262d;font-size:14px}
th{color:#8b949e;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:none}
.pass{color:#3fb950;font-weight:600}.fail{color:#f85149;font-weight:600}
.bar{height:8px;background:#21262d;border-radius:4px;min-width:120px}
.bar>i{display:block;height:8px;border-radius:4px;background:#1f6feb}
details{background:#161b22;border:1px solid #30363d;border-radius:10px;margin:10px 0}
summary{padding:12px 16px;cursor:pointer;font-size:14px}
.step{padding:10px 16px;border-top:1px solid #21262d;font-size:13px}
.step pre{background:#0d1117;padding:10px;border-radius:6px;overflow-x:auto;white-space:pre-wrap}
.tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:12px;background:#21262d;color:#8b949e;margin-right:6px}
.delta-up{color:#3fb950}.delta-dn{color:#f85149}
"""

_SUMMARY = """
<div class="wrap">
<h1>agent-evals run report</h1>
<div class="sub">run <b>{run_id}</b> &middot; suite {suite} &middot; runner {runner} &middot; {when}</div>
<div class="cards">
<div class="card"><div class="v">{passed}/{total}</div><div class="l">tasks passed</div></div>
<div class="card"><div class="v">{avg:.0%}</div><div class="l">average score</div></div>
<div class="card"><div class="v">{steps}</div><div class="l">trace steps</div></div>
</div>
"""


def _esc(text: str) -> str:
    return html.escape(text or "")


def _task_table(task_results: list[dict]) -> str:
    rows = []
    for tr in task_results:
        cls = "pass" if tr["passed"] else "fail"
        mark = "PASS" if tr["passed"] else "FAIL"
        width = int(tr["score"] * 100)
        rows.append(
            f"<tr><td><b>{_esc(tr['task_id'])}</b><br>"
            f"<span style='color:#8b949e'>{_esc(tr['task_name'])}</span></td>"
            f"<td><span class='tag'>{_esc(tr['category'])}</span></td>"
            f"<td><div class='bar'><i style='width:{width}%'></i></div></td>"
            f"<td>{tr['score']:.2f}</td>"
            f"<td class='{cls}'>{mark}</td></tr>"
        )
    return (
        "<h2>Tasks</h2><table><tr><th>Task</th><th>Category</th><th>Score</th>"
        "<th>Value</th><th>Result</th></tr>" + "".join(rows) + "</table>"
    )


def _diff_table(diff: dict) -> str:
    if not diff["per_task"]:
        return ""
    rows = []
    for pt in diff["per_task"]:
        d = pt["delta"]
        cls = "delta-up" if d > 0 else ("delta-dn" if d < 0 else "")
        arrow = "&#9650;" if d > 0 else ("&#9660;" if d < 0 else "=")
        rows.append(
            f"<tr><td><b>{_esc(pt['task_id'])}</b></td>"
            f"<td>{pt['score_a']:.2f}</td><td>{pt['score_b']:.2f}</td>"
            f"<td class='{cls}'>{arrow} {d:+.2f}</td></tr>"
        )
    return (
        f"<h2>Regression diff: {diff['run_a'][:8]} vs {diff['run_b'][:8]}</h2>"
        f"<div class='sub'>average {diff['avg_a']:.2f} &rarr; {diff['avg_b']:.2f} "
        f"(<b>{diff['avg_delta']:+.2f}</b>) &middot; improved: {len(diff['improved'])} "
        f"&middot; regressed: {len(diff['regressed'])}</div>"
        "<table><tr><th>Task</th><th>Before</th><th>After</th><th>Delta</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _traces(task_results: list[dict], traces: dict[str, list]) -> str:
    parts = ["<h2>Traces</h2>"]
    for tr in task_results:
        steps = traces.get(tr["task_id"], [])
        inner = []
        for st in steps:
            body = _esc(st.content)
            if st.tool_name:
                body = (
                    f"<b>tool: {_esc(st.tool_name)}</b> "
                    f"<pre>{_esc(str(st.tool_args))}</pre>"
                    f"<b>result:</b><pre>{_esc(st.tool_result)}</pre>"
                )
            inner.append(
                f"<div class='step'><span class='tag'>#{st.seq} {st.role}</span>"
                f"<span class='tag'>{st.latency_ms:.1f} ms</span><div>{body}</div></div>"
            )
        parts.append(
            f"<details><summary><b>{_esc(tr['task_id'])}</b> "
            f"&middot; score {tr['score']:.2f} &middot; {len(steps)} steps</summary>"
            + "".join(inner)
            + "</details>"
        )
    return "".join(parts)


def render_report(
    run_summary: dict,
    task_results: list[dict],
    traces: dict[str, list] | None = None,
    diff: dict | None = None,
) -> str:
    """Render the full HTML report for one run, optionally with a diff."""
    when = datetime.fromtimestamp(run_summary["started_at"]).strftime("%Y-%m-%d %H:%M:%S")
    total_steps = sum(len(v) for v in (traces or {}).values())
    avg = (
        sum(t["score"] for t in task_results) / len(task_results) if task_results else 0.0
    )
    passed = sum(1 for t in task_results if t["passed"])
    body = _SUMMARY.format(
        run_id=_esc(run_summary["id"]),
        suite=_esc(run_summary["suite"]),
        runner=_esc(run_summary["runner"]),
        when=when,
        passed=passed,
        total=len(task_results),
        avg=avg,
        steps=total_steps,
    )
    body += _task_table(task_results)
    if diff:
        body += _diff_table(diff)
    if traces is not None:
        body += _traces(task_results, traces)
    body += "</div>"
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>agent-evals report {html.escape(run_summary['id'][:8])}</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )

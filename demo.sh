#!/usr/bin/env bash
# End-to-end demo: run the builtin suite twice (perfect vs flaky mock),
# diff the runs, and generate an HTML report. Exits nonzero on any failure.
set -euo pipefail
cd "$(dirname "$0")"

if [ -x "$HOME/workspace/agent-evals-venv/bin/python" ]; then
  PY="$HOME/workspace/agent-evals-venv/bin/python"
else
  PY="python3"
fi
export PYTHONPATH="$PWD"

have_deps() { "$PY" -c "import click, yaml, requests, rich, jsonschema" 2>/dev/null; }
if ! have_deps; then
  echo "installing dependencies..."
  "$PY" -m pip install -q -r requirements.txt
fi

DB=".agent-evals/demo-runs.db"
REPORT="demo-report.html"
rm -rf .agent-evals "$REPORT" demo-trace.jsonl

fail() { echo "DEMO FAILED: $1" >&2; exit 1; }

echo "=== 1. builtin suite, mock runner (perfect) ==="
OUT1=$("$PY" -m agent_evals.cli run --suite builtin --db "$DB")
echo "$OUT1"
echo "$OUT1" | grep -q "10/10 passed" || fail "expected 10/10 passed in run 1"
RUN1=$(echo "$OUT1" | grep "^run id:" | awk '{print $3}')
[ -n "$RUN1" ] || fail "could not parse run id 1"

echo "=== 2. builtin suite, mock runner (flaky) ==="
OUT2=$("$PY" -m agent_evals.cli run --suite builtin --db "$DB" --mock-mode flaky)
echo "$OUT2" | tail -3
echo "$OUT2" | grep -q "8/10 passed" || fail "expected 8/10 passed in run 2"
RUN2=$(echo "$OUT2" | grep "^run id:" | awk '{print $3}')
[ -n "$RUN2" ] || fail "could not parse run id 2"

echo "=== 3. regression diff ==="
DIFF=$("$PY" -m agent_evals.cli diff "$RUN1" "$RUN2" --db "$DB")
echo "$DIFF"
echo "$DIFF" | grep -q "regressed: format_three_bullets, math_pythagoras" \
  || fail "expected the two flaky regressions in diff"
echo "$DIFF" | grep -q "delta -0.20" || fail "expected avg delta -0.20"

echo "=== 4. html report with diff ==="
"$PY" -m agent_evals.cli report "$RUN2" -o "$REPORT" --db "$DB" --vs "$RUN1"
[ -f "$REPORT" ] || fail "report file not created"
grep -q "Regression diff" "$REPORT" || fail "report missing diff section"
grep -q "format_three_bullets" "$REPORT" || fail "report missing task rows"

echo "=== 5. jsonl trace export ==="
"$PY" -m agent_evals.cli show "$RUN1" --db "$DB" --jsonl demo-trace.jsonl >/dev/null
LINES=$(wc -l < demo-trace.jsonl)
[ "$LINES" -gt 10 ] || fail "expected more than 10 trace lines, got $LINES"
echo "exported $LINES trace steps to demo-trace.jsonl"

echo ""
echo "DEMO RESULT: ALL CHECKS PASSED"

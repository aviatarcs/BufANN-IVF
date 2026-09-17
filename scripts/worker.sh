#!/usr/bin/env bash
# Runs the worker prompt in a loop: one unit of work per invocation, fresh
# context each time. Logs to logs/worker.log. Stop with Ctrl-C or by creating
# logs/STOP.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
CLAUDE="$(scripts/claude_bin.sh)"
mkdir -p logs
while [[ ! -f logs/STOP ]]; do
    echo "=== worker cycle $(date -Is) ===" | tee -a logs/worker.log
    "$CLAUDE" -p "$(cat .claude/prompts/worker.md)" \
        --settings .claude/worker.settings.json \
        --max-turns "${WORKER_MAX_TURNS:-250}" < /dev/null 2>&1 | tee -a logs/worker.log
    status=${PIPESTATUS[0]}
    # A usage-limit or API error exits non-zero; wait for the window to move.
    pause="${WORKER_SLEEP:-60}"
    (( status == 0 )) || pause="${WORKER_BACKOFF:-900}"
    echo "=== cycle done $(date -Is) exit=$status, sleeping ${pause}s ===" | tee -a logs/worker.log
    sleep "$pause"
done

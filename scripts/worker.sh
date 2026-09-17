#!/usr/bin/env bash
# Runs the worker prompt in a loop: one unit of work per invocation, fresh
# context each time. Logs to logs/worker.log. Stop with Ctrl-C or by creating
# logs/STOP.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs
while [[ ! -f logs/STOP ]]; do
    echo "=== worker cycle $(date -Is) ===" | tee -a logs/worker.log
    claude -p "$(cat .claude/prompts/worker.md)" \
        --settings .claude/worker.settings.json \
        --max-turns "${WORKER_MAX_TURNS:-250}" 2>&1 | tee -a logs/worker.log
    echo "=== cycle done $(date -Is), sleeping ${WORKER_SLEEP:-60}s ===" | tee -a logs/worker.log
    sleep "${WORKER_SLEEP:-60}"
done

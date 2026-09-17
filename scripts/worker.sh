#!/usr/bin/env bash
# The autonomous worker: one PLAN.md item or REVIEW.md fix per cycle, each
# cycle a background Claude session in its own worktree on branch agent/work.
# Logs to logs/worker.log; stop with `touch logs/STOP`.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/agent_lib.sh"
cd "$REPO" && mkdir -p logs
WT="$(ensure_worktree worker agent/work)"
PROMPT="$(cat "$REPO/.claude/prompts/worker.md")"
while [[ ! -f logs/STOP ]]; do
    echo "=== worker cycle $(date -Is) in $WT ===" | tee -a logs/worker.log
    git -C "$WT" fetch -q origin && git -C "$WT" pull -q --ff-only origin agent/work 2>&1 | tee -a logs/worker.log
    run_cycle "$WT" "$WT/.claude/worker.settings.json" "$PROMPT" "$REPO/logs/worker.log" "${WORKER_TIMEOUT:-7200}"
    status=$?
    pause="${WORKER_SLEEP:-60}"
    (( status == 0 )) || pause="${WORKER_BACKOFF:-900}"
    echo "=== cycle done $(date -Is) status=$status, sleeping ${pause}s ===" | tee -a logs/worker.log
    sleep "$pause"
done

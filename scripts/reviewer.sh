#!/usr/bin/env bash
# The adversarial reviewer: reviews each new commit range on origin/agent/work
# in its own worktree (detached at the range end, so it builds exactly what
# was pushed), appends the report to REVIEW.md and pushes that to agent/work.
# Logs to logs/reviewer.log; stop with `touch logs/STOP`.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/agent_lib.sh"
cd "$REPO" && mkdir -p logs
BRANCH="${REVIEW_BRANCH:-agent/work}"
STATE="logs/reviewer.last"
WT="$(ensure_worktree reviewer "review/scratch")"
git -C "$REPO" fetch -q origin
[[ -f "$STATE" ]] || git rev-parse "origin/$BRANCH" > "$STATE"
while [[ ! -f logs/STOP ]]; do
    git -C "$REPO" fetch -q origin
    last="$(cat "$STATE")"; head="$(git rev-parse "origin/$BRANCH")"
    # Skip ranges that consist only of the reviewer's own commits.
    if [[ "$head" != "$last" ]] && git log --format=%s "$last..$head" | grep -qv '^Review '; then
        echo "=== reviewing $last..$head $(date -Is) in $WT ===" | tee -a logs/reviewer.log
        git -C "$WT" fetch -q origin && git -C "$WT" checkout -q --detach "$head"
        rm -f "$WT/REVIEW.out"
        prompt="Review the commit range $last..$head. Write your report to the file REVIEW.out in the current directory (that is the only file you may write) and reply 'done'. $(cat "$REPO/.claude/prompts/reviewer.md")"
        run_cycle "$WT" "$WT/.claude/reviewer.settings.json" "$prompt" "$REPO/logs/reviewer.log" "${REVIEWER_TIMEOUT:-3600}"
        status=$?
        if (( status == 0 )) && [[ -s "$WT/REVIEW.out" ]]; then
            # Land the report on the tip of the branch, retrying once if the worker pushed meanwhile.
            for attempt in 1 2; do
                git -C "$WT" fetch -q origin && git -C "$WT" checkout -q --detach "origin/$BRANCH"
                printf '\n%s\n' "$(cat "$WT/REVIEW.out")" >> "$WT/REVIEW.md"
                git -C "$WT" add REVIEW.md && git -C "$WT" commit -q -m "Review ${last:0:7}..${head:0:7}" && \
                    git -C "$WT" push -q origin "HEAD:$BRANCH" && break
                git -C "$WT" reset -q --hard "origin/$BRANCH"
            done
            rm -f "$WT/REVIEW.out"
            cat "$WT/REVIEW.md" | tail -n +"$(grep -n "^## Review of ${last:0:7}" "$WT/REVIEW.md" | tail -1 | cut -d: -f1)" | tee -a logs/reviewer.log
            git rev-parse "origin/$BRANCH" > "$STATE"
        else
            echo "review did not produce REVIEW.out (status=$status); retrying in ${REVIEWER_BACKOFF:-900}s" | tee -a logs/reviewer.log
            sleep "${REVIEWER_BACKOFF:-900}"
            continue
        fi
    fi
    sleep "${REVIEWER_POLL:-300}"
done

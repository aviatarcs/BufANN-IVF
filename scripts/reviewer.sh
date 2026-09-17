#!/usr/bin/env bash
# Reviews every new commit on origin/agent/work as it appears. Appends the
# review to REVIEW.md and commits that on agent/work. Never edits source.
# Logs to logs/reviewer.log. Stop with Ctrl-C or by creating logs/STOP.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
CLAUDE="$(scripts/claude_bin.sh)"
mkdir -p logs
BRANCH="${REVIEW_BRANCH:-agent/work}"
STATE="logs/reviewer.last"
git fetch -q origin
[[ -f "$STATE" ]] || git rev-parse "origin/$BRANCH" > "$STATE"
while [[ ! -f logs/STOP ]]; do
    git fetch -q origin
    last="$(cat "$STATE")"
    head="$(git rev-parse "origin/$BRANCH")"
    # Skip ranges that consist only of the reviewer's own REVIEW.md commits.
    if [[ "$head" != "$last" ]] && git log --format=%s "$last..$head" | grep -qv '^Review '; then
        echo "=== reviewing $last..$head $(date -Is) ===" | tee -a logs/reviewer.log
        git checkout -q "$BRANCH" && git pull -q --ff-only origin "$BRANCH"
        review="$("$CLAUDE" -p "Review the commit range $last..$head. $(cat .claude/prompts/reviewer.md)" \
            --settings .claude/reviewer.settings.json \
            --max-turns "${REVIEWER_MAX_TURNS:-120}" < /dev/null 2>>logs/reviewer.log)" || {
            echo "claude exited non-zero (usage limit?); retrying in ${REVIEWER_BACKOFF:-900}s" | tee -a logs/reviewer.log
            sleep "${REVIEWER_BACKOFF:-900}"
            continue
        }
        printf '\n%s\n' "$review" >> REVIEW.md
        git add REVIEW.md && git commit -q -m "Review ${last:0:7}..${head:0:7}" && git push -q origin "$BRANCH"
        git rev-parse "origin/$BRANCH" > "$STATE"
        echo "$review" | tee -a logs/reviewer.log
    fi
    sleep "${REVIEWER_POLL:-300}"
done

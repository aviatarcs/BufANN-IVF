#!/usr/bin/env bash
# Stop hook for the autonomous worker: refuse to end a cycle with uncommitted
# changes to tracked files or with the last ctest run red. Reads the hook
# JSON on stdin; stop_hook_active means we already blocked once this turn,
# so let it through rather than loop forever.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input="$(cat)"
if echo "$input" | grep -q '"stop_hook_active": *true'; then
    exit 0
fi
reasons=()
if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]; then
    reasons+=("tracked files have uncommitted changes: $(git -C "$REPO" status --porcelain --untracked-files=no | head -5 | tr '\n' ' ')")
fi
failed="$REPO/build/Testing/Temporary/LastTestsFailed.log"
if [[ -s "$failed" ]]; then
    reasons+=("the last ctest run failed: $(tr '\n' ' ' < "$failed")")
fi
if (( ${#reasons[@]} )); then
    reason="$(printf '%s; ' "${reasons[@]}")"
    printf '{"decision":"block","reason":"Do not stop yet: %s Fix, run ctest, and commit before finishing."}\n' \
        "$(echo "$reason" | sed 's/"/\\"/g')"
fi
exit 0

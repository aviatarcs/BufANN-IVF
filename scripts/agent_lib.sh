# Shared by scripts/worker.sh and scripts/reviewer.sh. Source, do not run.
#
# Each agent works in its own git worktree on local disk so that neither the
# main checkout nor the other agent ever sees its half-finished edits. A cycle
# is one `claude --bg` session (visible in `claude agents` and the web agent
# view), started from an environment scrubbed of any enclosing Claude session
# so it authenticates on its own, and waited on until its state is done.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${BUFANN_BUILD_ROOT:-/var/tmp/bufann-ivf-$USER}"
CLAUDE="$("$REPO/scripts/claude_bin.sh")"

# Never let a cycle inherit an enclosing Claude session's identity.
unset CLAUDE_CODE_CHILD_SESSION CLAUDE_CODE_MESSAGING_SOCKET CLAUDE_CODE_MESSAGING_TOKEN \
      CLAUDE_CODE_SESSION_ID CLAUDE_CODE_SESSION_ATTENDED CLAUDE_CODE_ENTRYPOINT \
      CLAUDE_CODE_EXECPATH CLAUDE_PID CLAUDECODE CLAUDE_AGENT_SDK_VERSION AI_AGENT

# ensure_worktree <name> <branch>: creates $ROOT/wt-<name> on <branch> (from
# origin) if missing; prints its path.
ensure_worktree() {
    local name="$1" branch="$2" dir="$ROOT/wt-$1"
    if [[ ! -d "$dir/.git" && ! -f "$dir/.git" ]]; then
        git -C "$REPO" fetch -q origin
        git -C "$REPO" worktree prune
        if git -C "$REPO" show-ref --verify --quiet "refs/remotes/origin/$branch"; then
            git -C "$REPO" worktree add -q "$dir" -B "$branch" "origin/$branch" >&2
        else
            git -C "$REPO" worktree add -q "$dir" -b "$branch" origin/main >&2
        fi
    fi
    echo "$dir"
}

# bg_session_state <id>: prints the session's state (running/done/blocked/...)
# or "gone" if it is no longer listed.
bg_session_state() {
    "$CLAUDE" agents --json 2>/dev/null | python3 -c '
import sys, json
want = sys.argv[1]
for s in json.load(sys.stdin):
    if s.get("id") == want:
        print(s.get("state") or s.get("status") or "unknown"); sys.exit(0)
print("gone")' "$1"
}

# run_cycle <cwd> <settings-file> <prompt> <log> [timeout-seconds]: starts a
# background session in <cwd> and waits until it is done. Returns 0 on done,
# 1 on blocked (needs input), 2 on timeout (session stopped), 3 on launch
# failure. Sets CYCLE_ID.
run_cycle() {
    local cwd="$1" settings="$2" prompt="$3" log="$4" timeout="${5:-7200}"
    local out
    out="$(cd "$cwd" && "$CLAUDE" --bg "$prompt" --settings "$settings" 2>&1)" || {
        echo "$out" | tee -a "$log"; return 3; }
    CYCLE_ID="$(echo "$out" | grep -oE 'backgrounded · [0-9a-f]+' | awk '{print $3}')"
    [[ -n "$CYCLE_ID" ]] || { echo "could not parse session id from: $out" | tee -a "$log"; return 3; }
    echo "session $CYCLE_ID started in $cwd (claude logs $CYCLE_ID / claude attach $CYCLE_ID)" | tee -a "$log"
    local waited=0 state
    while (( waited < timeout )); do
        sleep 20; waited=$((waited + 20))
        state="$(bg_session_state "$CYCLE_ID")"
        case "$state" in
            done|gone) echo "session $CYCLE_ID finished ($state) after ${waited}s" | tee -a "$log"; return 0 ;;
            blocked)   echo "session $CYCLE_ID is blocked waiting for input; see: claude attach $CYCLE_ID" | tee -a "$log"; return 1 ;;
        esac
    done
    echo "session $CYCLE_ID exceeded ${timeout}s; stopping it" | tee -a "$log"
    "$CLAUDE" stop "$CYCLE_ID" >/dev/null 2>&1 || true
    return 2
}

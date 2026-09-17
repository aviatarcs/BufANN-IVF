#!/usr/bin/env bash
# Prints the path of a Claude Code CLI: $CLAUDE_BIN, else `claude` on PATH,
# else the newest binary bundled with the VS Code extension. Source or call.
set -euo pipefail
if [[ -n "${CLAUDE_BIN:-}" && -x "$CLAUDE_BIN" ]]; then
    echo "$CLAUDE_BIN"
elif command -v claude >/dev/null 2>&1; then
    command -v claude
else
    newest="$(ls -d "$HOME"/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude 2>/dev/null | sort -V | tail -1)"
    [[ -x "$newest" ]] || { echo "ERROR: no claude CLI found; install it or set CLAUDE_BIN" >&2; exit 1; }
    echo "$newest"
fi

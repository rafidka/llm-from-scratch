#!/bin/bash
set -euo pipefail

SESSION_NAME="llm-from-scratch"
REMOTE_WORKSPACE="/root/llm-from-scratch"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYNCIGNORE="$SCRIPT_DIR/.syncignore"

usage() {
    echo "Usage: $0 <ssh-host> <ssh-port>"
    echo ""
    echo "Examples:"
    echo "  $0 root@ssh-prod.vast.ai 42113"
    echo ""
    echo "Commands:"
    echo "  $0 <host> <port>          Start or restart sync"
    echo "  $0 status                 Show sync session status"
    echo "  $0 stop                   Stop sync session"
    exit 1
}

check_mutagen() {
    if ! command -v mutagen &>/dev/null; then
        echo "mutagen is not installed."
        echo ""
        echo "Install with Homebrew:"
        echo "  brew install mutagen-io/mutagen/mutagen"
        echo ""
        echo "Or download from: https://github.com/mutagen-io/mutagen/releases"
        exit 1
    fi
}

stop_sync() {
    if mutagen sync list "$SESSION_NAME" &>/dev/null; then
        echo "Stopping sync session '$SESSION_NAME'..."
        mutagen sync terminate "$SESSION_NAME"
        echo "Stopped."
    else
        echo "No active session named '$SESSION_NAME'."
    fi
}

if [ $# -eq 1 ] && [ "$1" = "status" ]; then
    check_mutagen
    mutagen sync list "$SESSION_NAME"
    exit 0
fi

if [ $# -eq 1 ] && [ "$1" = "stop" ]; then
    check_mutagen
    stop_sync
    exit 0
fi

if [ $# -ne 2 ]; then
    usage
fi

SSH_HOST="$1"
SSH_PORT="$2"

check_mutagen

# Terminate existing session if it exists
if mutagen sync list "$SESSION_NAME" &>/dev/null; then
    echo "Terminating existing session '$SESSION_NAME'..."
    mutagen sync terminate "$SESSION_NAME"
fi

IGNORE_ARGS=()
if [ -f "$SYNCIGNORE" ]; then
    while IFS= read -r pattern; do
        # Skip comments and empty lines
        [[ "$pattern" =~ ^#.*$ || -z "$pattern" ]] && continue
        IGNORE_ARGS+=(--ignore "$pattern")
    done < "$SYNCIGNORE"
else
    echo "Warning: .syncignore not found at $SYNCIGNORE"
fi

echo "Creating sync session '$SESSION_NAME'..."
echo "  Local:    $SCRIPT_DIR/"
echo "  Remote:   $SSH_HOST:$SSH_PORT:$REMOTE_WORKSPACE/"
echo ""

mutagen sync create \
    --name "$SESSION_NAME" \
    "${IGNORE_ARGS[@]}" \
    --sync-mode "two-way-resolved" \
    "$SCRIPT_DIR/" \
    "$SSH_HOST:$SSH_PORT:$REMOTE_WORKSPACE/"

echo ""
echo "Sync session created. Monitoring..."
echo "Press Ctrl+C to stop monitoring (sync continues in background)."
echo ""
echo "Useful commands:"
echo "  $0 status   - Check sync status"
echo "  $0 stop     - Stop sync session"
echo ""

mutagen sync monitor "$SESSION_NAME"
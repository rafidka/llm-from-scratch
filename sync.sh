#!/bin/bash
set -euo pipefail

CODE_SESSION="llm-code"
CKPT_SESSION="llm-checkpoints"
REMOTE_WORKSPACE="/root/llm-from-scratch"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYNCIGNORE="$SCRIPT_DIR/.syncignore"

usage() {
    echo "Usage: $0 <ssh-host> <ssh-port>"
    echo ""
    echo "Starts two sync sessions:"
    echo "  $CODE_SESSION  — one-way local→remote (code)"
    echo "  $CKPT_SESSION  — one-way remote→local (checkpoints)"
    echo ""
    echo "Commands:"
    echo "  $0 <host> <port>    Start or restart sync"
    echo "  $0 status           Show sync session status"
    echo "  $0 stop             Stop all sync sessions"
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
    for session in "$CODE_SESSION" "$CKPT_SESSION"; do
        if mutagen sync list "$session" &>/dev/null; then
            echo "Stopping session '$session'..."
            mutagen sync terminate "$session"
        fi
    done
    echo "All sessions stopped."
}

if [ $# -eq 1 ] && [ "$1" = "status" ]; then
    check_mutagen
    mutagen sync list
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

stop_sync 2>/dev/null || true

BUILD_IGNORE_ARGS=()
if [ -f "$SYNCIGNORE" ]; then
    while IFS= read -r pattern; do
        [[ "$pattern" =~ ^#.*$ || -z "$pattern" ]] && continue
        BUILD_IGNORE_ARGS+=(--ignore "$pattern")
    done < "$SYNCIGNORE"
else
    echo "Warning: .syncignore not found at $SYNCIGNORE"
fi

REMOTE_URL="$SSH_HOST:$SSH_PORT:$REMOTE_WORKSPACE"

echo "Creating code sync session '$CODE_SESSION' (local → remote, one-way)..."
mutagen sync create \
    --name "$CODE_SESSION" \
    "${BUILD_IGNORE_ARGS[@]}" \
    --sync-mode "one-way-replica" \
    "$SCRIPT_DIR/" \
    "$REMOTE_URL/"

echo ""
echo "Creating checkpoints sync session '$CKPT_SESSION' (remote → local, safe)..."
mkdir -p "$SCRIPT_DIR/checkpoints"
mutagen sync create \
    --name "$CKPT_SESSION" \
    --sync-mode "one-way-safe" \
    "$REMOTE_URL/checkpoints/" \
    "$SCRIPT_DIR/checkpoints/"

echo ""
echo "Both sessions created. Monitoring..."
echo "Press Ctrl+C to stop monitoring (sync continues in background)."
echo ""
echo "Useful commands:"
echo "  $0 status   - Check sync status"
echo "  $0 stop     - Stop all sync sessions"
echo ""

mutagen sync monitor "$CODE_SESSION" "$CKPT_SESSION"
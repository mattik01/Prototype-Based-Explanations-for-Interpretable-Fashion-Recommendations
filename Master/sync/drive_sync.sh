#!/usr/bin/env bash
# Sync this thesis repo to/from Google Drive via rclone, excluding large datasets.
#
# Usage:
#   drive_sync.sh up          # DRY-RUN preview of  local -> Drive
#   drive_sync.sh up   --go   # actually push       local -> Drive
#   drive_sync.sh down        # DRY-RUN preview of  Drive -> local
#   drive_sync.sh down --go   # actually pull       Drive -> local
#
# What syncs is controlled by Master/sync/drive_filter.txt (data/** is excluded,
# literature PDFs + *.pth checkpoints + Master/sensitive/ are included).
#
# Exit codes: 0 ok, 2 usage error, 3 rclone not found, 4 remote not configured.
set -euo pipefail

DIRECTION="${1:-}"
GO="${2:-}"

REMOTE_NAME="gdrive"
REMOTE="${REMOTE_NAME}:protomf-thesis"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FILTER="Master/sync/drive_filter.txt"

# Locate rclone: PATH first, then the winget install location on Windows.
if command -v rclone >/dev/null 2>&1; then
  RCLONE="$(command -v rclone)"
else
  WIN_LA="${LOCALAPPDATA:-}"; WIN_LA="${WIN_LA//\\//}"
  RCLONE="$(ls "$WIN_LA"/Microsoft/WinGet/Packages/Rclone.Rclone*/rclone-*windows*/rclone.exe 2>/dev/null | head -n1 || true)"
fi
[[ -n "${RCLONE:-}" ]] || { echo "ERROR: rclone not found. Install it or fix PATH." >&2; exit 3; }

# Verify the Google Drive remote exists.
if ! "$RCLONE" listremotes | grep -qx "${REMOTE_NAME}:"; then
  echo "ERROR: rclone remote '${REMOTE_NAME}:' not configured." >&2
  echo "       Run:  rclone config   and create a remote named '${REMOTE_NAME}' (type: drive)." >&2
  exit 4
fi

case "$DIRECTION" in
  up)   SRC="." ;       DST="$REMOTE" ;;
  down) SRC="$REMOTE" ; DST="." ;;
  *) echo "Usage: drive_sync.sh <up|down> [--go]" >&2; exit 2 ;;
esac

cd "$REPO_ROOT"

ARGS=( sync "$SRC" "$DST"
  --filter-from "$FILTER"
  --transfers 8 --checkers 16
  --track-renames
  --progress )

if [[ "$GO" == "--go" ]]; then
  echo ">>> APPLYING  $DIRECTION  ($SRC -> $DST)"
  "$RCLONE" "${ARGS[@]}"
else
  echo ">>> DRY-RUN  $DIRECTION  ($SRC -> $DST)  -- no changes made. Add --go to apply."
  "$RCLONE" "${ARGS[@]}" --dry-run
fi

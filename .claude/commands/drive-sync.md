Sync the thesis repo to/from Google Drive via rclone. Excludes large datasets (`data/**`); includes literature PDFs and `*.pth` checkpoints. A third channel beside git (code) and scp (data/secrets). Usage: `/drive-sync up` (push local → Drive) or `/drive-sync down` (pull Drive → local).

**Argument:** `up` or `down`. If not given, ask which direction.

**Prerequisite (one-time per machine):** a Google Drive rclone remote named `gdrive` must exist. The script checks this and errors with exit 4 if missing. If missing, run the new-machine setup below. The synced folder is `gdrive:protomf-thesis` (created automatically on first push).

**⚠️ `up` is ALWAYS user-run — Claude cannot push.** Claude's harness HARD-BLOCKS any bulk upload of the working tree to the external Drive as data exfiltration; this is *not* limited to `Master/sensitive/` and user consent cannot clear it. Do NOT attempt to work around it (no excluding sensitive, no running rclone directly — all are blocked). For `up`, your job is only to prepare and hand off:
1. **Preview (dry-run):** run `bash Master/sync/drive_sync.sh up` (no `--go`). Dry-runs are read-only and allowed.
2. **Summarize the diff** — file count, total size, and any **deletions** (`sync` is a mirror: `up` deletes Drive files missing locally). Confirm excluded paths (`data/**`, `.git/`, caches) are untouched.
3. **Hand off:** tell the user to run it themselves — `! bash Master/sync/drive_sync.sh up --go` (the `!` makes it the user's own action, which is not gated). You cannot run the `--go` step.

**`down` is Claude-runnable.** Pulling Drive → local is inbound, not exfiltration, so follow the dry-run → confirm → `--go` flow yourself: `bash Master/sync/drive_sync.sh down` then, after the user approves, `bash Master/sync/drive_sync.sh down --go`. Note `down` can delete local files missing on Drive — surface that before applying.

**Secrets:** `Master/sensitive/` IS included in the filter (user's choice) and rides the user-run `up`. Relies on the Drive account being private — flag if it's ever shared.

**New machine — setup + how to sync *down*:**
1. **Install rclone** — Windows: `winget install Rclone.Rclone`; Linux: `sudo apt install rclone` (or the official script). Open a **new** terminal afterward so it lands on PATH.
2. **Configure the remote once:** `rclone config` → new remote named exactly **`gdrive`** (the script depends on this name), type `drive`, scope `1` (full access), browser OAuth. Leave client_id/secret blank, advanced config `n`, auto config `y`, Shared Drive `n`.
3. **Pull:** `git clone` the repo first (code), then `/drive-sync down` to pull PDFs + checkpoints.
4. **Secrets:** copy `Master/sensitive/` over via scp separately (see secrets boundary).

**Notes:**
- The rclone binary is auto-resolved by the script (PATH, else the winget install dir on Windows).
- What syncs is defined in `Master/sync/drive_filter.txt` (authoritative).

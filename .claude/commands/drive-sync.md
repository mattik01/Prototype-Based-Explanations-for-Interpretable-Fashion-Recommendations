Sync the thesis repo to/from Google Drive via rclone. Excludes large datasets (`data/**`); includes literature PDFs and `*.pth` checkpoints. A third channel beside git (code) and scp (data/secrets). Usage: `/drive-sync up` (push local → Drive) or `/drive-sync down` (pull Drive → local).

**Argument:** `up` or `down`. If not given, ask which direction.

**Prerequisite (one-time per machine):** a Google Drive rclone remote named `gdrive` must exist. The script checks this and errors with exit 4 if missing. If missing, run the new-machine setup below. The synced folder is `gdrive:protomf-thesis` (created automatically on first push).

**Protocol — ALWAYS dry-run first, then confirm:**
1. **Preview (dry-run):** run `bash Master/sync/drive_sync.sh <up|down>` (no `--go`). This makes no changes.
2. **Summarize the diff** for the user. Call out, in particular, any **deletions** — `sync` is a mirror: `up` deletes Drive files missing locally, `down` deletes local files missing on Drive. Reassure that excluded paths (`data/**`, `.git/`, caches) are never touched in either direction.
3. **Confirm:** ask the user to approve before applying. Wait for an explicit yes.
4. **Apply:** run `bash Master/sync/drive_sync.sh <up|down> --go`.
5. **Report** transferred / deleted counts from the output.

**⚠️ Secrets boundary:** Claude's harness **HARD-BLOCKS** pushing `Master/sensitive/` (API keys, IPs) to Drive — secrets must not cross to an external service when Claude initiates it, and user consent cannot override it. Do not attempt to work around the block. Two valid setups:
- **(A)** Keep `Master/sensitive/` excluded from `Master/sync/drive_filter.txt` and move it across machines via scp. → `/drive-sync` runs fully Claude-driven.
- **(B)** Include `sensitive/` in the filter, but the **user** runs the push themselves in their own terminal: `rclone sync . gdrive:protomf-thesis --filter-from Master/sync/drive_filter.txt --progress`. Claude cannot trigger it.

**New machine — setup + how to sync *down*:**
1. **Install rclone** — Windows: `winget install Rclone.Rclone`; Linux: `sudo apt install rclone` (or the official script). Open a **new** terminal afterward so it lands on PATH.
2. **Configure the remote once:** `rclone config` → new remote named exactly **`gdrive`** (the script depends on this name), type `drive`, scope `1` (full access), browser OAuth. Leave client_id/secret blank, advanced config `n`, auto config `y`, Shared Drive `n`.
3. **Pull:** `git clone` the repo first (code), then `/drive-sync down` to pull PDFs + checkpoints.
4. **Secrets:** copy `Master/sensitive/` over via scp separately (see secrets boundary).

**Notes:**
- The rclone binary is auto-resolved by the script (PATH, else the winget install dir on Windows).
- What syncs is defined in `Master/sync/drive_filter.txt` (authoritative).

Cleanly shut down the remote GPU machine via SSH and verify it goes offline.

Run: `echo y | bash Master/sensitive/shutdown_gpu.sh`

Invoking this command is the user's confirmation, hence the piped `y`. If someone is using the PC (console screen unlocked), the script demands a typed `yes` instead, so the piped `y` cancels — a locked console is normal after boot and does not block. The script sends shutdown, then polls both SSH and ping to confirm the machine is truly off. It retries up to 3 times. Check the output and exit code:
- **"Shutdown cancelled." (exit 0):** Someone is using the PC. Report it, show the session list, and ask the user — never re-run with `yes` on your own.
- **Exit 0:** Report success — GPU machine is confirmed offline (no SSH, no ping).
- **Exit 1:** Report failure — machine is still reachable after retries. Suggest the user SSH in manually (`ssh gpu`) and run `shutdown /s /t 0`.

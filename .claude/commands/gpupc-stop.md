Cleanly shut down the remote GPU machine via SSH and verify it goes offline.

Run: `bash Master/sensitive/shutdown_gpu.sh`

The script sends shutdown, then polls both SSH and ping to confirm the machine is truly off. It retries up to 3 times. Check the exit code:
- **Exit 0:** Report success — GPU machine is confirmed offline (no SSH, no ping).
- **Exit 1:** Report failure — machine is still reachable after retries. Suggest the user SSH in manually (`ssh gpu`) and run `shutdown /s /t 0`.

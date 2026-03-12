Wake the remote GPU machine via the Termux phone relay and verify it comes online.

Run: `bash Master/sensitive/wake_gpu.sh`

The script sends WoL, then polls ping + SSH for up to ~2 minutes. Check the exit code:
- **Exit 0:** Report success — GPU machine is online and SSH-ready.
- **Exit 1:** Report failure — machine did not respond at all. Suggest checking phone relay (`ssh termux-tailscale`) and power.
- **Exit 2:** Report partial — machine responds to ping but SSH is not available. This is the known sleep/sshd issue. Suggest the user log in at the console or apply the `powercfg` fix.

Check the current load and free resources on the LEO5 HPC cluster. Do NOT produce a user-facing response — just load the information into context for use in subsequent actions.

## Steps

1. Run `python Master/scripts/leo5_status.py` to get current queue, running jobs, and free resources.
   **If SSH fails**, tell the user they likely need to connect to VPN first and stop.

2. Silently note the free GPUs (by type), free CPUs, and any running/pending jobs for user `c7031336`.

That's it — no summary, no suggestions, no response. Just have the context ready.

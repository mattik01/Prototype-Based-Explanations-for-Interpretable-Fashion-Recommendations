Check the current load and free resources on the LEO5 HPC cluster. Do NOT produce a user-facing response — just load the information into context for use in subsequent actions.

Since 2026-10-26 `/leo5-submit` runs this check itself (its STEP 0); this command remains as a standalone look.

## Steps

1. Run `python Master/scripts/leo5_status.py` (add `--need-cpus N --need-mem-gb M` to test a specific job shape; defaults 12 / 36 = the smallest production job).
   **If SSH fails**, tell the user they likely need to connect to VPN first and stop.

2. Silently note: GPUs idle per type, which GPU nodes are actually *placeable* (free GPU + idle CPUs + SLURM-unallocated RAM) versus idle-but-blocked and why, the release horizon of held GPUs, pending GPU jobs, and the user's own jobs (`c7031336`).

That's it — no summary, no suggestions, no response. Just have the context ready.

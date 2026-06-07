---
date: 2026-06-07
time: "23:10"
phase: 3
---
# Compute feasibility envelope for experiments (LEO5)

Established the planning limits that drive dataset and experiment design: all runs execute on the LEO5 cluster, with **no run fragmenting** (a run must complete in a single job). Hard limit — anything expected to run **> 10 days is infeasible** (LEO5 `std`-partition wall limit is 10 days; this corrected an earlier mistaken belief in a 12 h cap, which was actually a self-imposed smoke-test setting, not a cluster limit). Soft limit for development/testing — anything **> ~1 day is infeasible**, aim well below.

[[compute]] [[decisions]] [[phase-3]]

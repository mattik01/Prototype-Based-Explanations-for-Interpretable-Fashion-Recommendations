# dc01/dc05 walltime probes — 2026-07-15 (working input for SC.6 run plans)

**Method:** 3-epoch single-fixed-config probes on the LEO5 login-node A30 (solo GPU),
via the `--retrain-config` path; fI/fU configs mirror their hosts' canonical best
(dim 81, K 76, host reg weights, `use_id_feature=1`, canonical fields; synthesized at
`/scratch/c7031336/timing_probes/confs/`). Estimates scale the hosts' *measured* S0.7
dev-fleet wall times (30 trials / conc 5 / ASHA / A30) by the solo per-epoch ratio
variant÷host — the host anchor carries the hyperopt dynamics, the ratio carries only the
architectural overhead. Probe artifacts: `/scratch/c7031336/timing_probes/` (never
reportable). Raw log: `timing_probes.log` (+ `rerun_fi_ml1m.log` — first fI×ml-1m
attempt failed on the then-missing `item_features.csv` on scratch; file scp'd, learnings
entry 14efb96).

## Measured steady-state s/epoch (solo A30)

| | hm_1_month | ml-1m |
|---|---|---|
| item_proto (host) | 126.8 | 73.4 |
| **feature_item_proto** | 131.2 (**×1.04**) | 69.4 (**×0.95**) |
| user_proto (host) | 133.2 | 63.3 |
| **feature_user_proto** | 367.9 (**×2.76**) | 62.3 (**×0.98**) |

Readings: **fI is compute-free over its host** (both datasets). **fU costs ×2.76 on
H&M** — the padded per-user history gather (D_max = 112 rows summed per batch element)
is the hot path; on ml-1m (bags layout) it is free. Ratios measured at one config point
(top of the search ranges — conservative); fU's ratio under 5-way GPU sharing may shift,
the 2× margin absorbs it.

## Recommended SLURM time limits (dev-profile hyperopts, fleet shape)

| Run | Estimate | `--time` |
|---|---|---|
| fI hm_1_month (headline + noid/f0 arms) | 2h08m × 1.04 ≈ 2h13m | **8:00:00** |
| fU hm_1_month (headline + noid arm) | 3h22m × 2.76 ≈ **9h17m** | **1-00:00:00** |
| fI ml-1m | 2h08m × (73.4/126.8) × 0.95 ≈ 1h10m | **4:00:00** |
| fU ml-1m | 3h22m × (63.3/133.2) × 0.98 ≈ 1h34m | **4:00:00** |
| ml-1m host fleet rows (charter-lazy: mf, user_proto, item_proto, lightfm…) | ~1h–1h40 each | **4:00:00** |

Anchors: item_proto hm dev fleet 2h08m, user_proto 3h22m (S0.7 wave, jobs 6988961/62).
Cold-variant single-config retrains ran 0h45–2h04 in the S0.7 wave → cold retrains for
dc01/dc05 arms should get **4:00:00** (acf came within 9 min of a 2h limit).

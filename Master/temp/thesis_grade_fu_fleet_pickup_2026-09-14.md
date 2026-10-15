# Thesis-grade fU fleet on ml-1m — pickup note (queued 2026-09-14 ~12:10)

## Why
dc08 (identity rows) read as a controlled negative at dev tier on 2026-09-14
(see the dc08 doc + this session's discussion): C − A = +0.005 HR@10 / 0.000 NDCG on
identical negatives, sign flips under the in-run negative draw, and on the one
hyperparameter config shared by all four arms (adagrad, s_sm, d=81, K=76) identity
rows cost ~0.010–0.013 val HR@10 on both pairs. ID-row additivity held (B−A = 0.012,
D−C = 0.0125). fI′ centring dev→prod pairs showed 100 trials buys only +0.003…+0.007.

No fU arm existed at 100 trials; the host user_proto ml-1m production number (0.5933)
is a Windows-desktop artifact (99/100 trials, hand-picked best). So the thesis-grade
fU line is queued as a matched fleet — first thesis-tier rows on the user side.

## The five jobs (production: 100 trials / 100 epochs / patience 10 / grace 4)

| job | model | role |
|---|---|---|
| 7799617 | `user_proto` | clean cluster host row (replaces desktop artifact) |
| 7799618 | `feature_user_proto` | fU |
| 7799619 | `feature_user_proto_noid` | fU-noid |
| 7799620 | `lightfm_hist` | dot twin |
| 7799621 | `feature_user_proto_itemid` | dc08 arm D (user request; accuracy read only, C9 non-identified) |

Pinned fleet shape: seed 38210573, concurrency 5 (`--gpu-per-trial 0.2`), `--num-workers 1`,
10 CPUs / 40G / 1× any GPU, ASHA on, `hit_ratio@10`, explanations ON, tag `thesis-grade`,
time cap 1-00:00:00 (dev wall 1h17–2h03 × ~3.9 × 2 safety). All arms train with
leave-one-out pooling on (July dev twins did NOT — that confound is gone at this tier).

Results land in `/scratch/c7031336/protomf_results/<model>_ml-1m_s38210573_n100/`
(dev twins in the un-suffixed dirs stay untouched). Code at 694e1ee on both sides.

## When they land
- Run `Master/scripts/stratified_readout.py --results-dir <dir> --device cuda` on the
  login node for each (identical-negatives regime; the summary.md test number uses a
  different draw — do not mix the two).
- Reads: fU vs user_proto (host cost of the composition), fU vs lightfm_hist (head cost),
  fU vs fU-noid (ID row), D vs fU (identity rows at thesis tier).
- Two-tier policy: these are thesis-grade rows but SINGLE seed; 3-seed averaging is the
  later thesis-tier step.

## Also done this session (not yet mirrored to laptop)
Stratified readouts for the two dc08 dev checkpoints written on scratch:
`feature_user_proto_noid_itemid_ml-1m_s38210573/stratified_readout/` (0.5366) and
`feature_user_proto_itemid_ml-1m_s38210573/stratified_readout/` (0.5491).

## RESULTS (all five COMPLETED 2026-09-15, 100/100 trials, exit 0; readouts run 2026-09-17)

| model | summary test HR@10 / N@10 | identical-negatives readout HR@10 / N@10 | winner (loss, d, K, B) |
|---|---|---|---|
| user_proto (host) | 0.6142 / 0.3587 | 0.6130 / 0.3537 | s_sm, 72, 88, 256 |
| lightfm_hist (dot twin) | 0.6238 / 0.3665 | 0.6221 / 0.3639 | s_sm adam, 62, –, 128 |
| feature_user_proto_noid | 0.6082 / 0.3507 | 0.6041 / 0.3467 | s_sm, 70, 74, 64 |
| feature_user_proto (fU, ID row) | 0.5613 / 0.3189 | 0.5631 / 0.3204 | **bce**, 92, 98, 256 |
| feature_user_proto_itemid (arm D) | 0.5920 / 0.3381 | 0.5888 / 0.3363 | s_sm, 89, 94, 256 |

Reads (readout regime): noid − fU = **+0.041** (dev tier July was −0.012 → REVERSED);
noid − host = −0.009; dot − noid = +0.018 (head cost shrank from 0.068 at dev to 0.018);
D − fU = +0.026 (identity rows help WITH the ID row present; noid without either still best).
Top-8 trials per run form plateaus (fU 0.58–0.60, noid 0.615–0.632), so not single-trial luck.
Tail bucket (21–100 pop): fU 0.153, dot 0.174, all s_sm arms 0.05–0.06 — fU's bce winner trades
head for tail. Confounds vs July dev twins: LOO pooling now on, budget 30→100.

## Why the ID row reversed (dug in 2026-09-17; geometry + trial tables on scratch)
- July fU ID-energy share 0.003–0.07 (inert); prod fU 0.016–0.15 (still small) → "ID absorbed it" is NOT the mechanism.
- Loss regime: prod fU winner bce, B-variance share 0.26 vs 0.74 (noid, host) → weak popularity channel, wins tail (0.15 vs 0.05) loses head.
- Per loss, best val: fU s_sm 0.597 (31 trials) vs noid s_sm 0.632 (59) vs D s_sm 0.622; fU bce 0.601 vs noid bce 0.574 → ID row hurts under s_sm, helps under bce.
- Shared warm-up config (s_sm, d=81, K=76): July fU 0.5741 / noid 0.5618 → prod fU 0.5595 / noid 0.5676; host 0.5958; D 0.5658. LOO+cap swing ≈0.02 of the ~0.04 gap; rest = noid's search reaching batch-64/low-lr region fU's s_sm trials never did.
- sc8_fu_geometry_readout FAILS on arm D (`hist_value_ids` missing on the bag layout) — instrument gap, not a result.

---
date: 2026-09-08
time: "15:09"
phase: 4
---
# fI′ centring cycle closed: field-mean centring rejected; dc01 ml-1m regression reattributed to hyperopt budget

**What was run.** The fI′ single-variable candidate (fI + U2b field-mean centring of the
composed feature vector, `center_fields=True, center_mode=weighted`, knobs not a new ft_type)
was taken to the 100-trial prod budget on ml-1m for both arms (fI with IDs, fI-noid), seed
38210573, jobs 7767219 / 7767215, both `RUN_OK 100/100`. Together with the earlier 30-trial
centred rows (2026-08-25/26) and the uncentred 100-trial fI/noid rows (2026-08-24), the
ml-1m picture at matched budget is (test HR@10 / NDCG@10, single seed):

| Row | Trials | HR@10 | N@10 |
|---|---|---|---|
| item_proto host, paper figure | paper | 0.544 | 0.303 |
| item_proto host, our replication | 100 | 0.5723 | 0.3191 |
| fI uncentred | 100 | 0.5628 | 0.3204 |
| fI centred | 100 | 0.5123 | 0.2825 |
| fI-noid uncentred | 100 | 0.5156 | 0.2822 |
| fI-noid centred | 100 | 0.5297 | 0.2923 |
| fI-noid centred, interaction-weighted | 30 | 0.5288 | 0.3010 |

**Result 1 — centring splits by sign and hurts the candidate.** With IDs on, centring costs
≈0.05 HR@10 against the uncentred 100-trial fI (0.5123 vs 0.5628), pushing fI down to the
mf / noid level. With IDs off it gains ≈0.014 (0.5297 vs 0.5156). The centred-ID run's best
trial was the 12th started and is config-identical to the 30-trial winner — trials 31–100
bought nothing, the centred-ID landscape is saturated around val 0.525. The noid winner
came at trial 97, so there the budget mattered, but the gain is one-seed-small and noid is
an ablation, not the candidate.

**Result 2 — the dc01 ml-1m "regression" was a budget artefact.** The −0.0555 HR@10 host
deficit recorded in the dc01 chapter memo (sc01) compared a 30-trial fleet fI (0.5156) with
the replicated host. At 100 trials uncentred fI reaches 0.5628 / 0.3204: 0.0095 below the
replicated host on HR@10, at or slightly above it on NDCG@10, and ≈0.02 above the paper's
printed item_proto figure on both metrics. The memo's "real regression" language must be
revised to "parity at matched budget; the 30-trial deficit was hyperopt-budget-driven".
This also removes the empirical motivation for centring: the composition-expressivity
hypothesis that predicted an ml-1m tax was never actually observed once budgets match.

**Decision.** Centring is rejected, not parked. Reasoning: (i) the observation it was built
to fix does not exist at matched budget; (ii) it damages the actual candidate by a large,
unambiguous margin; (iii) the only gain is on the ablation, small, one seed, and still
0.04 below the host; (iv) no H&M centred rows were ever run, and spending two more
submissions to confirm a negative on the standing reference benchmark is not worth it.
The knobs stay in the code as documented-off options; the interaction-weighted variant
(one 30-trial noid row) is recorded here as untested further. Matteo's read: for fI there
is currently nothing left on the table that makes it better — the composition-upgrade
line for fI is closed; attention moves to the fU vocabulary question.

Caveats: single seed throughout; the replicated host row is from an older wave on different
hardware, so a few thousandths of drift against the fI rows is expected noise.

[[experiments]] [[decisions]] [[evaluation]] [[phase-4]]

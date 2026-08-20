# F-S0-14 tie-bracket sweep — all 11 cold rows (2026-08-20)

Instrument: `tie_bracket()` in `utilities/cold_eval.py` (commit a8b0d53); measurement run on the
LEO5 login node over the 11 landed `*_hm_1_month_cold_s38210573` runs, replicating the exact
2026-08-15 invocations (`run_cold_evals.sh` pattern + the dc01/dc05 log headers). Pre-bracket
reports preserved on the cluster at `<run>/cold_eval_pre_bracket/`; new reports mirrored locally
at `Master/experiments/cold_eval_brackets/<model>/`.

**Verification:** every point metric (warm test, cold-vs-cold, cold-vs-all, attr-kNN blocks) is
**bit-identical** old vs new across all 11 rows — the instrument changed nothing it measures.

## Headline: the F-S0-14 confound did NOT materialize — every claim-carrying row is tight

HR@10 cold-vs-cold, worst / expected / best (the row's *quoted* block in bold context):

| row | block | HR@10 w/e/b | straddle | any-tie | mean/max block |
|---|---|---|---:|---:|---|
| **lightfm_tags** (cold bar) | native | 0.4700 / 0.4706 / 0.4711 | 0.001 | 0.058 | 1.06 / 6 |
| **lightfm_tags_ids** (cold bar) | native | 0.5074 / 0.5081 / 0.5089 | 0.001 | 0.058 | 1.06 / 6 |
| **fI** (dc01) | native | 0.2869 / 0.2876 / 0.2882 | 0.001 | 0.058 | 1.06 / 6 |
| **fI-noid** (dc01 ablation) | native | 0.3875 / 0.3884 / 0.3892 | 0.002 | 0.058 | 1.06 / 6 |
| **fU** (dc05) | attr-kNN | 0.3259 / 0.3266 / 0.3274 | 0.002 | 0.059 | 1.07 / 6 |
| **fU-noid** (dc05 ablation) | attr-kNN | 0.3412 / 0.3419 / 0.3427 | 0.002 | 0.059 | 1.07 / 6 |
| user_proto | attr-kNN | 0.2993 / 0.3000 / 0.3007 | 0.001 | 0.059 | 1.07 / 6 |
| item_proto | attr-kNN | 0.2352 / 0.2359 / 0.2366 | 0.001 | 0.059 | 1.07 / 6 |
| user_item_proto | attr-kNN | 0.4148 / 0.4156 / 0.4163 | 0.002 | 0.059 | 1.06 / 6 |
| acf | attr-kNN | 0.3090 / 0.3096 / 0.3103 | 0.001 | 0.059 | 1.06 / 6 |
| mf | attr-kNN | 0.2449 / 0.2453 / 0.2458 | 0.001 | 0.058 | 1.06 / 6 |

- Bracket widths ≤ ~0.001 HR on every quoted row. Tie exposure inside the actual 99-negative
  ranking is tiny (any-tie ~6% of rows, straddle ~0.1–0.2%, mean tie block 1.06) — the
  full-catalog twin shares (0.99 on the LightFM rows and fI-noid) were exposure *indicators*,
  and the measured exposure they indicated does not reach the reported metric. Reason: a tie
  costs the target only when a sampled negative is an exact signature twin *of that target*,
  and 99 uniform draws from the cold pool rarely contain one.
- **The blocked orderings survive with non-overlapping brackets** and are hereby measurable:
  fI-noid worst (0.3875) > fI best (0.2882); the LightFM cold bar worst (0.4700) clears every
  other row's best; the S0.7 patched-CF ordering is unchanged. F-S0-14 (b)'s claim freeze can
  lift as soon as the (c) gate is decided.
- The degenerate native blocks behave exactly as the instrument predicts: `item_proto` /
  `user_item_proto` native cold-vs-cold are 100-way raw-score ties (bracket [0, 1], expected
  0.10 = chance) — their historical 0.0000 points were worst-case realizations of a chance-floor
  bracket. `acf` native is partially degenerate (mean block 38, bracket 0.08–0.14). None of
  these blocks carries a claim (the quoted rows are the patched ones).

## New finding (raised as F-S0-15): the eval sigmoid fabricates ties the raw ranking doesn't have

The `point_within_bracket` flag fired on five blocks, all with the same mechanism, verified
directly on the mf row: cold-vs-cold raw logits are numerical dust (|x| ≲ 1e-8 — near-zero cold
embeddings), all *distinct* in float32, but `torch.sigmoid` maps them to bit-equal 0.5;
argpartition then resolves the fabricated 99-way tie against the target.

| row | block | reported point HR@10 | raw-ranking bracket |
|---|---|---:|---|
| mf | native | 0.0010 | [0.0953, 0.0953] (tie-free, ≈ chance) |
| user_proto | native | 0.0000 | [0.1081, 0.1081] |
| fU | native | 0.0000 | [0.0955, 0.0955] |
| fU-noid | native | 0.0000 | [0.1133, 0.1133] |
| mf | attr-kNN | 0.2413 | [0.2449, 0.2458] |

- The dossier reading "unpatched cold-vs-cold = 0.0000 … established shape, not a defect" was
  wrong in an instructive way: the honest raw-ranking value of those blocks is the **chance
  floor (~0.10)**, not zero — no signal either way; 0.0000 was sigmoid collapse + argpartition
  convention. No landed claim is damaged (these blocks carry none), but the wording in the
  dc05 dossier ("the patched line is the one that carries meaning; unpatched is 0.0000,
  established shape") needs a dated correction at the next dossier touch.
- **mf attr-kNN is the one quoted row affected:** its point (0.2413) sits ~0.4pt below its own
  raw-ranking bracket (0.2449–0.2458) — sigmoid float32 rounding merges near-equal patched
  logits often enough to depress the point. No ordering changes (mf stays last among patched
  CF rows either way), but the honest quoted value is the bracket.

## Open user gate (F-S0-14 (c)) — now decidable on numbers

Options: (1) re-open the frozen S0.7 table, or (2) keep the table and layer the brackets as
standing disclosure. The measurement argues for (2): all quoted points sit inside tight
brackets (single exception mf attr-kNN, off by 0.4pt with no ordering effect), so re-opening
would change nothing material. Decision is Matteo's, not the protocol's default.

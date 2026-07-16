# dc01 SC.6 pause-filler — explanation-apparatus check (2026-07-15)

Toy + real-checkpoint exercise of the full explanation apparatus while the
SC.6 fleet runs. **Working evidence only** (smoke/toy checkpoints — nothing
here is a compared or reportable number).

## Contents

- `toy_fi_pipeline/` — full `run_explanations_pipeline` output on a trained
  toy fI checkpoint (t12 harness, 120 users × 300 items, canonical-5 toy
  vocab): `breakdown/` (auto-explainer artifact), `cosine/` (intrinsic pass:
  prototype cards, t-SNE, small multiples, naming CSVs, breakdown),
  `lift/` (dual post-hoc naming route, F-DC01-13).
- `toy_unit_slots/` — renderer unit-slot artifacts from t12 section A–C
  (fI slot, statement-only slots: mf, popularity).
- `real_lightfm_ml1m/` — breakdowns rendered from the login-node smoke's
  persisted `lightfm_tags × ml-1m` checkpoint (2-trial/3-epoch smoke — real
  data, barely-trained model): the bags-layout label chain (F-DC05-13
  machinery) on the real 1,061-token genres+tags vocabulary.

## Verdict

**Apparatus sound.** t12 all-pass (37 checks); arithmetic hand-verified on the
rendered numbers (baseline + discriminating = S; prototype bars sum exactly;
zoom rows sum exactly to their bar; feature-vs-ID footer consistent); honesty
markers present in figure + md, effect-epistemics correctly absent (F-DC01-12
re-scope); ID row hatched, own line, never folded; dual naming routes both
written with plausible partial agreement; statement slots correct (popularity
"rank 1 of 202 = top 0.5%" — the SC.5 off-by-one fix confirmed live); real
ml-1m render carries true item identity + tag labels with exact sums.

## Findings (no fixes applied — fleet in flight)

1. **Bags-fold dominance (design, route to SC.8 figure work):** on ml-1m a
   popular movie carries ~30–57 tokens, so the fixed top-6 + fold zoom makes
   **"(+ 50 smaller lines)" the largest bar** (user7/GoodFellas: +0.236 of
   S = +0.357, ~66% of mass folded). Arithmetic exact, but the figure reads
   oddly when the remainder dominates. Options to weigh at SC.8 (shared-frame
   contract applies): per-layout top-N, or a declared "long-tail" rendering.
2. **Near-duplicate tag tokens** ("violence" / "violent" as separate lines) —
   a vocabulary property of the Tag-Genome, not an apparatus bug; relevant to
   naming/label dedup discussions (hidden-effects adjacent).
3. **Cosmetic:** the md companion's zoom heading inherits the figure's
   ellipsis truncation ("inside p2 · Jersey, White, B…") though md has no
   width constraint.
4. `real_lightfm_ml1m/breakdown_user1234_item561` shows S = +0.0000 — a
   property of the barely-trained smoke checkpoint (near-zero embeddings),
   not of the renderer; self-check passed.

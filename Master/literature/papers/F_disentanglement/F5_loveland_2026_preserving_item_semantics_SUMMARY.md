# F5 — Loveland, Collins, Kumar, Koutra, Shah 2026, "Preserving Item Semantics for Free: Rethinking Token Initialization in LLM-Based Generative Recommendation"

Mini-summary, thesis-relevant only. arXiv:2608.07816 (Snap Research / U-Michigan, Aug 2026; no venue yet — re-check at BibTeX time). PDF + guide-annotated copy in this folder. Read 2026-08-15 (companion session). Insights in `protocol_vault/` (four entries, linked below).

## One line
Randomly initialized semantic-ID token embeddings reorganize around item **popularity** instead of semantics during training — a dedicated re-grounding stage (CPT) cannot recover them — while parameter-free **centroid initialization** preserves the semantic geometry for free, with the largest gains on cold/tail items.

## Why it matters to us
- **External + mechanistic corroboration of our own finding**: an independent group, in a wholly different architecture (generative LLM recsys), measured the same pathology class as our S0.7 prototype collapse ([[2026-07-15_1112_prototype-collapse-is-a-self-built-popularity-bias]]) — item representations degenerating into a popularity channel — on five datasets incl. MovieLens, with a gradient-level proof (Thm 3.1). Mechanisms differ (their CE-gradient accumulation vs our duplication-as-coefficient-amplification): cite as *convergent finding*, never "same mechanism".
- **Intrinsic-over-post-hoc, demonstrated in recsys**: semantics injected structurally (init) beats semantics bolted on by a training stage — Rudin's axis as an empirical result.
- **Instrument quarry** for SC.8 / Ch 10: P_m neighbourhood purity, R²_q latent-signal dominance, σ_geo — all need port amendments (entries below).
- **Centroid init = a Ch 11 knob candidate** (initialization as a missing adjustment class; cheapest knob conceivable; only *composed* models possess it — host free embeddings have no semantic space at t=0). dc04's anchor-centroid construction salvages as the init target without inheriting its readout defect.

## Most citable for (★ = strongest)
- ★ **CPT fails to recover discarded semantics** (§3.1, Fig 2): post-training P₁₀ = 0.24–0.37, barely above the random-neighbour floor, vs 0.58–0.93 in the source space — even after a stage explicitly designed to re-ground tokens in text.
- ★ **Popularity owns the learned geometry** (§3.2, Table 1): top-3 PCs capture ≥ 0.80 of popularity signal (pure-SFT), semantics negligible; **Theorem 3.1** (§3.3): GD adds η·ñ_v·Σh̄ to every code — a shared popularity axis written in by the optimizer, read out as a logit boost.
- ★ **Centroid init works** (§4.2, §5): purity ~doubles; R²₅(T_pop) 0.89 → 0.06 (Beauty); body averages **+6.5% Recall@5, 27% fewer SFT steps, +37% cold-item Recall@5** (quote these, not the abstract's "up to 16/40/60" single-dataset maxima).
- **Shuffled-centroid ablation** (App. H): permuting the code↔centroid assignment degrades results — the *semantic alignment* carries the benefit, not the marginal geometry statistics.
- **CPT as partial popularity debiasing** (App. E): deeper CPT lifts tail recall 34–75% relative, but head/tail ratios stay far from 1 — dilution, not repair.

## Coverage map
§1–2 LLM/SID plumbing (skim) · **§3 diagnostics (core: 3.1 purity, 3.2 spectra, 3.3 theorem)** · **§4 centroid init (core; mean-shift in 4.1)** · §5.1 results · **§5.2 cold items (core, but see regime caveat)** · App. B/C prompts (skip) · App. D/E/H selectively (E, H cited above).

## Our insights from reading it (protocol links)
- P_m port: addressability ≠ metric; retention-vs-demotion; per-field ground-truth profile vs the paper's silent single-column choice; port plan (start 1 column) — [[2026-08-15_1639_pm-purity-port-groundtruth-target-discussion]]
- σ_geo doesn't port: instrument invariances must match readout invariances (their effective-rank critique inverts under cosine); norm regains meaning at the double tie — [[2026-08-15_1651_spectral-scale-gauge-inversion-direction-diversity-metric]]
- Merge norm entanglement: shared rows weld popularity channel to direction structure; two-sided refund upgrade; built-in 2×2; noid/ids expressivity resolution — [[2026-08-15_1721_merge-norm-entanglement-shared-rows-two-sided-refund-2x2]]
- R²_q port notes (brief): dominance-not-presence; normalized + **partialed form required** (our targets correlate, ρ=0.672 ml-1m); noid>ids contamination prediction — [[2026-08-15_1729_r2-latent-signal-port-notes-brief]]

## Caveats when citing
- **Regime mismatch everywhere**: generative LLM sequential rec, full-catalog constrained beam, Recall@5 — we are MF item-ranking, 1+99 sampled, HR@10. Positioning and mechanism evidence, not a runnable baseline.
- **Their "cold" is our "tail"**: cold cohort = test targets appearing ≤1× *as supervision target* (items still appear in input histories → embeddings still trained), ranked against the full warm catalog. One full regime warmer than our zero-shot holdout (S0.3 variant) — never quote their cold gains beside our cold rows without this flag. No tie handling at all (contrast F-S0-14 machinery).
- Their σ_geo argument and unpartialed twin-R² are correct in their space and wrong in ours (entries above) — don't import either uncritically.
- Fix transfer to our prototypes is a **registered hypothesis, not a result** (roadmap thread): our collapse mechanism is training-time; a better starting point may not resist it.

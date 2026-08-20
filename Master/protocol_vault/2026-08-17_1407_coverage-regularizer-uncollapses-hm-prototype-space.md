---
date: 2026-08-17
time: "14:07"
phase: 4
---
# The coverage regularizer un-collapses the H&M prototype space — collapse = data pressure × regularizer config

Probe run during deep-dive 1.2 (testing the history-length hypothesis, [[2026-08-17_1407_user-history-gap-quantified-hm-vs-ml]]): regenerated explanations for the old **user_proto_hm_3_month** checkpoint (current pipeline, login node). Result: **58 prototypes → 56 distinct lift names, largest clone block = 2. Essentially zero collapse — on H&M, at median history 7** (vs hm_1_month's 5 → 5 distinct names out of 76, block of 37). If history length were the driver, +2 median purchases could not rescue anything.

**Confound check → the real difference.** Same loss (sampled_softmax), same negative strategy (uniform); but:

| run | K | dim | sim_proto_weight (herd) | sim_batch_weight (cover) |
|---|---|---|---|---|
| hm_1_month (collapsed) | 76 | 81 | 1.80 | **0.0018** |
| hm_3_month (healthy) | 58 | 46 | 3.02 | **5.71** |

A ~**3000×** difference in exactly the term that matters: `sim_batch_weight` is the **coverage regularizer ("every batch user must be near some prototype")** — the force that spreads prototypes over the user cloud. The 1-month winner had it effectively OFF (collapse free to happen, herding term dominant — see the three-force resolution in [[2026-08-17_1128_shifted-cosine-baseline-asymmetry-popularity-channel]]); the 3-month winner had it strong (collapse suppressed).

**Reframe:** prototype collapse is **not data-fated** — it is *data pressure × regularizer configuration*. The popularity channel supplies the pull; whether the prototypes actually get recruited depends on whether anything pushes back, and sim_batch is that push.

**Candidate decisive experiment (cheapest yet, not queued):** rerun hm_1_month with its own winning config, only `sim_batch_weight` raised to ~5.7 — one mechanism-tier run isolating whether the coverage term alone un-collapses the 1-month space. This knob may matter as much as cosine_type × use_bias for the popularity-channel thesis section; candidate third axis for that section's experiment family (relates to the cosbias2x2 wave, jobs 7426702–7426722, and the base-settings gate in [[2026-08-17_1320_popularity-stance-2x2-figure-plan-base-settings-gate]]).

Caveats: the 3-month probe is NOT a controlled comparison (different K, dim, user count, catalog); it disproves "history is binding" and spotlights the regularizer, but the clean attribution needs the isolating rerun. Scrutiny-evidence-not-thesis applies.

[[phase-4]] [[explanations]] [[experiments]] [[decisions]]

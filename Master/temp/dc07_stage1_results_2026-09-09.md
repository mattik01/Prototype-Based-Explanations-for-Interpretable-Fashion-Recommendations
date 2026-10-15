# dc07 stage 1 — first host run: `user_proto_sm × ml-1m` (job 7773952)

**Dev tier, seed 38210573, 30 trials / 60 epochs, conc 5, A30-class node n029, 2h15m, completed
2026-09-09 01:54.** Results `/scratch/c7031336/protomf_results/user_proto_sm_ml-1m_s38210573/`
(+ `dc07_membership/`, `sc8_fu_geometry/`). Working evidence only (two-tier policy).

## Numbers

| | HR@10 | NDCG@10 |
|---|---|---|
| **user_proto_sm (softmax membership)** | **0.4463** | **0.2424** |
| user_proto cosine host (dev, frozen) | 0.5668 | 0.3230 |
| mf (dev) | 0.5022 | 0.2793 |
| acf (prod; membership-family anchor) | 0.6006 | 0.3364 |

Winner: τ = 1.145 (log-position 0.81, not at an edge), K = 81, d = 32, sampled_softmax, adagrad
lr 0.099, wd 3.7e-4, **sim_batch_weight 0.91 (row 'max'), sim_proto_weight 0.0018 (column 'max')**.

## Read-outs (pre-registered §8)

- **Void check (uniform regime): NOT triggered.** Only 3.3 % of users have H > 0.9·log K; mean
  H/log K = 0.19, 34 % in the hard regime. ‖q‖ mean 0.45 — the table did not shrink to zero as in
  the hm smoke. The weight-decay-to-uniform route did not fire here.
- **Popularity rediscovered the OTHER way — through one dominant prototype.** Mean membership
  vector peaks at 0.67 on prototype 65 (uniform would be 0.012); it is the argmax for 76 % of
  users; the top-prototype score share over the top-10 recommendations is **0.97** (p50 0.998);
  the per-item channel Σ_l t_{i,l}·mean_m_l correlates **0.98 Spearman with item popularity**.
  The model is a popularity ranker with a sharp, near-one-hot membership on an "everyone"
  prototype — B(t) rebuilt at the layer level, exactly the §9 mechanism, but via hard assignment,
  not via τ→∞.
- **Geometry: collapsed harder than the cosine host.** Activation effective rank **1.59**
  (numerical rank 20 of 81); user-cloud effective rank 3.9 (uniform) / 6.2 (interaction-weighted);
  coverage P→U max-sim mean 0.22 with p1 = 0.012 — most prototypes have no user for whom they are
  the top membership. The cosine host on ml-1m had 11–15 effective directions (vault 2026-08-17_1937).
- **One thing that works as designed (P4):** membership entropy falls with history length
  (Spearman −0.31 vs |H|, −0.23 vs ‖q‖): users with more evidence get sharper memberships. The
  "confidence" reading of the norm is real; it just sharpened onto the wrong (shared) prototype.

## Reading

Verdict by the pre-registered rule: **B** (accuracy below the host by 0.12 HR@10, forty times
the hm yardstick) **with A's geometry** (collapse not cured, worse). The likely mechanism is
visible in the winning config: under softmax the row-'max' regulariser (weight 0.91, the largest
in the search space's upper decade) maximises each user's top membership — i.e. it rewards hard
assignment — while the column-'max' term that would keep every prototype owned is switched off
(0.0018). Hyperopt on HR@10 chose the config that turns the layer into "assign everyone to the
popularity prototype", because on this host that scores 0.45 and nothing else in the 30 trials
scored higher. D1 anticipated this ("'max' becomes sharpen, record this"); the ACF-style
inclusiveness term (spread the load over prototypes) is the direct counter and was reserved as
the stage-2b axis. Duplicate/unused prototypes still cost nothing (P1 untouched by dc07).

## Options (Matteo's call; the gate says stop unless a follow-up recovers it)

1. **Stop dc07 on the host** — controlled negative for the knobs chapter: "replacing the
   cosine by a membership does not move the bottleneck; the popularity channel re-forms through
   a dominant prototype." Cheapest; the hm host run would only confirm the mechanism.
2. **One more host run with the membership-native regularisers** (`reg_proto_type='incl'` +
   `reg_batch_type='soft'`, already implemented; or ACF's exc/inc pair): the inclusiveness term
   directly penalises the 0.67-on-one-prototype load. Second variable, but it is the variable the
   result points at. ml-1m again, dev tier, one submit.
3. **Bias arm (D6, `use_bias=1`)** — gives the popularity channel its own slot so prototypes need
   not carry it. Less targeted than 2; the cosbias2×2 showed bias recovers only part of the +1 loss.
4. **Skip to fU on hm regardless** (the shared-direction argument): the host result says the
   layer collapses onto a popularity prototype when it can; on the composed branch the same
   pressure exists plus the shared direction — a composed run without fixing 2 first is unlikely
   to read differently.

Not queued: item_proto_sm, user_proto_sm × hm, and the composed arms — pending this decision.

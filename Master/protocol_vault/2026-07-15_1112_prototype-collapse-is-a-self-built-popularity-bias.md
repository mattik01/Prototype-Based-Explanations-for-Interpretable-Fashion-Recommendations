---
date: 2026-07-15
time: "11:12"
phase: 4
tags: [thesis/hidden-effects, thesis/knobs-metrics, thesis/interpretability-objective]
placed: 2026-08-20
---
# Prototype collapse in the reference fleet is a self-built popularity-bias channel

**Status of the evidence:** single seed (38210573), dev-profile configs, June checkpoints
(≡ S0.7 canonical per the Q1 cross-check), all measurements scrutiny-phase **working
basis** — thesis treatment requires dedicated experiments designed at writing time
(standing rule). Captured under the verbatim rule because the argument chain was worked
through interactively and is destined for the hidden-effects section.

## 1. How it was found

While eyeballing the S0.7 explanation artifacts (first real look at the reference fleet's
explanations), the `user_item_proto` weight-viz footer listed its three "top prototypes"
p6, p1, p7 **with the identical name** ("Unknown, Unknown, Denim Other Garments"). That
looked like a labeling bug. It wasn't: the naming CSV shows prototypes 1, 5, 6, 7 with
byte-identical descriptors *and scores*, and the top-k table shows all four with the
**same top items at similarity weight 2.0** — the maximum of the shifted cosine
(1 + cos = 2 ⇒ cos = 1). Four of eleven item prototypes are literally the same direction.
The naming pass faithfully reported duplicate prototypes; the "bug" was in the model.

## 2. How bad the collapse is (measured, not eyeballed)

Direct measurement on the raw checkpoint prototype matrices (no t-SNE, no naming route
in between), duplicates defined as pairwise cosine > 0.999:

| Model (side) | K nominal | Effective directions |
|---|---|---|
| `item_proto` (item) | 76 | **~2** — median pairwise cos = 1.000; one bundle holds 62/76 |
| `user_proto` (user) | 76 | **5** — tight clusters, genuine spread between them (median cos −0.11) |
| `user_item_proto` (user) | 76 | ~64 (healthiest) |
| `user_item_proto` (item) | 11 | 8 |

Convergent readings: `user_proto`'s 5 direction-clusters match its 5 unique prototype
names exactly — **the naming route is a faithful collapse detector**, while ordered
top-k item lists overstate distinctness (tiny angular differences reorder the list).
June and S0.7 runs show the same pattern: systematic, not run noise.

**Direction vs. norm (user question):** the model reads prototypes *only through the
shifted cosine*, so a prototype's norm affects nothing — it is a pure gauge freedom.
The only force on the norm is weight decay, and indeed all norms have decayed to
near-identical small values (`item_proto` 0.123–0.135, cv 1.6%). So the collapse is NOT
a norm illusion; the directions themselves coincide. Same L2-as-canonicalizer mechanism
as F-DC01-08, here on the prototype-norm axis. The t-SNE uses `metric='cosine'` (the
right geometry) but therefore **stacks duplicate directions on one dot — it understates
collapse**; instrument note for SC.8: prototype multiplicity needs explicit encoding
(dot size / count) before a t-SNE can honestly show degeneracy.

## 3. Why it happens (mechanism, code- and config-anchored)

Three things stack:

1. **The objective permits collapse for free.** The ranking loss consumes only the
   similarity vector; no term cares whether two prototypes are duplicates. Neither
   regularizer penalizes prototype–prototype similarity: `reg_proto` ('max') rewards
   each prototype's similarity to its best-matching batch item *independently*
   (`feature_extractors.py:495`) — 76 prototypes parked on one dense direction all
   collect full reward. Separation only ever arises *indirectly*, via coverage
   (`reg_batch`): duplicated prototypes waste coverage. This is exactly the F-DC01-05
   revised account from the dc01 SC.1b gate — now empirically confirmed by its absence.
2. **The winning config turned the separating force off.** Best single-sided config:
   `sim_proto_weight = 1.798` vs `sim_batch_weight = 0.0018` — anchoring ~1000× coverage.
   Anchoring onto an item direction yields shifted cosine → 2.0, precisely the measured
   top weights.
3. **Model selection is collapse-blind.** Hyperopt selects on val HR@10 only, and
   accuracy evidently doesn't need distinct prototypes here (a rank-5 representation
   carries `user_proto` to 0.5725 HR@10). TPE walks freely into the degenerate corner.
   Footnote: `user_proto` and `item_proto` selected the byte-identical config — the
   known fixed-seed HyperOptSearch startup-sharing.

Why the double-tie stays healthier (hypothesis-grade): its prototypes hold a second job
as the projection basis of the other side's linear half, so redundancy costs
representational rank that accuracy *does* see (consistent with the F-DC01-07
host-specificity insight); plus its item side happened to select small K=11 and a 5×
stronger coverage weight.

## 4. The popularity-bias identity (user hypothesis; confirmed with inverted mechanics)

The user asked whether the collapse has to do with popularity bias. The naive version —
prototypes collapsing *onto popular items* — is **refuted**: the dominant `item_proto`
bundle (62/76 prototypes) points at the **rare pole** of the item cloud
(Spearman(cos-to-bundle, train popularity) = **−0.76** over all 13,651 items; the top-50
bundle-aligned items have median popularity 6 vs catalog median 13; the anti-aligned end
is the head).

The true mechanics: `cos(i, p*)` is a learned monotone *unpopularity score*, and
**89.6% of the 73,418 users weight the bundle coordinates negatively** (mean summed
coefficient −0.73). Negative weight × unpopularity score = a global popularity boost
applied for nearly every user regardless of taste. **The bundle is a self-built item
popularity-bias term.** The kicker: the fleet runs `use_bias=0` (charter C3) — removing
the explicit bias term didn't remove the bias, the model *rebuilt it out of prototype
capacity*. The 62-fold duplication is coefficient amplification: per-coordinate user
weights are magnitude-bounded by weight decay (mean |coord| = 0.0127 identical inside
and outside the bundle), so duplicating the coordinate is how the model buys a large
total coefficient on its most valuable feature — and popularity calibration is the most
valuable feature under sampled softmax with popular negatives. Coheres with D4:
`item_proto` is the most popularity-skewed model (HR@10 0.914 on head items, ~0 on the
tail).

## 5. Punchlines (as spoken)

- "Accuracy-only model selection actively selects for whatever the interpretability
  machinery tolerates" — equal-accuracy configs range from 5 effective prototypes to 64,
  and nothing in the pipeline prefers the legible one. The Rashomon/knobs-and-metrics
  chapter's thesis in one measured example.
- "Removing the bias term didn't remove the bias — the model rebuilt it out of
  prototype capacity."
- ~80% of the "interpretable prototype" capacity is "a popularity knob wearing 62 name
  tags" — which the naming route dutifully labeled "Unknown, Unknown, Denim Other
  Garments."

## 6. Routing

- **Hidden-effects section** (thesis): selection-blindness + gauge freedoms + missing
  bias channel ⇒ interpretability capacity silently repurposed. Sibling of the dc01/dc05
  hidden-effects entries.
- **Rashomon/knobs chapter**: concrete degeneracy exhibit; effective-K vs nominal-K as a
  candidate interpretability metric; reg weights + K as the knobs that move it.
- **Charter C3 bias-on ablation**: this finding is candidate trigger evidence (vault
  2026-07-11_1556 reserved bias-on as evidence-triggered ablation). Decision pending,
  nothing queued.
- **SC.8 instruments**: pairwise prototype cosine / effective-direction count join the
  committed geometry set; t-SNE multiplicity encoding before comparison figures.
- Dossier record with all numbers: `s0_foundation.md` §S0.7 (commits `3d73189`,
  `f1bcd22`, `d3f713c`).

[[phase-4]] [[experiments]] [[explanations]] [[evaluation]]

> ⚠ **CORRECTION 2026-08-17:** parts of this entry are corrected in [[2026-08-17_1823_correction-double-tie-projection-basis-and-ui-effective-count]] — the double-tie "prototypes as projection basis" mechanism is code- and paper-refuted (prototype norms stay pure gauge at UI; only entity embeddings hold a second job), and the UI-user "~64 effective" figure is unreproducible (measured: 10 canonical / 8 old run).

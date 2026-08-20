---
date: 2026-07-14
time: "15:11"
phase: 4
tags: [thesis/fu-protomf, thesis/conclusion]
placed: 2026-08-20
---
# Balog 2019 as dc05's transparent ancestor; scrutability as a latent affordance of linear composition

*(From the S2 read, §3.7 p54, worked through interactively. Two claims Matteo flagged — "citable" and "scrutability applies to us" — both confirmed, each with a sharpening.)*

## The citation and why it matters specifically to dc05

Balog et al. (SIGIR 2019; S2 discusses it under rule mining but notes it generalizes) build a user model as a **weighted tag set aggregated over the items the user rated** — transparent by construction, no embeddings. That is structurally fU-ProtoMF's move: dc05's history-composed user representation (user = Σ history items' feature rows) is the *embedded* version of the same aggregation. Balog therefore belongs in **dc05's positioning** as the closest transparent ancestor of the history-composition — not just in a generic related-work pile. Reading-list entry: B11 (◦, cite-in-RW).

## Scrutability applies to us — via linearity, as an enabled-not-built affordance

Scrutability (Tintarev's user-facing aim; Balog's headline property) = the user can **correct** the system's belief about them and see immediate effect. Balog gets it because the preference model is an editable tag set. **fU inherits a latent version for free from linear composition:** the user vector is a linear sum of readable rows, so "remove this purchase from my profile" is literally subtracting a row — scores change immediately, no retraining, by construction. Same epistemic status as the route-1 projection: **enabled, not built** — the thesis may claim the affordance exists (and demonstrate it cheaply if desired), not that a scrutable interface was delivered. Caveats: the **ID row is the non-scrutable residual** (nothing user-readable to edit there), and per-row *weights* are whatever the composition defines — the edit affordance is row-level presence/absence, not fine-grained preference tuning.

## Terminology guard (thesis-writing directive)

**Scrutability ≠ Scrutiny Protocol.** Scrutability is a user-facing property of the recommender (user can inspect and correct their model); the Scrutiny Protocol is our internal audit-and-repair process. The words must never appear near each other in the thesis without the distinction being drawn — a reader meeting both unlabeled will conflate them.

[[decisions]] [[explanations]] [[paper-understanding]] [[phase-4]]

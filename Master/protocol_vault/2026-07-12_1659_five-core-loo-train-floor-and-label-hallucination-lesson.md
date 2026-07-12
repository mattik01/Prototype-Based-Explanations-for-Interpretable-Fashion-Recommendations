---
date: 2026-07-12
time: "16:59"
phase: 4
---
# 5-core + leave-one-out ⇒ every user has a 3-TRAIN-row floor — and the label-hallucination lesson

*From the dc05 concept review sitting. Two things live here: the exact split
fact (user-checked framing, confirmed), and the anatomy of the wrong label
that triggered the check — kept per the verbatim rule because the lesson is
process-critical.*

## The fact (correct framing)

V1 `hm_1_month` is **5-core** (verified 2026-07-12 against both the artifact —
minimum total interactions per user AND per item is exactly 5 — and
`hm_dataset_notes.md`; the splitter CLI default of 10 was overridden at V1
generation). The k-core counts **deduped distinct (customer, article) pairs**,
and the counts include the rows that later become val/test. Leave-one-out then
takes each user's last purchase as test and second-to-last as validation —
exactly 2 rows per user. Therefore:

> **Every user has a floor of 3 TRAIN interactions** (5 total − 1 val − 1 test).
> The total floor is 5; "floor of 3" is only correct with the word "train".

Measured on V1: 18,442 users (25.1%) sit exactly on that 3-train-row floor;
median train history 5, p90 = 12. For dc05 this floor is load-bearing: the
minimum basket a composition ever sees is 3 purchases.

**Asymmetry note (extended same sitting — user clarification: the k-core
intuition "no cold items either" is wrong, and the direction surprised):** the
floor logic is user-side only, because the k-core counts are computed BEFORE
the split and the leave-one-out split is **user-anchored** (it takes each
USER's last + second-to-last purchases; items have no say). An item whose ≥5
interactions all happen to be someone's last purchases lands entirely in
val/test — a truly cold item scored on its random-init embedding in the
standard eval. Measured (S0.1/F-S0-04): V1 has 3 such items (9/73,418 test
rows ≈ 0.012%, negligible), hm_3_month 23, **amazon2014 101 → 7.1% of test
rows (material, disclosed for replication readings)**. A train-unseen USER is
impossible by construction. Thesis statement: *neither side contains
deliberate cold entities; truly cold users are impossible, truly cold items
are possible and marginally present — the opposite asymmetry from what k-core
intuition suggests. Deliberately cold items exist only in the constructed
S0.3 variant; deliberately cold users exist nowhere yet.*

## The lesson (how the wrong label happened — user challenge, answered)

The dc05 doc briefly called this "the 3-item k-core floor" — asserting core=3.
Anatomy of the error: the NUMBER was real (floor of 3, measured by script);
the LABEL was fabricated by fusing it with a remembered half-fact (S0.2's
"every evaluated user has ≥3 train interactions"), which *seemed* to confirm
it — a remembered sentence acting as false confirmation for a fresh
measurement. Nothing downstream computed wrongly; the review caught it before
anything was built on it.

**Sharpened rule (added to the scrutiny learnings ledger, [fU-cycle]):**
hallucination enters through *labels*, not numbers. Any sentence naming a
mechanism or parameter (core, seed, fraction, threshold) is a preprocessing
claim and must be verified against code/artifact — even when it decorates a
correctly measured number, and especially when a remembered document seems to
confirm it. The existing "verify against splitter code, never against other
docs" watch-out gates on the sentence's SHAPE (names a parameter), not on
whether the fact feels important.

[[phase-4]] [[data]] [[evaluation]] [[decisions]]

---
date: 2026-07-12
time: "18:40"
phase: 4
---
# dc05 concept review CLOSED — fU-ProtoMF approved for build

The dc05 design cycle (run earlier today under the fU brief M1–M7) and its
user review closed the same day. The candidate — **history-composed user
factors on the U-ProtoMF host** (q_u = mean of train-basket attribute words +
user-ID row; `ft_type feature_user_proto`) — is **reviewed, awaiting build**.

**Ratified at review:** the mechanism incl. mean normalization (sum = mean at
fixed width explains fI's convention; mean = the only exponent whose ID
counterweight doesn't fade with history); both configs
(`feature_user_proto` / `_noid`); naming; renderer slot with spec defaults
(top-6 bars, per-purchase zoom default, truncated purchase labels, host
post-hoc zoom as steel-man, B(t) line).

**Added during review (dated amendments + vault entries):** zero-interaction
directive (1653); 5-core/3-train-row floor + label-hallucination lesson +
cold-entity asymmetry (1659); information ordering (1713); ids-arm dominance
theorem + thesis placement (1739); B(t) implicit per-item intercept — full
bundle (1808) + UI/merge extension & "check which side is free" (1815); M5′
noise-for-crowding + ids division-of-labor + K6′ centering option (1830).

**Deferred/scoped:** `lightfm_hist` charter request → decided once runs are
in; stationarity → deliberately out of scope; gauge/lockstep → dc01
discipline unchanged.

**Next (separate sessions):** build (SC.3-style on `feat/scrutiny`; keystone
bit-identity vs `user_proto`; renderer/cards/naming slots; new build
obligations from the review: A1 term-by-term objective-equality check,
variant-train weight-rebuild test, cold-user ID-drop mirror) → then SC(dc05)
scrutiny. In parallel, unchanged: S0.7 part 2 → dc01 SC.6 on cluster progress.

[[phase-4]] [[decisions]]

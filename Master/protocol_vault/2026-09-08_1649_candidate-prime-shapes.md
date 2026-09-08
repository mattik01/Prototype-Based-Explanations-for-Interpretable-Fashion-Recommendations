---
date: 2026-09-08
time: "16:49"
phase: 4
---
# Candidate prime (user attributes on the fU side): the three shapes, their math, and the precondition

Discussion 2026-09-08 (Matteo's question: can fU′ be the averaged interaction vector plus an
appended vector built by summing user-attribute words). Answer: yes, and it is lineage step 7
("candidate prime", vault 2026-07-11_2351: demographics enter last, weakest, only where nothing
else exists). Recorded here: the shape question and the ramifications. No decision to build yet.

**Shape 1 — one sum, one space.** q_u = mean of history words + Σ user-attribute rows + e_ID,
one cosine, one prototype set. Exact attribution, intrinsic naming (prototypes nameable by
attribute values too). Two defects: with e_ID present the attribute sum is a fixed function of
the user and is absorbable (inductive bias only); and the norm handover runs the wrong way —
the history mean shrinks with |H| while the attribute sum has a fixed norm, so demographics
take over the cosine for HEAVY users, the opposite of a fallback. Fixable by scaling the
attribute sum by 1/(|H|+1): then it is shrinkage toward a *demographic-cell prior* (a few dozen
directions, less collapse than shrinkage toward the single population mean).

**Shape 2 — two prototype layers, added (preferred).** score = Σ_l t_il(1+cos(m_u+e_ID, p_l))
+ Σ_k s_ik(1+cos(a_u, q_k)); the structure user_item_proto already uses. Each cosine normalises
its own block (no handover), each prototype set named from its own vocabulary, attribution =
two exact sums side by side. The second term has at most a few dozen distinct profiles (hm
usable fields: age band, club status, news frequency, two activity flags; postal code EXCLUDED
as identity-like) — it is a segment-popularity model with a prototype bottleneck, visibly a
fallback, and its score share is one readable number. Explanation sentence "shoppers in your
group tend to buy this" is honest under the two-part rule (membership yours, strength the
population's). Cold items: s_i untrained → term 0 until the merge composes it from item words,
at which point the layer becomes user-attributes × item-attributes through a small prototype
set (LightFM cross term with a bottleneck), cold-capable on both sides.

**Shape 3 — concatenation [m_u ; a_u], one cosine over the long vector (M14).** Legitimate:
blocks are orthogonal by construction so a word row can never point in a demographic direction
and each prototype has two separately readable halves. But the norm couples the blocks (same
handover as shape 1). Fix: normalise each half to unit norm, then weight the demographic half by
1/(|H|+1); guard the empty-history zero vector. Result: one prototype set with joint patterns
("black basics AND age 25–34"), one explanation object, most compact form. Costs vs shape 2: the
stereotype share is per prototype and per user (hard to summarise), and in the double-tie merge
the user vector's length must equal the number of item prototypes, so a 2d user vector forces
2d item prototypes whose demographic half has no natural item-side meaning. Shape 2 leaves the
tie alone.

**Ramifications common to all shapes.**
- Does nothing for the prototype-layer bottleneck; any gain arrives through a path beside the
  word prototypes and must be read out as a score share.
- On the warm hm split the expected gain is ≈0: five purchases in five fields already reveal
  ladies/men/kids, which is most of what age carries. The dc05 Seed-2 rejection stands for the
  warm headline. The one honest use is COLD USERS, unmeasurable today → a cold-user split
  (hold out users entirely, evaluate with empty history) is an S0-level precondition; on it
  fU-noid gives a zero vector, fU-ids an untrained row, prime a segment ranking — the whole story.
- Not hm-only: ml-1m ships gender/age/occupation (the ProtoMF §5.3 audit field). Position:
  age and membership fields only, gender excluded and stated as excluded, §5.3 audit kept as a
  read-out. This touches the parked fairness block deliberately and minimally.
- A third family (item-ID history rows, M1) is possible but two layers/objects is the maximum
  a shopper should see.

**Ordering.** Finish the fU 100-trial wave (post leave-one-out); design the cold-user split
as scrutiny work; then prime as its own /design-candidate cycle in shape 2, both datasets,
after or alongside the merge. Not now — not because the idea is weak, but because there is
nothing to measure it against yet.

[[decisions]] [[explanations]] [[phase-4]] [[evaluation]]

---
date: 2026-09-08
time: "17:05"
phase: 4
---
# Query-time context enters at the double tie; past-event attributes cannot — and what the tie actually shares

Verbatim-capture entry (discussion 2026-09-08, Matteo's question: could interaction attributes
enter "at the double tie" instead of alongside the user history).

**What the double tie is (from the code, `feature_extractor_factories.py` 'prototypes_double_tie').**
Each side has ONE base embedding table read in TWO roles. User: (a) through the user prototype
layer → similarities to the K_u user prototypes; (b) through an `EmbeddingW` — the SAME table
(one assignment: `user_embed.embedding_layer.weight = user_proto.embedding_ext.embedding_layer.weight`)
followed by a free linear projection R^d → R^{K_i} → coefficients, one per item prototype. Item
side mirrored. Score = ⟨user sims, item coefficients⟩ + ⟨user coefficients, item sims⟩. So the
"shared weighting table" is the per-side base embedding shared between its two roles. It is NOT
shared between users and items; the projections are not shared with anything; there is no table
both sides read. Consequence: anything added to a side's base vector is read in both roles for
free — directly (cosine against named prototypes) in the prototype role, and only through the
learned projection in the coefficient role (the dc01 honesty-inventory asymmetry: the
coefficient half is attributable but points at units named by the other side).

**Two different things "interaction attributes at the tie" could mean; opposite answers.**
1. *Attributes of PAST events* (price paid, channel, date of a purchase already made): cannot
   enter at the tie. The tie is evaluated for (user, candidate); the candidate has no event with
   the user — no date, no price, no channel. Past-event attributes exist only for the history and
   the history lives on the user side. The taxonomy entry (2026-09-08_1650) stands unchanged.
2. *Context of the event being PREDICTED* ("it is December, you are browsing online"): an
   attribute of the query, neither user nor item — and the tie is the only point where user and
   item are in the same room, so it is the right and only place. Context-aware recommendation
   (factorization-machine form): context = extra rows added to the user's base vector at query
   time: u_c = history mean + e_ID + r_December + r_online. Both tie roles see it (the prototype
   profile shifts with the season — "in winter you resemble the outerwear community more" — and
   the coefficients on item prototypes shift). Attribution exact (linear sum), naming intrinsic
   (context words are rows in the same table).

**Why query context is special.** Every attribute discussed so far is a fixed function of the
user's own history and is absorbable by e_ID in function class. A context row is NOT: it varies
across the same user's events (March ≠ December for the same person), so no per-user row can
reproduce it. Query context is the one attribute family that escapes the absorption argument
WITHOUT making the profile per-candidate: the profile becomes per (user, situation) — "your
winter profile" — still an object that belongs to the user and can be shown.

**Second form — context sets the history weights.** Purchases in the same season / same channel
as the query count more: a per-event multiplier that depends on the relation between a past
event and the present one. This is the honest home for timestamps: not a property of a purchase
but the distance between a purchase and now. Exact attribution, nameable weight ("counted double:
also a winter purchase"), not ID-absorbable (varies with the query).

**Third form — rejected.** A three-way product (user sim × item coefficient × per-prototype
context factor; tensor factorization): works, but the context factor per prototype has no
vocabulary row → post-hoc naming.

**Ramifications (heavy).**
- Evaluation changes: the scorer gets the held-out event's date/channel and scores negatives
  under the same context (standard in CARS) — but every frozen S0 row was evaluated without it;
  either every row is re-evaluated with context or the model is compared against nothing. S0-level.
- Current data has almost no context: hm_1_month is one month → season constant, weekday noise;
  channel (store/online) is the only varying field and its predictiveness is unmeasured.
  hm_3_month/full would have real seasonal variation; ml-1m has years of timestamps but no
  channel/season.
- Norm handover again (fixed-norm context rows vs shrinking history mean) → block normalisation.
- Training needs the positive's context → dataloader change (positive's date/channel travel with
  the batch; negatives inherit it); interacts with the leave-one-out path (the user extractor
  would receive context alongside the positive index).
- Merge: context rows on the composed user base vector pass through both tie roles with no new
  machinery — the merge is the natural host. Symmetric item-side context rows ("a December item")
  are equally legal.

**Verdict.** Past-event attributes: wrong place, they are history. Query-time context: right
place, the only place, principled — but needs a dataset where context varies and an evaluation
protocol change touching every frozen row. Same drawer as the topic layer: part two / future work,
hosted by the merge. Sources (Rendle FM with context; Karatzoglou et al. Multiverse tensor
factorization 2010; Adomavicius & Tuzhilin CARS chapter; Baltrunas CAMF) cited from memory —
`paper-search` was down; VERIFY before any thesis use.

[[decisions]] [[explanations]] [[phase-4]]

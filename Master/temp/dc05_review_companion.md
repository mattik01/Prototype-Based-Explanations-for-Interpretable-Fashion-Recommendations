# dc05 in short — review companion

> Discussion-only distillation of `dc05_history_composed_user_factors.md`.
> No history, no protocol trail. The long file stays the source of truth.

## 1. What the model is

Host: **U-ProtoMF** (`user_proto`) — user prototypes, free item vectors, score
= item vector · user's prototype-similarity vector. ONE change:

> **Your user vector is no longer a free learned embedding. It is the average
> of the attribute words of the items you bought (train window), plus your
> personal ID row.**

- Vocabulary: the same 426 item-attribute values dc01 uses (department,
  product type, section, colour, pattern). No demographics anywhere.
- "Average": each purchase contributes equally; every user carries the same
  total feature mass regardless of history length. The ID row keeps a fixed
  relative weight next to it.
- Everything else (prototype layer, regularizers, loss, item side) is the
  host, byte-identical. With zero words + ID row the model IS `user_proto`
  exactly (verified bit-identical at toy scale).

## 2. What you get from it

**User prototypes become nameable taste communities, from parameters alone:**
each user prototype gets a profile over the 426 item words ("dark-denim
menswear community"). Today the paper interprets user prototypes by feeding
the model imaginary synthetic users — pure post-hoc.

**Two exact explanation zooms (same sum, two groupings):**
- by attribute word: "your activation of community 7: department Trousers
  Denim +0.29, section Menswear +0.07, …"
- **by purchase (new):** "you belong to community 7 because of these 12
  dark-denim purchases (+0.31 of 0.42)" — every purchase gets an exact
  additive share. This is the per-purchase attribution you asked for at the
  SC.1b gate.

**Thin users get a real representation:** a 3-purchase user is represented by
well-trained shared words instead of a barely-trained free vector. This is
the candidate's central accuracy bet (LightFM saw exactly this effect with
user metadata on its sparse dataset).

**Cold users, honestly:** an interaction-free user is NOT rescued (empty
basket → the model falls back to a non-personalized ranking — verified). But
one purchase in, the representation exists without retraining. No cold-user
eval exists in our split; nothing cold-user is promised this cycle.

## 3. The one surprise of the cycle — the B(t) baseline

The score is S = Σ_l t_l·(1 + cos_l) = **B(t) + personalized part**, with
B(t) = Σ_l t_l a per-ITEM scalar.

In fI the analogous "+1 mass" was the same for every item a user ranks —
harmless. **Here it is different per item, so it actually moves rankings: a
hidden, non-personalized item-bias/popularity channel inside our "bias-free"
fleet.** It exists in plain `user_proto` too (nobody separated it before,
including the paper). Consequences:

- Explanations honestly split into "item baseline (same for every user): +x"
  plus the personalized, fully decomposed part.
- How much of the ranking signal training routes into B(t) is UNKNOWN — if it
  absorbs a lot, the explanation narrates a minority of the score. This is
  the biggest single risk; measuring its share is committed as a primary
  scrutiny metric (for fU AND the host).

## 4. Numbers worth knowing (V1, measured this session)

- Median train history: **5 purchases**; 25.1% of users sit at the 3-train-row
  floor (V1 is 5-core; leave-one-out removes 1 val + 1 test row, so minimum
  train history = 3); p90 = 12. The "thin user" stratum is a quarter of the
  population — but the thin/heavy contrast is narrow (fallback: hm_3_month).
- Only **0.61% of users share an identical normalized basket** (fI's item
  twins: 67.5%) — losing the ID row costs almost no *exact* individuation.
  (How much angular individuation survives in geometry is a scrutiny
  instrument, not settled.)
- Cost: +27K parameters (~0.5%); ~1.05–1.2× step time; no feasibility flags.

## 5. Main risks (all flagged, none resolved — they are the empirical bets)

1. **B(t) dominance** (above) — explanation covers only the personalized part.
2. **Heavy-basket mean-regression:** averaging many purchases pulls heavy
   buyers toward the "average customer" direction (toy-demonstrated) —
   personalization may flatten exactly where the most data is. The ID row is
   the counterweight; the normalization choice (mean vs √ vs sum) is the knob
   — a mechanism decision that is yours.
3. **Diffuse/overlapping community profiles** — same unenforced-grounding risk
   as dc01, same instruments; optional penalty knobs designed, none stacked.
4. **ID-row routing** — signal can drain into the personal ID rows, shrinking
   the feature-explained fraction; measured by the ids/noid ablation pair.
5. **Win attribution:** beating `user_proto` doesn't yet separate "history
   features help" from "prototype layer helps" — a no-prototype control row
   (`lightfm_hist`) is formally requested as a charter amendment; decision
   yours at the scrutiny stage.

## 6. What was considered and rejected (one line each)

- **Demographic mirror** (age/club/news composition): ~dozens of user classes
  for 73k users, "because you are 34" explanations — rejected; stays material
  for the later "candidate prime".
- **dc04-style user-prototype anchoring / dc02-style count bottleneck:**
  ground the prototypes but not the user representation, or delete all user
  capacity — both kept as mapped quarry, not taken.

## 7. Decisions on your table at this review

1. Accept the mechanism (mean-composition + ID row) as specced, or change the
   normalization now.
2. Bless `feature_user_proto` / `_noid` as the two fixed configs.
3. Note the two standing requests that surface later: the `lightfm_hist`
   control row (charter amendment) and the B(t)-share metric as primary.
4. Anything you want renamed/reframed before the build session.

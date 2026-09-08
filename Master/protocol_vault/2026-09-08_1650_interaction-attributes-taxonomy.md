---
date: 2026-09-08
time: "16:50"
phase: 4
---
# Interaction attributes and the per-event sum: a taxonomy of what each kind means and how the fU shapes reduce it away

Verbatim-capture entry (worked through 2026-09-08 at Matteo's request, following the candidate-prime
shape discussion). Written so the argument can be regenerated without the conversation.

**What an interaction attribute is.** An item attribute describes the thing; a user attribute
describes the person; an interaction attribute describes the EVENT where the two met — it happened
at a moment, in a way, possibly alongside other events, and it may carry a value. hm raw fields:
date, price paid, sales channel (store/online); derivable: discount vs the article's usual price,
weekday, season, repeat purchase, gap to previous purchase, same-day count (order proxy). ml-1m:
timestamp and rating (the one signed interaction attribute).

**Four kinds, by what they are really about.**
1. *Acquisition* (price paid, discount, channel): belong to the (item, event) pair; meaning is
   bound to the item they came with — a discount on a dress and on socks are different facts.
2. *User state at the moment* (date → season, mood, need; position in own history): belong to the
   event alone but are RELATIVE — a date means nothing except relative to now, to the target's
   season, or to the other purchases.
3. *Relations between events* (same day/order, bought-after, time gap, repeat): attributes of
   PAIRS of events or of the sequence, not of any single event.
4. *Valence / intensity* (rating; quantity, repeat count): the only kind that says how the user
   felt about the item rather than the circumstances.

**The structural fact.** Every fU shape (single sum, concatenation, two layers) is a per-event
sum: permutation-invariant by construction (Deep Sets: ρ(Σφ)). Exactly three ways an
interaction attribute can enter a per-event sum:
- *As a multiplier on the event* (w_j = f(event attributes)): per-event weights change the
  DIRECTION of q_u, so the cosine sees them (unlike per-user scalars, which it normalises away);
  attribution exact; the weight's reason is nameable ("counted double: bought last week") so the
  attention critique does not apply. Cost: total reduction of meaning to "how much does this
  purchase count". Only recency and valence have a meaning that IS honestly a weight. With e_ID
  present every such weighting is a fixed function of the user's row → absorbable → testable
  only in the noid arm.
- *As extra words attached to the event* ("online", "discounted", "winter"): summed like item
  words. Two failures: the event word is treated as taste with no item-side counterpart; and the
  count bottleneck strips the BINDING — you know 3 things were bought online and 2 were black,
  not whether the black ones were the online ones. Repair = conjunction tokens
  (black∧discounted), same vocabulary explosion as the pair research found.
- *As a modulation of the item's composition* (event words multiply item words elementwise):
  a bilinear, FM-style term; keeps the binding, exact per (word, event-word) pair; a new object
  with many units, and still absorbable by e_ID.
None of the three can see RELATIONS between events. Options for relations: pair units (O(|H|²),
no support at 5 purchases); grouping event → basket → user (collapses to a re-weighting unless
the basket level gets its own row or a nonlinearity, which breaks the contract the same way
pairs do); or sequence models (attribution dies).

**Time specifically.** Reference "now" (end of train window) → time collapses to recency decay,
a per-event weight — the one temporal attribute that fits the multiplier form honestly.
Reference the TARGET (purchase season vs candidate season) → the weight depends on the candidate,
the user vector differs per candidate (target-conditioned attention family; the user profile
stops being the user's object). Reference the other purchases → relational; the sum is blind.

**What the shapes do to each kind, plainly.** Acquisition → a counting weight (meaning destroyed)
or a free-floating taste word (detached from its item). State → recency (honest) or season words
(no counterpart, no reference). Relations → nothing. Valence → a signed weight, which IS its
correct meaning: the Rocchio form, positive rows for liked, negative for disliked — the one
interaction attribute the sum handles well (ml-1m only).

**What would be necessary mathematically.** Binding: pair units or bilinear modulation.
Relations: pair units over events or a nonlinear basket level. Target-relative time: a
candidate-dependent weight. Each either leaves the additive contract or multiplies the units
past legibility, and each is a fixed function of the user's own history, hence ID-absorbable in
the shipping arm. This is the structural reason the 2026-09-07 research brief excluded event
context: the additive sum with a cosine on top has no honest slot for it except as a weight.

**Dataset facts that soften it for hm.** 94.3 % of purchases share their date with another by the
same user; median 1.73 purchase days → for most users the history IS one or two orders, basket
level = user level, relational structure degenerate. Within a one-month window recency is nearly
flat. So on hm_1_month the loss from ignoring interaction attributes is small. On ml-1m (56 films
over years) drift and sequence are real costs.

**Consequence for fU′ / prime.** Interaction attributes do not fit any prime shape as
vocabulary because their meaning is bound to an event, not the user. They fit all shapes only as
per-event multipliers inside the history block (recency; valence where it exists). Price band
belongs on the ITEM side (it is an item attribute; deferred there already). Channel, season,
basket and sequence are named out of the contract with the reason above. Anything relational is
a different model class → future work beside the topic-layer idea.

[[explanations]] [[decisions]] [[phase-4]] [[hidden-effects]]

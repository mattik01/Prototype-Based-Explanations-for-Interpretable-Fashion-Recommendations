---
date: 2026-08-24
time: "12:29"
phase: 4
---
# The bag-size channel on ml-1m: a popularity prior smuggled through composition geometry

Verbatim-capture entry (worked through interactively 2026-08-24, right after the U2a
instrument found corr(tag-bag size, ‖q_meta‖) = 0.80 on fI ml-1m / 0.58 noid). Written
to regenerate the full argument without the conversation in context.

**The measurement:** on ml-1m (bags layout: genres + Tag-Genome tags, mean ~12 tokens
per movie, everything summed at weight 1.0), the number of tags a movie has correlates
0.80 with the length of its composed metadata vector. Question asked: is that a
problem — what are the implications?

**Layer 1 — the reassuring part: cosine shields us from the crude version.** In
LightFM-style dot-product scoring, a longer item vector means bigger scores, full
stop — so "more tags → longer vector" would be a direct score boost for heavily-tagged
movies. Our host scores with the (shifted) cosine, which throws the item's length away.
The 0.80 correlation between tag count and vector *length* does not, by itself, touch a
single ranking. If that were the whole story it would be a non-issue.

**Layer 2 — but length isn't the whole story; bag size bends DIRECTION too.** A movie
with 40 tags has a direction that is the average of 40 token directions, and averages
of many things drift toward the middle — the common direction of the cloud. A movie
with 3 tokens points wherever those 3 tokens point. So many-tag movies become
geometrically *generic*, few-tag movies stay *sharp*. To the extent the prototypes lean
toward the common direction (the collapse story), generic heavily-tagged items get
mildly elevated similarity to many prototypes at once. And tag count is almost
certainly a popularity proxy — popular movies accumulate Tag-Genome tags. Punchline:
**a popularity prior smuggled in through composition geometry** — no bias term, no
popularity feature, just "popular movies are described more, and more-described items
sit closer to the center of the space." Exactly the shape of thing the hidden-effects
chapter exists to name: invisible in the architecture diagram, visible only in the
geometry.

**Layer 3 — it is the item-side twin of a story the thesis already tells.** The dc05
length-channel narrative (user history length as a hidden channel; tax-and-refund
hypothesis, vault 2026-07-12_1224) is the same phenomenon on the user side. Bag size is
its ml-1m item-side sibling. Narratively a gift, not a problem: "the same structural
channel appears wherever a variable-size set is summed," two independent instances.

**Where it concretely bites — three places:**
1. **Explanation comparability.** Per-word shares of a 40-tag movie are spread ~40
   ways; a 3-token movie splits three ways. A share of 0.08 means something completely
   different on the two items, and the ID-share meter is confounded the same way (big
   metadata sum → ID looks proportionally smaller). Not wrong numbers — but not
   comparable across items without disclosing bag size. Cheap render-time fix: put the
   bag size on the explanation card.
2. **Cross-dataset readings.** H&M structurally cannot have this channel (always
   exactly 5 words, fixed layout). Comparing fI behaviour across hm and ml-1m —
   including when hunting the −0.0555 ml-1m regression — partly reflects this
   structural layout difference, not the candidate. Every cross-dataset claim needs
   the caveat.
3. **Cross-model comparisons on ml-1m.** If the LightFM baselines score with a dot
   product, bag size IS a live score channel for them while cosine mutes it for us —
   an apples-to-oranges asymmetry hiding inside "fI vs lightfm_tags on ml-1m."
   (Open check: verify how our lightfm rows actually score before leaning on that
   comparison.)

**Why NOT to intervene in the model:** mean-pooling instead of sum-pooling — the
textbook "fix" — changes nothing here, because dividing by bag size is a per-item
scalar and cosine ignores it. The real object is the directional drift; per the
hidden-effects philosophy the right move is measure-and-disclose, not redesign.

**The three cheap correlations that decide if the channel is ACTIVE (designed, not
run):** (a) bag size vs train popularity — is tag count really a popularity proxy on
our split; (b) bag size vs mean activation level — do big-bag movies really sit closer
to the prototypes; (c) bag size vs the ID-share meter — quantify confound (1). Hot →
the channel graduates to a named hidden effect with numbers; cold → a disclosed
structural fact.

Bottom line: not a bug to fix, but a channel to name — mostly defused by the cosine,
alive in the geometry and in explanation comparability, and a ready-made second
instance of the length-channel story.

[[hidden-effects]] [[explanations]] [[evaluation]] [[phase-4]]

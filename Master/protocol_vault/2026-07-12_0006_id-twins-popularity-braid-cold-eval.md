---
date: 2026-07-12
time: "00:06"
phase: 4
---
# The ID/twins/popularity braid under the cold-eval lens — one arrow, two losses, two evals, one hypothesis

*(Verbatim-capture entry from the dc01 SC.1b gate walkthrough (Point 5 discussion, night of 2026-07-11→12). User: "this is a relevant testing phase — insights we absolutely should report." Reporting-phase material for SC.8 and the thesis.)*

## The braid: three questions that are one

The ID drop at cold inference, the signature twins, and the popularity/bias terms are not three separate questions — **the ID row is the strand that ties them, because the ID row is where popularity and twin-separation both live**:

- Under our bias-free fleet, item popularity has nowhere to live *except* the ID rows (no bias scalars, and feature rows are shared across items — they can only carry class-level signal).
- The ID row is also the *only* thing separating signature twins (items with identical 5-field attribute tuples are otherwise the same sum of arrows).

One arrow carries both: "how popular this exact item is" and "which of the identical-looking items this is."

## Drop the arrow, and watch what each eval sees

- **Cold-vs-cold (the headline):** every candidate in the 1+99 slate loses the same arrow, so *both* losses cancel symmetrically. Popularity: nobody has it — the popularity reference row duly sits at the chance floor (HR@10 = 0.10). Twins: everyone's twins tie — and the S0.5 measurement showed this barely touches the sampled metric (~0.06 same-signature competitors per 99-negative row); the honest display of the full-catalog ceiling is the tie-block diagnostic. Clean isolation, exactly as the B5-faithful design intended.
- **Cold-vs-all (the diagnostic):** the cold positive is missing its arrow while its ~99 warm competitors keep theirs. **Key insight: the adversary's Point 5 (projection-half magnitude handicap from the missing ID row) and Point 2 (SGD routes popularity into ID rows) are the same phenomenon seen twice** — the warm items' ID-row advantage *is largely their popularity*.

## The reframing that matters for the thesis

The cold-vs-all "handicap" is only *partly* a measurement artifact — **partly it is reality**. A genuinely new item really does lack popularity evidence; a deployed system really would rank it against incumbents that have it. So the cold-vs-all gap is not noise to apologize for: it **decomposes into a real-world effect (new items lack standing) plus any artifact on top**, and the popularity-decile slices of the cold results (built into the cold evaluator) are the tool to tell the two apart. Interpretation rule adopted for SC.8.

## Where the bias-gap note joins the braid

LightFM's feature-composed biases (b_i = Σ_f β_f) would have given cold items a *class-level* popularity prior ("denim trousers sell well") even with zero item history — precisely the cheap fix `dc01_feature_composed_bias_gap.md` sketched and the fleet deliberately left unstacked (bias-free parity, S0.4 gate). **If SC.8 shows the cold-vs-all gap is dominated by the popularity component, that unstacked knob becomes the evidence-triggered ablation the S0.4 gate already anticipated.**

## Registered hypothesis (falls out for free, checkable at SC.8)

**The warm feature-explained fraction should inversely predict the `_ids` config's cold-vs-cold → cold-vs-all degradation.** Reasoning: the feature-explained fraction (the honesty meter of the ID-residual discussion, [[2026-07-11_2132_dc01-id-residual-three-roles]]) measures how much of the model's scoring flows through IDs; everything routed into IDs is lost at the drop; so how much a model *leaned* on IDs and how much it *loses* against warm incumbents should be **two views of one number** — the explanation-quality metric and the cold-robustness metric coinciding. Cheap to check (both metrics already committed); lovely if it holds; informative either way.

## The twins strand, resolved (Point 6, same walkthrough)

The adversary's tie-artifact attack ("twin ties make the no-ID metric an artifact of sort order; the tuple count was never run") resolves with numbers it couldn't see: the count ran at S0.5 — **6,517 distinct signatures / 13,651 items, 67.5% share theirs, largest class 97** (squarely inside the adversary's guessed "~6–8k" kill zone — the concern was right to raise, and is now quantified). But the kill shot misses the *metric*: the ceiling is a **full-catalog** phenomenon and our eval is 1+99 *sampled* — measured ~**0.06 expected same-signature competitors per row** (~94% of rows contain no twin of the positive), bounding sort-order effects far below a percentage point. Ties resolve deterministically through the ranking internals (known convention, learnings ledger); adopted: a *disclosure line*, not a tie-handling mechanism. The full-catalog ceiling's honest display is the tie-block diagnostic. Scoping accepted: the warm-item ceiling is a property of the no-ID *ablation arm*, not the headline config. **Braid closure: twins are the extreme end of the F-DC01-06 identifiability continuum — a twin pair is a value clique with zero decoupling mass, where not just the split but the entire distinction is untrained. One framework covers both findings.**

**Testing around this is warranted (user directive):** don't leave the bounds theoretical — at SC.8, count actual tie blocks in the real eval rows of the no-ID run and report the measured (not just expected) incidence; run the tie-block diagnostic for every model row on the variant; spot-check twin pairs on the real checkpoint (do twins in fact tie bit-exactly end-to-end; does the ID arm separate them and by how much — which doubles as a direct read of the ID row's twin-separation role in the braid). The SC.2 toy exhibit (lockstep pair, multiple inits) covers the mechanism side.

Cross-refs: dossier `sc01_feature_item_proto.md` SC.1b fold-in (registered-hypothesis block); [[2026-07-11_2350_interactions-explained-by-item-features-taste-revealed]] (the taste argument this braid presupposes).

[[decisions]] [[evaluation]] [[explanations]] [[experiments]] [[phase-4]]

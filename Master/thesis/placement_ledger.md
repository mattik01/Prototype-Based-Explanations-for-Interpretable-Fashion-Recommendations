# Protocol-Entry Placement Ledger

Maps every protocol-vault entry to the thesis section(s) it feeds. Maintained by the placement skill; seeded 2026-07-13 from a full vault read (78 entries); slugs reconciled 2026-07-14 after the Ch2+3 merge and Part II three-way split; first physical placement (\vault lines) executed 2026-07-14. **Placements are proposals until Matteo confirms them in a writing session.**

**Statuses:** `placed` (≥1 slug) · `not-thesis` (process/infrastructure — deliberately unplaced, never forced) · `deferred` (thesis-relevant but no ripe home yet). Slugs refer to `outline.md`. An entry may feed multiple sections. New vault entries get appended with status `new` until triaged.

| Protocol entry | Placement(s) | Status | Note |
|---|---|---|---|
| 2026-03-08_1030_folder-structure-creation | — | not-thesis | repo scaffolding |
| 2026-03-08_1045_claude-md-update | — | not-thesis | conventions |
| 2026-03-08_1100_protocol-vault-setup | — | not-thesis | tooling (feeds LLM appendix only as artifact) |
| 2026-03-08_1112_protocol-system-test | — | not-thesis | test entry |
| 2026-03-08_1725_phase-1-5-complete | — | not-thesis | tooling |
| 2026-03-09_1641_phase2-data-config | `eval-framework` §design | placed | lfm2b-unavailable fact only; rest process |
| 2026-03-12_1640_phase2-6-paper-reread-complete | `background` §protomf | placed | s^user equivalence + post-hoc naming seeds; entry itself is process |
| 2026-03-18_1234_double-tie-explainability-tradeoff | `background` §tie, `fufi-merge` | placed | |
| 2026-06-07_2306_replication-complete | `eval-framework` §design | placed | numbers thesis-relevant, bookkeeping not |
| 2026-06-07_2307_kaggle-dataset-usage-insight | `data` §splits | placed | short-window precedent |
| 2026-06-07_2308_map12-noncomparable-focus-shift | `intro`, `eval-framework` §design | placed | framing move |
| 2026-06-07_2309_eval-scope-lightweight-map12 | `eval-framework` §design | placed | |
| 2026-06-07_2310_compute-feasibility-envelope | — | not-thesis | infra constraint |
| 2026-06-07_2311_why-hm-sparse-feature-rich | `intro`, `data` §splits | placed | |
| 2026-06-07_2312_hm-two-version-dataset-strategy | `data` §splits | placed | |
| 2026-06-08_1149_prototype-naming-and-similarity-routes | `background` §iml-intrinsic, `fi-protomf` §explanations | placed | post-hoc naming yardstick the intrinsic method replaces |
| 2026-06-11_0843_hm-feature-analysis-dataset-suitability | `data` §splits, §splits | placed | |
| 2026-06-11_0905_r7-user-perceivable-feature-vocabulary | `data` §splits, `interpretability-tuning` §channels (UW) | placed | R7 |
| 2026-06-11_1111_protomf-within-interpretable-ai | `background` §iml | placed | structural decision itself |
| 2026-06-11_1144_proto-regularizers-interpretability-not-accuracy | `background` §reg, `interpretability-objective` §motivation | placed | block-two seed |
| 2026-06-11_1220_posthoc-naming-vs-intrinsic-model-distinction | `background` §iml-intrinsic | placed | key nuance |
| 2026-06-11_1230_regularizer-tuning-as-rashomon-search | `interpretability-objective` §rashomon | placed | |
| 2026-06-11_1504_protomf-not-a-gam-additive-only-in-proto-space | `background` §iml-additivity | placed | one sentence |
| 2026-06-11_1522_part-based-prototypes-unlocked-by-features | `background` §iml (landscape block), `conclusion` §future | placed | |
| 2026-06-11_1555_interpretability-constraints-as-expert-knowledge-infusion | `interpretability-objective` §cost | placed | sleep-scoring example |
| 2026-06-11_1642_protomf-solves-c-to-y-open-on-c-semantics | `background` §iml-concepts, `background` §iml | placed | |
| 2026-06-11_1722_rashomon-set-recsys-positioning | `interpretability-objective` §rashomon | placed | amended 07-12 |
| 2026-06-11_1806_why-protomf-over-cf-the-positioning-argument | `intro` §motivation | placed | CF→MF→ProtoMF chain |
| 2026-06-15_1238_protomf-two-regularizers-geometric-effect | `background` §reg | placed | |
| 2026-06-16_1631_protomf-item-ranking-basecase | `background` §mf, `eval-framework` §design | placed | |
| 2026-06-16_1704_eval-design-citable-justification | `eval-framework` §design | placed | pre-empts reviewer |
| 2026-06-16_1715_feature-injection-design-fork | `background` §lightfm, `fi-protomf` §model | placed | route B novelty |
| 2026-06-16_1807_attribute-source-positioning | `data` §splits, `fu-protomf` §cold-users (afU road) | placed | item primary, user candidate-dependent |
| 2026-06-16_1808_intrinsic-over-posthoc-principle | `background` §iml-intrinsic, `background` §gap | placed | design principle |
| 2026-06-16_1824_feature-aware-baselines | `background` §lightfm, `eval-framework` §design | placed | exclusion reasoning |
| 2026-06-16_1837_fmxprototype-richer-interaction | `conclusion` §future | placed | logged experiment idea |
| 2026-06-16_1838_fm-gmf-suitability-hybrid-bridge | `background` §lightfm | placed | grounding-bridge verdict |
| 2026-06-16_1858_thesis-intro-storyline-feature-motivation | `intro` | placed | the storyline itself |
| 2026-06-16_2014_popularity-bias-prototype-interpretability-ablation | `hidden-effects` §braid, `interpretability-tuning` §demo | placed | ablation idea, evidence-triggered |
| 2026-06-16_2057_protomf-vs-ppm-positioning | `background` §iml (landscape block) | placed | axis mapping + disclaimers |
| 2026-06-16_2102_cosine-readout-caveat-steck | `hidden-effects` §cosine, `conclusion` §limitations | placed | probe plan |
| 2026-07-07_1651_end-to-end-visual-features-under-rec-loss | `conclusion` §future | placed | lineage step 8 |
| 2026-07-11_1515_ppm-prominence-related-work-anchor | `background` §iml (landscape block) | placed | credibility move |
| 2026-07-11_1526_facet-a-claim-boundary-coldstart-2x2 | `eval-framework` §design, `fi-protomf` §model (inventory block) | placed | claim boundary + score-share rule |
| 2026-07-11_1556_bias-channel-popularity-prototype-interpretability | `eval-framework` §design, `hidden-effects` §braid | placed | fleet stays bias-free |
| 2026-07-11_1629_baseline-suite-published-rosters | `eval-framework` §design | placed | |
| 2026-07-11_1643_canonical-feature-set-price-productcode | `data` §splits | placed | DIRECTIVE: section-worthy |
| 2026-07-11_2023_coldstart-methodology-b5-anchoring | `eval-framework` §design | placed | adopt/deviate mapping |
| 2026-07-11_2132_dc01-id-residual-three-roles | `fi-protomf` §readouts | placed | |
| 2026-07-11_2201_latent-directions-meaning-prototype-recoordinatization | `fi-protomf` §recoordinatization | placed | one-sentence framing |
| 2026-07-11_2214_thesis-recoordinatization-narrative-and-3d-figure | `fi-protomf` §recoordinatization | placed | committed 3D figure |
| 2026-07-11_2310_dc01-share-identifiability-problem-and-thesis-framing | `fi-protomf` §readouts, `hidden-effects` §shares | placed | F-DC01-06 |
| 2026-07-11_2350_interactions-explained-by-item-features-taste-revealed | `fi-protomf` §why, `intro` §motivation | placed | core argument |
| 2026-07-11_2351_thesis-outline-c1-lineage | (whole-thesis structure) | placed | governs outline.md itself |
| 2026-07-12_0006_id-twins-popularity-braid-cold-eval | `hidden-effects` §braid | placed | USER: absolutely report |
| 2026-07-12_0051_future-direction-interaction-attribute-aware | `conclusion` §future | placed | |
| 2026-07-12_1109_dc01-host-redirection-fi-protomf-lineage | — | not-thesis | self-marked NOT thesis text; conceptual payoff lives elsewhere |
| 2026-07-12_1200_linearity-interpretability-additive-attribution | `background` §iml-additivity | placed | DIRECTIVE: must become paragraph |
| 2026-07-12_1208_giving-up-predictive-power-sparsity-flips-the-trade | `interpretability-objective` §cost | placed | DIRECTIVE: reviewer inoculation |
| 2026-07-12_1224_length-channel-refund-hypothesis-hidden-effects-section | `hidden-effects` (design + §length-refund) | placed | defines the chapter |
| 2026-07-12_1251_bias-fairness-parked-out-of-scope | `conclusion` §limitations, §future | placed | visible deliberate omission |
| 2026-07-12_1409_two-route-rashomon-methodology | `interpretability-tuning` §channels | placed | founding note |
| 2026-07-12_1410_knob-metric-axis-and-principledness-spectrum | `interpretability-tuning` §channels, §taxonomy | placed | |
| 2026-07-12_1411_interpretability-knobs-map-to-rudin-families | `knobs-metrics` §taxonomy | placed | citable spine |
| 2026-07-12_1412_rashomon-set-is-partially-measurable-correction | `interpretability-objective` §rashomon | placed | correction supersedes older claims; AMENDED 2026-07-14_1153: set never measured, device only |
| 2026-07-12_1413_understandability-weighted-feature-use-and-sparsity-axes | `interpretability-tuning` §channels (UW), §taxonomy | placed | own knob |
| 2026-07-12_1653_dc05-zero-interaction-user-behavior-thesis-directive | `fu-protomf` §cold-users | placed | DIRECTIVE: exact & prominent |
| 2026-07-12_1659_five-core-loo-train-floor-and-label-hallucination-lesson | `data` §splits, `eval-framework` §design | placed | split fact placed; label-hallucination half = process lesson, not placed |
| 2026-07-12_1713_information-ordering-uhost-fu-fi-same-info-different-side | `fu-protomf` §intro | placed | lineage punchline |
| 2026-07-12_1739_ids-arm-dominance-theorem-thesis-placement | `hidden-effects` §ids-dominance, `interpretability-objective` §cost | placed | DIRECTIVE: meticulously checked |
| 2026-07-12_1808_bt-implicit-item-intercept-shift-asymmetry-directives | `hidden-effects` §bt, `fu-protomf` §model | placed | 3 DIRECTIVES |
| 2026-07-12_1815_ui-merge-intercept-feature-attributable-check-which-side-is-free | `fufi-merge` §tie, `hidden-effects` §bt | placed | |
| 2026-07-12_1830_m5-noise-for-crowding-ids-division-of-labor-centering-option | `fu-protomf` §model, `hidden-effects` §ids-dominance | placed | |
| 2026-07-12_1840_dc05-concept-review-closed | — | not-thesis | review bookkeeping; design content lives in dc05 doc |
| 2026-07-12_2114_dataset-scope-ml1m-in-amazon-out | `data` §ml1m, §splits | placed | |
| 2026-07-12_2208_ml1m-token-mass-raw-bags-popularity-coupling | `data` §ml1m, `hidden-effects` §token-mass | placed | DIRECTIVE: mention with all ml-1m fU results |
| 2026-07-12_2313_ml1m-popularity-coupling-live-magnitude | `hidden-effects` §token-mass | placed | live magnitudes |
| 2026-07-13_1817_core-level-modality-light-positioning | `background` §gap, `conclusion` §future | placed | modality-light claim |
| 2026-07-13_2033_what-remains-latent-after-grounding-still-mf | `background` §mf, §protomf | placed | verbatim capture; still-MF + latent/learned/anchored + measured-latent-mass punchline (echoes into results chapters) |
| 2026-07-13_2108_part-two-motivation-explained-well-tuning-is-reasoning-change | `intro` §1.4, `interpretability-objective` §motivation, §rashomon | placed | verbatim capture; explained-well→reasons-well collapse + knobs-change-the-reasoner Rashomon tie; bullets already in both chapters |
| 2026-07-13_2113_why-f-not-attribute-in-model-names-mechanism-vs-contribution | `fi-protomf` §model | placed | DIRECTIVE: a SINGLE SENTENCE where the f first appears (moved from intro, Matteo 2026-07-13); intro §1.3 only explains "attribute-aware" lightly |
| 2026-07-13_2134_thesis-sentence-interpretability-graded-quantify-optimize | `intro` §1.4 OR `interpretability-objective` §motivation | placed | Matteo-authored literal thesis sentence (voice preserved, "One can" deliberate); bridge sentence — exactly ONE of the two homes; both homes use a chapter reference, no model names |
| 2026-07-13_2205_anchors-combination-vs-membership-cluster-survives-in-naming | `background` §protomf, §reg; `intro` §1.2 (echo) | placed | NOT thesis prose as-is (touches too much at once) — decompose; anchors/combination-vs-membership/cluster-survives-in-naming; sets up the naming-gap story |
| 2026-07-13_2225_low-ceiling-makes-explanation-routine-bias-adjustment-part-iii | `intro` §1.1, `bias-adjustment` §avenue, `conclusion` §future (fallback) | placed | low-ceiling motivation (MNIST contrast; ties to Rashomon size) + bias-control avenue; Part III CONDITIONAL, decision after Part II design |
| 2026-07-13_2242_unfaithful-names-erode-trust-worse-than-bad-recommendations | `intro` §1.2 (+ §1.1 echo) | placed | explanation-as-interface: unfaithful names misfire visibly, trust erosion > bad rec; faithfulness-by-construction hinge to §1.3; scale caveat recorded |
| 2026-07-13_2326_part-ii-framework-knob-entry-points-explanation-backward-tracing | `interpretability-tuning` §channels, §taxonomy, §experiments | placed | STARTING POINT (Matteo): knob taxonomy by entry point + explanation-backward tracing pipeline; tailored not general; compound-loss reality check; feeds the Part II proof-of-concept milestone |
| 2026-07-14_1152_outline-refinement-ch2-8-structural-decisions | (whole-thesis structure) | placed | governs outline.md; backs Ch2 ablation bullets + merge naming bullet |
| 2026-07-14_1153_part-ii-three-chapters-method-dispatch-rashomon-device | (Part II structure), `interpretability-objective` §rashomon | placed | never-measure scoping; amends 1412 |
| 2026-07-14_1154_vocabulary-dimension-constraining-vocab-interpretability | `knobs-metrics` §taxonomy | placed | vocabulary dimension + C6 (cite original publication) |
| 2026-07-14_1218_descriptor-gap-always-a-gap-fidelity-vs-legibility | `knobs-metrics` §taxonomy, `interpretability-tuning` §method, `background` §gap (echo) | placed | descriptor gap; shrink-never-close nuance tempers The Gap hand-off |
| 2026-07-14_1220_fidelity-vs-legibility-two-metric-families | `knobs-metrics` §taxonomy | placed | two metric families, per-row tag; legibility = honest output caveat |
| 2026-07-14_1223_transparency-fidelity-legibility-ladder | `interpretability-objective` §motivation | placed | thesis-destined cursive line; upgrades SQ-2 |
| 2026-07-14_1338_demo-run-dataset-choice-informed-highlighting-transparent | — | new | untriaged (likely: data §splits V2 role + eval-framework; next /thesis-place) |

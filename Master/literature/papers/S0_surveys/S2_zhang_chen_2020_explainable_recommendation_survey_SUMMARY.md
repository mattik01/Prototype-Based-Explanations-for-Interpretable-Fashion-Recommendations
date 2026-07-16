# S2 — Zhang & Chen 2020, "Explainable Recommendation: A Survey and New Perspectives"

Mini-summary, thesis-relevant only. *Foundations & Trends in IR* 14(1) — **cite the FnTIR version; local PDF is the content-identical arXiv preprint (1804.11192).** Read fully 2026-07-13→15 (guide on `-annotated.pdf`). Insights in `protocol_vault/`.

## One line
The recsys-explainability map: a two-dimensional taxonomy (information source/display × generating model), the intrinsic-vs-post-hoc split, and the honest admission that offline explainability evaluation barely exists.

## Why it matters to us
The single highest-mileage survey cite of the thesis — four distinct jobs: (1) **positioning** (the dominant strand is review/aspect-mined and dataset-tailored; we are core-level, modality-light — [[2026-07-13_1817_core-level-modality-light-positioning]]); (2) **eval-section backbone** (its §4.3 metric families are what we anchor to and generalize — [[2026-07-15_1137_score-fidelity-metric-and-eval-section-skeleton]]); (3) **fidelity vocabulary** (§3.5 p48, §6.2.1 p81 — [[2026-07-14_1504_fidelity-term-disambiguation-relocated-not-escaped]]); (4) **ancestry** (EFM = aligning latent dims with features, our motivation with a different mechanism; explicit+latent split = the ID residual's 2014 ancestor).

## Most citable for (★ = strongest)
- ★ **The explanation-aims list + system-designer debugging** (abstract/§1.1 — *user annotation p4*): transparency, persuasiveness, effectiveness, trustworthiness, satisfaction.
- ★ **The eval gap + commissioning sentence** (§4.3 p61 "measures yet to be proposed"; §6.2.1 p80–81 "still lacks a usable offline explainability measure" + user/algorithm two-perspective split) — the license for Score Fidelity, MEP@θ, and the property-based metrics; the scoping shield for no-user-study.
- ★ **Model Fidelity formula + MEP/MER** (§4.3 p61) — what Score Fidelity generalizes (items counted → score mass measured).
- **Canonical CF explanation templates** (§1.2 p9 — *user annotation*): "users similar to you loved this item" / "similar to items you loved" — ancestry for relevant-user/relevant-item styles; item-side > user-side trust argument at §2.1 p20.
- **Industry practice = post-hoc templates over statistics, decoupled but not fake** (§3.8 p54) — [[2026-07-14_1522_industry-explanations-posthoc-templates-citable]].
- **Dacrema concession** (§3.1 p34: whether deep models progress recsys "is controversial") — survey-level support for the flat-leaderboard/Rashomon premise.
- **Fairness bridge + hedge** (§6.3 p82) — [[2026-07-15_1147_fairness-two-way-relation-linear-correction-affordance]].

## Coverage map
§1 intro/taxonomy **(core)** · §2 info sources (§2.1–2.2 relevant; §2.3–2.6 review/visual/social — skip) · §3 models (**§3.2 EFM/EMF, §3.8 post-hoc, p48 fidelity core**; rest catalogue) · §4 eval **(core)** · §5 applications (skip; §5.6 disputed — below) · §6 open directions (**§6.2 core**; rest thin).

## Our insights from reading it (protocol links)
- Modality-light positioning, generality↔richness dial, evaluation cost — [[2026-07-13_1817_core-level-modality-light-positioning]]
- Fidelity disambiguation: exact-by-construction, relocated to semantics — [[2026-07-14_1504_fidelity-term-disambiguation-relocated-not-escaped]]
- Balog = dc05's transparent ancestor; scrutability via linearity — [[2026-07-14_1511_balog-scrutability-fu-linear-edit-affordance]]
- Score Fidelity + eval-section skeleton (exists / can't / can) — [[2026-07-15_1137_score-fidelity-metric-and-eval-section-skeleton]]
- Fairness two-way relation, linear correction affordance — [[2026-07-15_1147_fairness-two-way-relation-linear-correction-affordance]]
- Reading-list additions from this read: F4 (FacT), B11 (Balog); TEM pointer in `2026-07-12_0051`.

## Caveats when citing
- **Regime mismatch everywhere:** the survey's model sections live in rating-prediction over review text; we are item-ranking/implicit with structured attributes — translate, never port.
- **§5.6 "when explanations hurt" is disputed (Matteo):** its scenarios are imported from non-recsys domains; if cited at all, cite adversarially (the Dietvorst point argues *for* faithfulness, not against explaining).
- **Attention: contested ≠ refuted** (W&P stands) — never caricature; our claim is that the faithfulness *question doesn't arise*, not that attention fails it.
- Terminology: scrutability (user-facing, Tintarev/Balog) ≠ our Scrutiny Protocol (internal audit).

# C3 — Rudin 2019, "Stop Explaining Black Box ML Models for High-Stakes Decisions and Use Interpretable Models Instead"

Mini-summary, thesis-relevant only. Position/commentary paper (Nature Machine Intelligence; 20-page arXiv manuscript v3). PDF + skim-guide highlights in this folder. Insights in `protocol_vault/`.

## One line
For high-stakes decisions, don't explain black boxes post-hoc — build models that are *inherently* interpretable, because post-hoc explanations are necessarily unfaithful and the accuracy–interpretability trade-off is largely a myth.

## Why it matters to us
- **The canonical citation for the thesis's "Rudin axis"**: intrinsic / by-design interpretability > post-hoc. Frames why feature-grounded prototypes (interpretable representation built *into* the model) beat after-the-fact explanation. Links [[2026-06-16_1808_intrinsic-over-posthoc-principle]], [[2026-06-11_1111_protomf-within-interpretable-ai]] (one of the candidate principle-frameworks for the positioning section).
- Rudin explicitly **endorses the prototype / "this looks like that" lineage** (ProtoPNet) as the path to intrinsic interpretability — the exact ancestry ProtoMF descends from.

## Most citable for (★ = strongest)
- ★ **Post-hoc explanations must be unfaithful** (p2): "Explanations must be wrong. They cannot have perfect fidelity… if the explanation was completely faithful… one would not need the original model." → the cornerstone reason to prefer intrinsic.
- ★ **Prototype reasoning is intrinsic** (p12): the prototype layer's "explanations are the actual computations of the model, and these are not posthoc explanations." → Rudin blessing ProtoMF's mechanism.
- The **accuracy–interpretability trade-off is a myth**, especially with structured/meaningful features (p1).
- **Rashomon-set argument** (p13): a large set of near-optimal models likely contains an interpretable one → theoretical backing that prototype constraints can sit inside MF's accuracy envelope.
- **Interpretability is domain-specific** (no universal definition): sparsity, monotonicity, case-based reasoning, domain constraints (p0). Saliency shows *where* not *what* (p3).

## Coverage map (relevance-flagged)
Intro **(defn, domain-specific)** · **Issues w/ Explainable ML (5 pts — trade-off myth, "must be wrong")** · Issues w/ Interpretable ML (adoption/IP) · Governance/policy *(off-axis)* · Algorithmic Challenges 1–2 logical/scoring *(off-axis)* · **Challenge 3 — case-based / prototype nets** · **Rashomon argument** · appendices.

## Our insights from reading it (protocol links)
- Prefer intrinsic over post-hoc; the extension moves prototype *grounding* post-hoc→intrinsic — [[2026-06-16_1808_intrinsic-over-posthoc-principle]]
- Finer axis: base ProtoMF is *already* intrinsic in its reasoning; only prototype **naming** is post-hoc — don't cite Rudin to imply base ProtoMF is a black box — [[2026-06-11_1220_posthoc-naming-vs-intrinsic-model-distinction]]
- Regularizer tuning under an accuracy tolerance = searching the Rashomon set for its most interpretable member — [[2026-06-11_1230_regularizer-tuning-as-rashomon-search]], [[2026-06-11_1144_proto-regularizers-interpretability-not-accuracy]]
- Part-based-prototype critique has little bite for ProtoMF (similarity-to-all-prototypes already breaks an entity into facets) — [[2026-06-11_1522_part-based-prototypes-unlocked-by-features]]

## Caveats when citing
- **Stakes/regime mismatch:** Rudin's domain is high-stakes *classification* (recidivism, healthcare, credit) with "catastrophic harm." Recommendation is low-stakes item-ranking — cite for the *principle* (intrinsic > post-hoc), not the urgency/harm framing.
- **Grand-Challenge numbering:** this 2019 paper has only **3 Challenges**; the part-based-prototype point lives in its **Challenge #3**. The vault's "GC#4"/"GC#9" references belong to the **separate Rudin et al. 2022 "10 Grand Challenges" survey** — never attribute "GC#N" to this paper.
- **Scope discipline (reading note, p5):** the paper lists ~5 distinct problems with black-box explanations — in the thesis, *mention* the list, don't *explain* each one.

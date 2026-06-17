# S5 — Elhadri et al. 2025, "This looks like what? Challenges and Future Research Directions for Part-Prototype Models"

Mini-summary, thesis-relevant only. Survey (arXiv 2502.09340, 2025). PDF in this folder. Insights in `protocol_vault/`.

## One line
A 2025 survey of **vision** part-prototype models (PPMs, the ProtoPNet family) that derives a taxonomy of their open challenges — prototype number/quality, methodology, generalization, safety — and five future directions; the cleanest available statement of the problems intrinsic prototype-interpretability still faces.

## Why it matters to us
- PPMs share ProtoMF's **prototype-by-similarity, ante-hoc** principle, so ProtoMF inherits S5's **modality-independent *semantic* challenges** (latent-similarity ≠ human-similarity, unsupervised prototypes lack meaning, cosine unreliability) — exactly what **facet-1** (feature-grounded prototypes) attacks.
- But ProtoMF is **whole-object** prototype-similarity, **not a spatial part-prototype model** → the spatial/part challenges are n/a. Positioning: [[2026-06-16_2057_protomf-vs-ppm-positioning]].

## Most citable for (★ = strongest)
- ★ **§4.2** — *"PPMs are not intrinsically interpretable in practice, because prototypes are learned in an unsupervised manner, and therefore lack human-understandable semantics."* The gap facet-1 fills.
- ★ **§3.1.2 similarity-perception gap** — latent-space similarity ≠ human similarity (Kim et al., HIVE); *"matching does not specify what the model looks at (shape, color, texture)."*
- **§4.4** — cosine similarity is *"an unreliable metric for similarity"* (Steck, WWW 2024) — lands *harder* on us (we are MF).
- **§4.4** — proposes *"semantically annotated prototypes via CBM-style semantic features"* as an open direction → facet-1 instantiates this for recsys ("field points here, we go here").
- **Abstract / Rudin framing** (user-highlighted) — intrinsic models still secondary to post-hoc/black-box despite being "better"; the thesis motivation.
- **§3.1.1** — number-of-prototypes trade-off: fewer = interpretable, more = meaningful.

## Coverage map (relevance-flagged)
§1 ProtoPNet bg · §2 method · **§3.1 Prototypes (number; quality = semantics + similarity)** · §3.2 Methodology · §3.3 Generalization (**§3.3.2 bottleneck assumption**) · §3.4 Safety · **§4 directions (§4.2 theory-grounded, §4.4 alignment)** · Tab 1–2 idea catalog (mostly vision-specific, low transfer).

## Our insights from reading it (protocol links)
- ProtoMF vs PPM — shared principle, differing axes (A: whole-object vs part; B: ranking vs classification readout); + S5 §4.4 related-work anchor — [[2026-06-16_2057_protomf-vs-ppm-positioning]]
- Cosine prototype-readout inherits Steck's unreliability, *harder* because MF; computation-fidelity intact, semantic-fidelity at risk; + future readout-metric experiment — [[2026-06-16_2102_cosine-readout-caveat-steck]]

## Caveats when citing
- **Regime mismatch:** vision, single-label image classification — not item-ranking/implicit feedback. Flag it.
- **Transfer only the modality-independent semantic challenges;** spatial/part-specific ones (misalignment, patch-viz ambiguity, deformable prototypes, pixel attribution) are **n/a**.
- Never call ProtoMF a "part-prototype model" — frame it as a **recsys instance of the shared prototype-similarity principle**.
- CBM (§4.4 [29]): borrow the **named-concept principle, not the bottleneck architecture** — the bottleneck is structurally unsuited to recommendation's dyadic, collaborative target.

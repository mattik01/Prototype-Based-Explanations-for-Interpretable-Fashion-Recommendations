# Prototype-Based Explanations for Interpretable Fashion Recommendations

> **Work in progress.** I am following a plan roughly (Master/docs/masterplan.md) and checking things off as I go and adding new things as they come up

This repository is a fork of [ProtoMF](https://github.com/karapostK/ProtoMF) (Melchiorre et al., RecSys 2022), extended as part of a Master's thesis exploring **feature-grounded prototype-based explanations** for fashion recommendation using the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) dataset.

**Author:** [Matteo Gläser](https://www.linkedin.com/in/matteo-gl%C3%A4ser-144422270/)
**Supervisor:** [Assoc. Prof. Dr. Eva Zangerle](<!-- [link to supervisor profile](https://dbis.uibk.ac.at/zangerle) -->)
**Institution:** [University of Innsbruck](https://www.uibk.ac.at/)

## Thesis Goals

Two contributions, built on a replicated ProtoMF baseline:

**Foundation**
1. **Replicate** the original ProtoMF results across models and datasets, validating the baseline before extending it.
2. **Integrate the H&M fashion dataset** (~31M transactions, ~105K articles with rich categorical metadata) into the ProtoMF pipeline — a domain where recommendations genuinely benefit from human-readable explanations.

**⭐ Contribution 1 — a feature-grounding lineage**
3. **Ground the prototypes in item feature vocabulary** (product type, colour, department, …) so the axes of the prototype space carry names read from the model's *own parameters*, rather than from a post-hoc probe bolted on afterwards.
4. **Extend the grounding variant-by-variant**, mirroring ProtoMF's own item → user → dual structure, and compare each grounded variant *like-for-like* against its ungrounded ProtoMF counterpart — measuring exactly what interpretability costs, or gains, in accuracy.

**⭐ Contribution 2 — interpretability, quantified**
5. **Extract the fashion explanations and *measure* their quality** (prototype coherence, how much of a recommendation the named features actually account for, user-profile sharpness, …) — treating interpretability as something evaluated, not merely asserted.

*Guiding principle:* an interaction is *taste × content* — item content is recorded, but user taste is only revealed through behaviour — so grounding starts on the item side.
*Open directions (out of current scope):* pulling product images in directly under the recommendation loss; bias/fairness implications and opportunities of transparent prototypes.

## Background: ProtoMF

[ProtoMF](https://doi.org/10.1145/3523227.3546756) (Melchiorre et al., RecSys 2022) replaces standard matrix factorization embeddings with **prototype-based representations**: users and items are described by their cosine similarity to a set of learned prototypes, producing inherently interpretable recommendations. The paper introduces user-side, item-side, and dual prototype variants and shows they match or exceed standard MF baselines while providing built-in explanations.

This thesis extends ProtoMF by **grounding its prototypes in real item attributes** — turning the prototype-space axes from opaque coordinates into named, feature-derived directions — and then quantifying how well those explanations hold up (see *Thesis Goals* above).

![ProtoMF Diagram](./pdfs_and_images/3-protomf_models_schema.png "ProtoMF Diagram")

## License

The code in this repository is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

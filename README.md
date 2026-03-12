# Prototype-Based Explanations for Interpretable Fashion Recommendations

> **Work in progress.** I am following a plan roughly (Master/docs/masterplan.md) and checking things off as I go and adding new things as they come up

This repository is a fork of [ProtoMF](https://github.com/karapostK/ProtoMF) (Melchiorre et al., RecSys 2022), extended as part of a Master's thesis exploring **feature-aware prototype-based explanations** for fashion recommendation using the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) dataset.

**Author:** [Matteo Gläser](https://www.linkedin.com/in/matteo-gl%C3%A4ser-144422270/)
**Supervisor:** [Assoc. Prof. Dr. Eva Zangerle](<!-- [link to supervisor profile](https://dbis.uibk.ac.at/zangerle) -->)
**Institution:** [University of Innsbruck](https://www.uibk.ac.at/)

## Thesis Goals (for now)

1. Replicate the original ProtoMF results across models and datasets to validate the baseline
2. Integrate the H&M Fashion dataset (~31M transactions, ~105K articles with rich metadata) into the ProtoMF pipeline, to test the proposed method further.
3. ⭐ **Extend the prototype architecture/method to leverage item features** (categorical metadata such as product type, colour, department), not just collaborative filtering signals.
4. ⭐ **Extract prototype-based explanations** from the prototypes, used in the recommendation algorithm, and evaluate their quality and them as a benefit of using prototype-based recommendation.

## Background: ProtoMF

[ProtoMF](https://doi.org/10.1145/3523227.3546756) (Melchiorre et al., RecSys 2022) replaces standard matrix factorization embeddings with **prototype-based representations**: users and items are described by their cosine similarity to a set of learned prototypes, producing inherently interpretable recommendations. The paper introduces user-side, item-side, and dual prototype variants and shows they match or exceed standard MF baselines while providing built-in explanations.

This thesis will extend the approach by adding a **feature-aware prototype space** alongside the collaborative one, enabling (hopefully) richer explanations grounded in actual item attributes. The Details of the integration are yet to be determined. 

![ProtoMF Diagram](./pdfs_and_images/3-protomf_models_schema.png "ProtoMF Diagram")

## License

The code in this repository is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

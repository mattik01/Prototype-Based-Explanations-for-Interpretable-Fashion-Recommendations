---
date: 2026-06-15
time: "12:38"
phase: 4
tags: [thesis/background]
placed: 2026-08-20
---
# ProtoMF's two prototype regularizers and their geometric effect

ProtoMF's `PrototypeEmbedding` applies **two distinct losses** to the batch×prototype cosine-sim matrix (`feature_extractors.py:248-329`): **`reg_batch`** (`-max(dim=1)`, over the prototype axis) pulls *each object* toward its nearest prototype → every user/item is covered; **`reg_proto`** (`-max(dim=0)`, over the batch axis) pulls *each prototype* into a dense region of objects → no dead/unused prototypes. (`soft` variants = softmax-entropy on the same axis.) Total = `sim_proto_weight·r_proto + sim_batch_weight·r_batch`; `user_item_proto` has two such modules (user+item) → **four weights**.

Geometric consequence, visible in the reg-sweep t-SNE grids: stronger regularization reshapes the raw embedding space into **prototype-centred clusters** (objects condense around prototypes, prototypes sit in dense regions), vs a diffuse cloud at reg-off — the geometric signature behind the genre-purity rise (ml-1m off→x1: 0.70→0.87). **Caveats:** at x100 the clustering is degenerate and accuracy collapses (more clusters ≠ better); and our reg_sweep scaled all four weights by a single multiplier, so it **cannot disentangle** the two regularizers or the user vs item side — a 2D/4D grid would be needed for that.

[[paper-understanding]] [[explanations]] [[experiments]] [[phase-4]]

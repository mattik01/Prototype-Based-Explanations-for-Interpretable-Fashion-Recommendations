# dc07 — prior-art probe (lightweight path, §10 of the requirements memo)

**Written:** 2026-09-08 (build session). **Purpose:** verify the citations the memo listed from memory
and place dc07 against them. paper-search MCP was down again; verified via web search (URLs below).
Working notes, not thesis text.

## Verified references

| ref | verified as | relation to dc07 |
|---|---|---|
| Barkan, Hirsch, Katz, Caciularu, Koenigstein — *Anchor-based Collaborative Filtering*, CIKM 2021, doi 10.1145/3459637.3482056 | ✔ users/items = convex combinations (softmax) over a shared anchor set; **exclusiveness** (each item picks one anchor) and **inclusiveness** (all anchors used) constraints | the in-fleet precedent (`acf`, `feature_extractors.py:534`): softmax membership over dot products, τ fixed at 1, representation = reconstruction Σ m·anchor. dc07 keeps ProtoMF's score form Σ_l t_{i,l} m_l instead (membership vector IS the representation) and starts with the host's 'max' regularisers; the ACF entropy pair is the stage-2b axis (D1). Sanity anchor D5: acf ml-1m 0.6006. |
| Li, Liu, Chen, Rudin — *Deep Learning for Case-Based Reasoning through Prototypes*, AAAI 2018, arXiv 1710.04806 | ✔ prototype layer over an autoencoder latent; squared-L2 distances to prototypes; two interpretability regularisers (each prototype near a training point / each training point near a prototype) | ProtoMF's ancestor for the *inclusion* regularisers ('max' row/col). ProtoMF replaced L2 by cosine (paper §3.1); dc07 replaces cosine by a dot-product softmax — a different departure from the same ancestor. |
| Chen, Li, Tao, Barnett, Su, Rudin — *This Looks Like That* (ProtoPNet), NeurIPS 2019, arXiv 1806.10574 | ✔ log-ratio activation log((‖z−p‖²+1)/(‖z−p‖²+ε)), prototype projection onto training patches | bounded, positive, monotone-in-distance activation; dc07's sigmoid variant is the bounded-per-prototype analogue without competition; ProtoPNet has no normalisation across prototypes (like sigmoid, unlike softmax). |
| Snell, Swersky, Zemel — *Prototypical Networks for Few-shot Learning*, NeurIPS 2017, arXiv 1703.05175 | ✔ p(y=k|x) ∝ exp(−d(f(x), c_k)), softmax over negative squared Euclidean distances (no explicit temperature in the paper; the scale rides on the embedding norm) | the canonical "softmax over prototype similarities = class membership" reading. Note for dc07: with squared L2, −‖q−p‖² = 2⟨q,p⟩ − ‖q‖² − ‖p‖²; the ‖q‖² term cancels in the softmax over l, so ProtoNets' softmax equals dc07's dot-product softmax with a per-prototype offset −‖p_l‖²/τ — the sigmoid bias b_l can absorb exactly that. τ absent there ⇔ the D2 remark that ‖q‖ and τ are jointly redundant. |
| Steck, Ekanadham, Kallus — *Is Cosine-Similarity of Embeddings Really About Similarity?*, WWW '24 Companion, arXiv 2403.05440 | ✔ for regularised linear models cosine similarity of learned embeddings can be arbitrary/non-unique; sometimes worse than the raw dot product | taxonomy row 10's source; the analytic reason to try a dot-based read-out at all (P6). |
| Jain & Wallace — *Attention is not Explanation*, NAACL 2019; Wiegreffe & Pinter — *Attention is not not Explanation*, EMNLP 2019, arXiv 1908.04626 | ✔ the debate on reading normalised weights as explanations; W&P: depends on the definition and needs the whole model in the test | applies verbatim to reading m as "who you are": memberships are not counterfactual importances. dc07's thesis framing must present m as a *representation unit* and the log-odds as the exact attribution, not m as an explanation of the score. |
| Hofmann — *Latent semantic models for collaborative filtering*, ACM TOIS 22(1) 2004; Blei, Ng, Jordan — *LDA*, JMLR 3 2003 | ✔ pLSA/aspect model for CF: users as mixtures over latent aspects; LDA adds Dirichlet priors (mixed membership) | the probabilistic ancestor of "user = distribution over K types". dc07's softmax membership is the discriminative, non-Bayesian version; the F5 "Dirichlet-style shrinkage" idea in the memo §9 is exactly the LDA step — recorded as future work, not built. |

## Placement (one paragraph for the knobs chapter)

Prototype layers read their input through a similarity; the three families in the literature are L2
(Li et al., ProtoPNet, ProtoNets), cosine (ProtoMF) and normalised dot-product memberships (ACF,
attention, pLSA/LDA in the probabilistic reading). ProtoMF chose cosine for bounded, non-negative,
scale-free activations; Steck et al. show that scale-freeness on learned embeddings is not a
guarantee of meaning, and the lineage's own measurements locate the layer's losses in exactly the
properties cosine enforces (norm blindness, +1 channel, angle bottleneck). dc07 moves the host
one step along that axis — to ACF's membership read-out under ProtoMF's score form — as a single
switch with one temperature, so that the comparison isolates the read-out and nothing else. The
attention-as-explanation debate fixes how the result may be *presented*: the exact unit is the
log-odds between prototypes, not the membership itself.

## Sources
- https://dl.acm.org/doi/10.1145/3459637.3482056 · https://www.semanticscholar.org/paper/dbb907578e82c0b8f775fad2c67b9474a2cd5b4a
- https://arxiv.org/abs/1710.04806 · https://ojs.aaai.org/index.php/AAAI/article/view/11771
- https://arxiv.org/pdf/1806.10574
- https://arxiv.org/abs/1703.05175 · https://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning
- https://arxiv.org/abs/2403.05440 · https://dl.acm.org/doi/10.1145/3589335.3651526
- https://aclanthology.org/D19-1002/ · https://arxiv.org/abs/1908.04626
- https://dl.acm.org/doi/10.1145/963770.963774

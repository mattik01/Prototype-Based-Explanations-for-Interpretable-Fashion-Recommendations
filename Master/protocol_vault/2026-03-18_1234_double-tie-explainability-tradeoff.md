---
date: 2026-03-18
time: "12:34"
phase: 2
tags: [thesis/background, thesis/fufi-merge]
placed: 2026-08-20
---
# Double-tie architecture creates compromised embeddings

Deep analysis of the `prototypes_double_tie` (user_item_proto) architecture revealed a fundamental explainability tradeoff. Weight sharing forces each side's embedding to serve two conflicting optimization pressures: (1) being well-positioned for cosine similarity with its own prototypes, and (2) containing information that survives linear projection into a useful signal for the other side's prototype space. This compromises the clean one-dimension-one-prototype correspondence that makes single-prototype variants (`item_proto`, `user_proto`) fully explainable. The projected embedding branches are opaque linear combinations, and even the prototype similarity values are computed from embeddings shaped by both pressures. This is a key consideration for Phase 4 feature-aware design — extending the architecture should preserve explainability where possible.

[[paper-understanding]] [[decisions]] [[phase-2]]

---
date: 2026-06-16
time: "16:31"
phase: 2
---
# ProtoMF as item-ranking base case for positioning

ProtoMF is trained and evaluated as an **item-ranking** model (model selection on `hit_ratio@10`, losses BPR / sampled-softmax / BCE), not rating prediction. This is the base case the feature-aware extension positions against. Noted while reading the S7 attribute-aware CF survey (Chen et al. 2020), which is framed almost entirely around **rating prediction / RMSE** (pointwise) — so its 4-category taxonomy (DMF / GMF / GF / HG) is useful for *positioning* the feature-aware prototype work, but its evaluation philosophy differs from ours and should not be adopted wholesale.

[[paper-understanding]] [[evaluation]] [[decisions]] [[phase-2]]

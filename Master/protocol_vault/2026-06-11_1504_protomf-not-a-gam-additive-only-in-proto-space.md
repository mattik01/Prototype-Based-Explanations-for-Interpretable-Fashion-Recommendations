---
date: 2026-06-11
time: "15:04"
phase: 4
---
# Minor note: ProtoMF is not a GAM — additive only in prototype space

Side note for the interpretable-AI thesis section (one sentence there at most): ProtoMF is **not** a GAM — inputs are IDs not features, the additive terms are *learned* prototype similarities, and the user×item product is bilinear (the interaction GAMs exclude). It does *borrow* the GAM virtue of additive decomposability — the score splits into per-prototype contributions, which is what makes the explanations readable — so it can be called "GAM-like in the derived prototype-similarity space" while remaining classified under case-based reasoning.

[[phase-4]] [[explanations]] [[paper-understanding]]

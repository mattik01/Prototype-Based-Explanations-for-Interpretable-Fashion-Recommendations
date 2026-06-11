---
date: 2026-06-11
time: "15:22"
phase: 4
---
# Minor note: part-based prototypes — situational, worth remembering

From Rudin GC#4's part-based prototypes discussion. Rudin's criticism (whole-to-whole comparison can't handle multi-faceted observations) has **little bite for ProtoMF**: representing an entity by its similarity to *all* prototypes simultaneously already breaks the problem up quite firmly — each prototype can capture a facet, and the mixture covers multi-faceted taste. The part-based concern only becomes relevant in one specific design situation: if the feature-aware extension **attaches feature information to existing CF prototypes** (one prototype = CF vector + feature profile, compared whole-to-whole again) rather than having separate feature-prototype spaces — and that attachment design is actually quite a desirable implementation route. If we go that way, remember part-based / feature-subset comparison (Kim et al. 2014, prototype-parts on tabular data) as the refinement and possible future work; otherwise this stays a side remark.

[[phase-4]] [[explanations]] [[paper-understanding]]

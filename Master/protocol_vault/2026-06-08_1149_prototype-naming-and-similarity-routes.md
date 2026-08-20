---
date: 2026-06-08
time: "11:49"
phase: 4
tags: [thesis/background, thesis/fi-protomf]
placed: 2026-08-20
---
# Derived prototype naming (lift) + the two similarity routes

Built a **post-hoc / derived** prototype-naming layer (baseline yardstick; the future feature-aware method should produce *intrinsic* names instead): each prototype's top-k items → a feature profile (per categorical value: count, top-k freq, global freq) → a selection score → the top descriptors become the name, with all numbers dumped for transparency. The two routes are asymmetric and named accordingly: **item prototypes = closeness** (items share the item-prototype space, so top-k = highest shifted-cosine similarity) vs **user prototypes = activation** (items don't share the user space, so top-k = items whose learned projection coefficient `t̂[p]` is most strongly recommended for that prototype direction).

**Decision — `lift` (top-k freq ÷ global freq) is the default scoring, not raw frequency:** lift divides out base rates so *distinctive* values win over merely *common* ones (e.g. a prototype names as "Romance, Drama" not "Drama, Romance"; Horror at lift ~11.7). Raw frequency drags ubiquitous values like Drama/Comedy to the front and is kept only as a comparison mode (`scoring={lift,raw}`, output nested per scoring). Validated on ml-1m (18 genres, 53% multi-genre): item-prototype names are coherent (Horror/Sci-Fi/Thriller, Fantasy/Children's/Adventure). Also added a **genre small-multiples** figure — one TSNE panel per value, items with the value highlighted, prototypes coloured black→red by their lift score — which makes lift's base-rate effect visible (rare genres glow, common genres stay dim under a shared scale).

[[explanations]] [[decisions]]

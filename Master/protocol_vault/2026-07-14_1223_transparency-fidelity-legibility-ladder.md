---
date: 2026-07-14
time: "12:23"
phase: 4
tags: [thesis/interpretability-objective]
placed: 2026-08-20
---
# The transparency–fidelity–legibility ladder: "explained well" decomposed into three established terms

*(Verbatim-capture entry — explanation worked through in the 2026-07-14 session and explicitly praised by Matteo ("the explanation is amazing, and should be present in the thesis"). Arose from his question whether "transparency" should replace "fidelity" as the truth-family name; the answer is no, and the reasoning produced the ladder. Children of [[2026-07-14_1218_descriptor-gap-always-a-gap-fidelity-vs-legibility]] and [[2026-07-14_1220_fidelity-vs-legibility-two-metric-families]]; upgrades the SQ-2 terminology story.)*

## Why fidelity must NOT be renamed to transparency

Both terms are established — **for different things** — so swapping would misuse the established meaning rather than borrow it:

1. **Transparency** is taken twice. In the literature (Lipton's "Mythos" taxonomy) it is a property of the *model* — you can see its workings (simulatability, decomposability, algorithmic transparency); it says nothing about whether claims made *about* the model are true. And in the thesis's own vocabulary it is the *window* — architectural, delivered by the cosine read-out, essentially binary ("transparency is architectural; the regularizers shape legibility, not transparency"; SQ-2 demoted transparency as goal-word for exactly this weakness).
2. **Fidelity/faithfulness** IS the established term for explanation-truthfulness in the explanation-evaluation literature (G1 fidelity metrics, S4's axis).
3. **The decisive argument:** the model can be perfectly transparent while its names are unfaithful — that is the naming problem itself ([[2026-07-13_2242_unfaithful-names-erode-trust-worse-than-bad-recommendations]]). If transparency also meant descriptor-truthfulness, the sentence "the host is fully transparent, yet its prototype names may be lies" becomes unsayable — and that sentence is roughly the thesis of §1.2. One word for both destroys the exact distinction the contribution lives in.
4. **Definitional nuance for the prose:** classic fidelity asks "does the post-hoc explanation approximate the black box?" — a question that dissolves for intrinsic models (no approximation gap). What survives intrinsic-ness is precisely the narrower version used here: "does the *descriptor* match the *element*?" The thesis inherits the established word and sharpens it.

## The ladder (thesis-destined, near-verbatim)

A clean three-step decomposition of "explained well", each term keeping its literature meaning, each mapping to a distinct thesis mechanism — the window/room image made technical:

- **Transparency: the window exists** (architectural, Part I's read-out — essentially free in this model family).
- **Fidelity: the window doesn't lie** (descriptor gap small and auditable — grounding's contribution).
- **Legibility: the room is tidy** (complexity low — the knobs' main territory).

## Placement

Primary home: Part II's motivation chapter ("Why Quantify Interpretability?"), where the window/room bullets already live — the ladder is their technical form. Echo: the fidelity/legibility bullet in the knobs-and-metrics taxonomy. Upgrades SQ-2: transparency gets a precise, subordinate role (rung 1) instead of competing with "interpretable" as goal-word.

[[explanations]] [[decisions]] [[thesis-writing]] [[phase-4]]

---
date: 2026-07-12
time: "17:13"
phase: 4
---
# Information ordering: info(U-ProtoMF) ⊂ info(fU) ⊆ info(fI) — same information, different side

*(Verbatim-capture entry — worked through at the dc05 concept review sitting;
the user selected exactly this point for the protocol out of a longer
two-forces discussion. His framing question: "in fU we offer more info to the
model than in U-ProtoMF, but not more information than fI — is that fair to
say?" Answer: fair, and slightly sharper is available.)*

## The ordering, with the sharp part

- **U-ProtoMF sees only the interaction matrix.** fU sees interactions + the
  item-attribute catalog (through baskets) — **strictly more**.
- **fU vs fI: both consume the same two data sources** — same interactions,
  same 426-value catalog. So "fU offers not more than fI" is fair. The sharper
  statement: **fU actually uses at most fI's information**, because fU only
  ever reads the attribute rows of items that *someone purchased in the train
  window* — an item never bought in train contributes its attributes nowhere
  in fU, whereas fI reads every item's attributes at scoring time. That is
  precisely why fI can represent cold items and fU cannot. On V1 the
  difference is 3 items out of 13,651 (the train-unseen accident), so
  practically identical — but conceptually the ordering is clean:

> **info(U-ProtoMF) ⊂ info(fU) ⊆ info(fI)**

## The punchline (thesis-quality, for the lineage narrative)

fI and fU differ not in *how much* information they receive, but in **which
side of the score the same information is installed on**: items built from
their own attributes (fI) vs. users built from their purchases' attributes
(fU). **The lineage holds the information constant and moves where it lives.**
Corollary kept honest: in fU the purchase history itself is NOT new
information — the training loss already consumes every interaction; what
history contributes in fU is *structure* (a stated aggregation replacing free
compression), while the only genuinely new information at both stages is the
catalog metadata.

[[phase-4]] [[decisions]] [[explanations]]

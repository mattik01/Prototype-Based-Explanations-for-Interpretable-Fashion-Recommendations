# Design-Candidate Scratchpad

> Free-form idea inbox for the `/design-candidate` protocol (v1.2). Jot anything
> while reading — seed ideas, extensions, constraints, doubts, "what if X" — as
> quickly and roughly as you like; one `###` block per entry, date optional but
> helpful.
>
> What the protocol does with it: **Step 0** reads it, **Step 1** gives every
> open entry a one-line disposition (seed / shapes the candidate / not this
> cycle / answered inline) — nothing is silently ignored — and **Step 6**
> appends an indented `↪ dcNN` marker under consumed entries. Your text is
> never edited or deleted. Entries without a marker are still open.
>
> The protocol may also **add** entries here (v1.3, Ground rule 9): adjacent
> ideas or variants that surface mid-cycle but can't be pursued now (R8) are
> captured as `### [captured dcNN] <title>` blocks instead of being discarded.
> They are swept in Step 1 like any open entry; they're distinguishable from your
> own entries by the `[captured dcNN]` tag.

---

## Entries

### Attach feature profiles to existing CF prototypes (2026-06-11)
Instead of building separate "feature prototypes," attach feature information directly to the existing CF prototypes — one prototype = CF vector + feature profile (keeps the prototype set unified, features make existing prototypes nameable). Whether this or separate feature-prototype spaces is preferable is to be decided later. Caveat to remember from Rudin GC#4: this design reintroduces whole-to-whole comparison, so the part-based prototypes idea (compare on feature subsets, Kim et al. 2014 for tabular data) becomes the relevant refinement there — usable as future work. With separate feature-prototype spaces this concern mostly disappears, since similarity-to-all-prototypes already breaks an entity into facets.

<!-- Template (copy, or just write — format is not enforced):

### <short title> (2026-MM-DD)
<the idea / consideration / question, any length, any roughness>

-->

# Review v8 (closing) — Section 5.1 `fi-why` ("Prototype Grounding")

**Date:** 2026-08-21 · **Skill:** /thesis-review (eighth invocation) · **Git state:** HEAD 9d8bdd4, uncommitted on `feat/scrutiny`
**State reviewed:** Matteo's saved rework — title "Prototype Grounding"; hinge paragraph cut, asymmetry sentence moved to head of transition; "positions" clause removed from P1; "which are in the user's language" rewording; clustering-methods sentence; inline uncertainty note in P3.
*LLM-usage-disclosure artifact — do not delete.*

## Phase 1 — mechanical
1. "The next section t herefore constructs" → "therefore" (broken word)
2. Rewrap to 78 chars

## Phase 2 — findings

### V8-1 — MAJOR · accuracy of "names them after training through clustering methods"
> "To make its prototypes interpretable, ProtoMF names them after training through clustering methods (Section~\ref{sec:bg-protomf})."
The ProtoMF paper's interpretation mechanism is inspecting a prototype's most similar items (top-k by the model's own similarity) and naming it from those representatives — a nearest-neighbor read-out, not clustering. The Background section this points to will teach the actual method, making the mismatch visible. **Fix direction:** "by inspecting each prototype's most similar items" (or whatever phrasing sec:bg-protomf ends up using — the two must agree).

### V8-2 — NOTE · Matteo's inline marker "(this sentence is uncertain might need revision)"
Sits in compilable prose — it will print if forgotten. Suggest moving it to a % comment on its own line (content untouched). On the sentence itself (the forcing move): reviewer's opinion is that it is sound and load-bearing — it is what separates grounding from side-information; if anything, "may well add" could firm up to "may add". No revision required from this side.

### Structure after the hinge cut — CLEAN
Payoff → item paragraph junction reads directly ("the two ingredients" → "For the item ingredient"); the asymmetry sentence at the transition head gives "therefore" its explicit antecedent. The cut cost nothing.

### Carried (unchanged): notes block to write out or cut; koren/samuelson decision; "appropriate vocabulary" ×2 in the item paragraph (V7-1, kept by Matteo); "(see Chapter II)" stays until internal targets firm up (Matteo).

## Close
Section retitled "Prototype Grounding" (label/slug unchanged). Prose converged; V8-1 is the only substantive open point from this pass.

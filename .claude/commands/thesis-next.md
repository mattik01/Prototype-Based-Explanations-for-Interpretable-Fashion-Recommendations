# /thesis-next — Recommend which thesis section(s) to work on next

Advise Matteo where writing effort pays off most *right now*. Core principle: work sections that are **material-rich AND upstream-stable AND influx-quiet**. Introduction, abstract, and conclusion are structurally excluded until the end (famously bad early targets).

## Steps

1. **Refresh inputs**: read `Master/thesis/outline.md` (status column), `Master/thesis/placement_ledger.md`, and skim git log / MEMORY.md for what changed since the last run (gates closed? fleet landed? new vault entries?). If the ledger looks stale (untriaged vault entries), say so and suggest `/thesis-place` first — don't advise from stale data.
2. **Score each section** (chapter or subsection granularity where it matters) on three axes, plain 0–2 each:
   - **Material richness** — placed ledger entries + docs density; for results sections, do the numbers exist as thesis-grade runs?
   - **Upstream stability** — is the underlying work gate-closed (scrutiny closed, design frozen, splits frozen)? A section whose subject is still being audited scores 0.
   - **Influx quiet** — how likely is new material to arrive and force a rewrite? (Active literature reading feeds related-work; pending SC steps feed candidate chapters.)
   Writability = min of the three axes matters more than the sum — one hard blocker kills a section.
3. **Account for writing state**: sections already outlined/drafted rank higher for a *finishing* session; untouched ripe sections rank higher for a *starting* session. Ask which mood if unclear — one short question max.
4. **Literature readiness check (Matteo, 2026-07-14)** — for each candidate target, do an *honest and careful* evaluation of what should be in Matteo's head before writing it:
   - **Determine reading state** from `Master/literature/`: the reading list (IDs, tiers, ★/◦ markers), `*-annotated.pdf` (= read with margin notes), `*_SUMMARY.md` (= finalized via `/paper-finalize`). No summary and no annotations ⇒ treat as unread.
   - **Identify what the section actually leans on** — papers it will cite or argue against (use the section's material-map comments, its placed vault entries, and the reading list's per-ID purpose notes). Only literature the section genuinely needs — never demand reading for its own sake; "nothing needed" is a perfectly good answer.
   - **Issue one of three verdicts per relevant paper**: **READ UP** (unread or skimmed only — read before writing), **REVIEW** (read + finalized — re-reading the `_SUMMARY.md` before the session suffices), or **optional** (nice-to-have, not blocking).
   - A target whose must-read pile is heavy is still recommendable — but say so plainly and count the reading time as part of "what the session produces."
5. **Recommend 1–3 targets**, plainly: section, why now (one or two sentences each), what the session would concretely produce (paragraph outline? prose? review pass?), **and its literature verdicts** (READ UP / REVIEW lists, or "nothing needed"). Name the single best next `/thesis-write` invocation.
6. **Also surface** (briefly, at the end): sections that just became blocked or unblocked since last time, and any 🔴→🟡 transitions expected soon (e.g. "S0.7 fleet landing unblocks Ch 5 numbers + Ch 6/7 results").
7. Update the status column in `outline.md` if scoring revealed it is out of date.

## Calibration notes

- Stable-forever content beats exciting content: dataset descriptions, frozen protocols, background mechanics are ideal early targets.
- Related Work is only "ripe" once its feeding reading list is closed for the relevant subsection — check `Master/literature/` reading-list state, don't assume.
- Results sections are never writable from scrutiny evidence alone (standing rule: dedicated thesis-grade runs required).
- Don't recommend more than 3 targets; a ranking of everything is noise.

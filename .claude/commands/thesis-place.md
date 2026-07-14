# /thesis-place — Place protocol entries under the thesis bullets they feed

**What placement IS (Matteo's design, 2026-07-13):** literally putting the vault entry name under the bullet point(s) where it matters, inside `chapters/*.tex`. Nothing more.

- Placement lines are **visible in the compiled PDF** (Matteo's choice, 2026-07-13): use the `\vault{...}` command defined in `main.tex` (gray, scriptsize, handles underscores) — placed inside the `\item`, after its text. Scaffolding; all `\vault` lines die before submission.
- Entry feeds a specific bullet → `\vault{entry-name}` directly under that bullet's text.
- Entry matches a chapter but no existing bullet → a `\vault{...}` line right under the chapter/section heading.
- Multiple entries on one bullet → one line, comma-separated.
- One entry may be placed under multiple bullet points (across sections/chapters) — normal, not an exception.
- Placement is a *conversation*: while placing, new bullet points may pop up as warranted (propose them, Matteo filters) — or not. Both fine.

`Master/thesis/placement_ledger.md` stays as the **triage index** (which entries have been processed, their status) so runs are incremental: **only triage entries not yet in the ledger** — never re-mine the whole vault unless explicitly asked for a full re-audit.

## Bullet & text discipline (Matteo, 2026-07-14)

- **BULLETS ARE MATTEO'S VOICE — HANDS OFF (standing rule).** After the one-time crisping pass below, Claude never edits, shortens, splits, or rewords existing bullet points unless Matteo explicitly asks. He expands them himself; placement only *adds* `\vault{...}` lines under them (and may *propose* new bullets at the ask step — Matteo filters).
- **One-time crisping pass (2026-07-14 run ONLY — DONE, executed 2026-07-14):** shorten every existing bullet to ultra-crisp, one idea per bullet; splitting a bloated bullet into several one-idea bullets is encouraged. **The Introduction is exempt — its bullets stay exactly as they are (Matteo is happy with them).** Complex descriptions/text do not survive in bullets: delegate richer content to the placed protocol entry — if not already present in a vault entry, amend the relevant entry or propose a new one at the ask step; the bullet keeps only the one-idea kernel.
- **Thesis-destined sentences render in italics.** Textual passages explicitly flagged as "supposed to enter the thesis" (candidate sentences, near-verbatim formulations) may be included in the chapter as *cursive* text (`\emph{...}` / `\textit{...}`) — visibly distinct from bullets.
- **LaTeX `%` comments are Claude's scratchspace.** Matteo does not read or interact with them — use them as needed (material maps, directives, status), but do not bloat them; keep them relevant.
- **When a placement is unclear: keep the entry unplaced and ASK.** Collect all uncertain cases during the run and raise them as a batch at the end — never guess a home, never go gung ho (extends the never-force rule).

## Steps

1. **Diff vault against ledger.** List `Master/protocol_vault/*.md` (dated entries only; skip `index.md` and `tags/`). Collect filenames absent from the ledger table. If none: report "ledger up to date, N entries, last triaged <date>" and stop.
2. **Read each new entry fully.** Also read `Master/thesis/outline.md` (section slugs + current status) — placements reference **slugs**, never chapter numbers.
3. **Propose a verdict per entry**, one of:
   - `placed` — one or more section slugs (multi-placement is normal). Note in one phrase *what* the entry contributes there.
   - `not-thesis` — process/infrastructure/bookkeeping. **This is a normal, respectable verdict — never force a placement** (explicit user instruction, 2026-07-13). When in doubt between a weak placement and `not-thesis`, prefer `not-thesis` and say why.
   - `deferred` — thesis-relevant but no ripe home yet (e.g. section doesn't exist yet, or the entry's fate depends on an open decision). Note what it's waiting on.
   - Mixed entries are fine: place the thesis-relevant half, note the process half is ignored.
4. **Flag specially** while reading:
   - **Directives** — entries where Matteo bound how the thesis must present something. Mark `DIRECTIVE` in the note column AND verify the target section's `.tex` file carries a matching comment; add it if missing.
   - **Structure-relevant entries** — anything implying the outline itself should change (new section, reorder, scope shift). Do NOT silently edit `outline.md`; surface these to the user as proposed outline changes.
   - **Supersessions** — entries that correct/amend earlier ones; update the earlier entry's ledger note.
5. **Present the triage to the user before writing** — compact table: entry → proposed verdict → one-line reason. Plain language. Let them adjust. Expect back-and-forth; new bullets proposed here get filtered by Matteo before landing.
6. **Write the placements into the chapter files** (entry names under their bullets, per the design above) and **update the ledger** (append rows in date order). Update the seeded-note line's "last triaged" date.
   - Bullet handling per the **bullet & text discipline** above: normally hands-off (add `\vault` lines only); the one-time crisping applies only in the 2026-07-14 run (Introduction exempt).
7. **Ask-batch.** Present all entries/content that resisted placement (no clear home, ambiguous bullet, content needing a new vault entry) as one batch of short questions. Nothing from this batch gets written until answered.
8. **Report**: how many placed / not-thesis / deferred, which sections got denser (feeds `/thesis-next`), any directives or outline-change flags raised.

## Rules

- Ledger rows are one line each — pointers, not summaries. The entry file remains the source of truth.
- Never delete a row; verdicts change by editing the status (history is in git).
- If `outline.md` slugs changed since last triage, reconcile ledger references first and tell the user.

# /thesis-place — Place protocol entries under the thesis bullets they feed

**Invocation:** `/thesis-place` = incremental (untriaged entries only, the normal case) · `/thesis-place full` = full rebuild from scratch (see below; only when the outline has moved too much for incremental to make sense).

**What placement IS (Matteo's design, 2026-07-13):** literally putting the vault entry name under the bullet point(s) where it matters, inside `chapters/*.tex`. Nothing more.

- Placement lines are **visible in the compiled PDF** (Matteo's choice, 2026-07-13): use the `\vault{...}` command defined in `main.tex` (gray, scriptsize, handles underscores) — placed inside the `\item`, after its text. Scaffolding; all `\vault` lines die before submission.
- Entry feeds a specific bullet → `\vault{entry-name}` directly under that bullet's text.
- Entry matches a chapter but no existing bullet → a `\vault{...}` line right under the chapter/section heading.
- Multiple entries on one bullet → one line, comma-separated.
- One entry may be placed under multiple bullet points (across sections/chapters) — normal, not an exception.
- Placement is a *conversation*: while placing, new bullet points may pop up as warranted (propose them, Matteo filters) — or not. Both fine.

`Master/thesis/placement_ledger.md` stays as the **triage index** (which entries have been processed, their status) so runs are incremental: **only triage entries not yet in the ledger** — never re-mine the whole vault except in the explicit **full rebuild mode** below.

## Bullet & text discipline (Matteo, 2026-07-14)

- **BULLETS ARE MATTEO'S VOICE — HANDS OFF (standing rule).** After the one-time crisping pass below, Claude never edits, shortens, splits, or rewords existing bullet points unless Matteo explicitly asks. He expands them himself; placement only *adds* `\vault{...}` lines under them (and may *propose* new bullets at the ask step — Matteo filters).
- **One-time crisping pass (2026-07-14 run ONLY — DONE, executed 2026-07-14):** shorten every existing bullet to ultra-crisp, one idea per bullet; splitting a bloated bullet into several one-idea bullets is encouraged. **The Introduction is exempt — its bullets stay exactly as they are (Matteo is happy with them).** Complex descriptions/text do not survive in bullets: delegate richer content to the placed protocol entry — if not already present in a vault entry, amend the relevant entry or propose a new one at the ask step; the bullet keeps only the one-idea kernel.
- **Thesis-destined sentences render in italics.** Textual passages explicitly flagged as "supposed to enter the thesis" (candidate sentences, near-verbatim formulations) may be included in the chapter as *cursive* text (`\emph{...}` / `\textit{...}`) — visibly distinct from bullets.
- **LaTeX `%` comments are Claude's scratchspace.** Matteo does not read or interact with them — use them as needed (material maps, directives, status), but do not bloat them; keep them relevant.
- **When a placement is unclear: keep the entry unplaced and ASK.** Collect all uncertain cases during the run and raise them as a batch at the end — never guess a home, never go gung ho (extends the never-force rule).

## Back-tags in the vault entry

Placement also writes **back** into the entry, so the vault side answers "where did this land?" without opening the ledger.

- **Key:** Obsidian-native `tags:` in the entry's YAML frontmatter, added after `phase:`. Plus `placed: <YYYY-MM-DD>`, the date of the run that wrote them.
- **Namespace:** the skill owns exactly the `thesis/*` entries in that list and **nothing else**. Every run removes all `thesis/*` tags from an entry it re-triages and writes the current set fresh; any non-`thesis/` tag is preserved verbatim. The tags therefore always reflect the **most recent invocation**, never an accumulated history.
- **Granularity:** one tag per section slug — `thesis/<slug>` — even when the entry lands on several bullets inside that one section. Bullet-level detail lives in the `\vault{}` lines and the ledger note, not in the tags.
- **Verdicts are tagged too:** `thesis/not-thesis` and `thesis/deferred`, mutually exclusive with slug tags. This makes triage state visible from inside Obsidian, which today only the ledger knows.
- An **untagged dated entry = never triaged**. That is the cheap cross-check against the ledger.

```yaml
---
date: 2026-08-17
time: "14:07"
phase: 4
tags: [thesis/data, thesis/interpretability-tuning]
placed: 2026-08-20
---
```

Tags are bookkeeping, not content: never touch the entry's body, headline, or existing frontmatter keys.

**Backfill (pending):** the ~100 entries triaged before back-tags existed carry none. Backfilling needs no re-triage — the ledger already holds their verdicts, so tags can be generated straight from its rows in one mechanical pass. Do this on request; it is not part of a normal run, and it is not a reason to trigger a full rebuild.

## Steps

1. **Diff vault against ledger.** List `Master/protocol_vault/*.md` (dated entries only; skip `index.md` and `tags/`). Collect filenames absent from the ledger table. **Also re-triage any entry whose ledger slug no longer exists in `outline.md`** — structure drift orphans placements silently, and this catches most of it without needing a full rebuild. If neither set has members: report "ledger up to date, N entries, last triaged <date>" and stop.
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
   - **Write the back-tags** into every entry triaged in this run, per the section above (`thesis/*` replaced, other tags untouched, `placed:` stamped with today's date).
7. **Ask-batch.** Present all entries/content that resisted placement (no clear home, ambiguous bullet, content needing a new vault entry) as one batch of short questions. Nothing from this batch gets written until answered.
8. **Report**: how many placed / not-thesis / deferred, which sections got denser (feeds `/thesis-next`), any directives or outline-change flags raised.

## Full rebuild — `/thesis-place full`

For when the outline has moved so much that incremental placement is patching a map that no longer fits the terrain (large restructure, chapter split/merge, mass slug rename). **Rare by design — reach for it only when the stale-slug re-triage in step 1 is clearly not enough.** It re-mines every dated entry, including ones previously ruled `not-thesis` or `deferred`.

0. **Checkpoint first.** Prompt `━━━ CHECKPOINT ━━━ Consider saving current state before a full placement rebuild. Trigger with /gitcheck` — the run rewrites `chapters/*.tex`, the ledger, and frontmatter across the whole vault.
1. **Reconcile slugs before anything else.** Read `outline.md`, diff its slug set against the ledger's, and report renamed / dropped / new slugs to the user.
2. **Clear the old state.** Strip **all** `\vault{...}` lines from `chapters/*.tex` and all `thesis/*` tags from vault frontmatter. Bullets, prose, `%` comments, and non-`thesis/` tags are untouched — the hands-off rule holds unchanged.
3. **Re-triage every dated entry** through steps 2–4 of the normal flow.
4. **Present the full triage chapter by chapter and wait for confirmation.** Nothing is written until Matteo has walked it. Never-force and ask-batch apply exactly as normal; a long back-and-forth here is expected, not a failure.
5. **Rewrite the ledger wholesale**, carrying the Note column over verbatim wherever the verdict is unchanged. Re-stamp the seeded-note line.
6. **Report the diff against the ledger at git HEAD** — which entries changed verdict, which moved sections, which slugs lost all their entries. That diff is the actual product of a full run: it says what the restructure cost.

## Rules

- Ledger rows are one line each — pointers, not summaries. The entry file remains the source of truth.
- Never delete a row; verdicts change by editing the status (history is in git).
- If `outline.md` slugs changed since last triage, reconcile ledger references first and tell the user.
- **The ledger is authoritative, tags are derived.** On any disagreement between an entry's `thesis/*` tags and its ledger row, the ledger wins and the tags get rewritten — never the other way round.

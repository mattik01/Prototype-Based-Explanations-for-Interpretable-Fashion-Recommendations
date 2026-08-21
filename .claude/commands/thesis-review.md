# /thesis-review <section-slug | free description> — Review pass on Matteo's own AI-free first draft

Review ONE section (or paragraph) of prose that **Matteo wrote himself, without AI tooling**, in `Master/thesis/chapters/*.tex`. Three phases, strictly in order: mechanical pass → flagging pass → improvement discussion. This is the constructive-editor counterpart to `/thesis-attack` (the hostile examiner) — the goal is to make *his* draft better while it stays his.

**Prime directive — the thesis stays Matteo's.** Same as `/thesis-write`: AI-generated prose never enters `chapters/*.tex`. The one deliberate carve-out here is Phase 1: strictly **mechanical** fixes (spelling, grammar slips, LaTeX/formatting) may be applied in-place. Anything that changes wording, meaning, emphasis, or structure is a *finding*, never an edit.

**Prose-only rule** (shared with `/thesis-attack`): bullets are the drafting layer — never review, expand, or delete them. If the target is bullets-only, report "nothing to review yet" and stop. In mixed sections, review the prose, ignore the bullets.

## Phase 0 — Scope + context gathering

1. Resolve the slug via `Master/thesis/outline.md`; read the target chapter `.tex` (headline `[directive: ...]` comments are binding). A free description / pasted paragraph is reviewed as-is, but still load its section's directives.
2. Confirm with Matteo which prose is the fresh draft (usually obvious from git diff of the `.tex` — check it), so pre-existing reviewed prose isn't re-litigated unbidden.
3. Gather ALL planning material for the section — the same sweep as `/thesis-write` Phase A:
   - ledger rows for this slug in `placement_ledger.md` → read those protocol entries;
   - design docs / scrutiny dossiers / literature `_SUMMARY.md`s the outline names;
   - always check `Master/docs/scrutiny/` for a chapter-draft memo covering this section's candidate (e.g. `sc01_chapter_draft.md`) — information only, never a structure/prose template;
   - any Phase C sample for this slug in `Master/thesis/samples/` — used ONLY as a coverage checklist (facts/points the sample carried that the draft may be missing), never as a style or phrasing reference;
   - the agreed paragraph outline (in the `.tex` or from the last `/thesis-write` session) — the reference for the red-line check.

## Phase 1 — Mechanical pass (in-place edits allowed)

Fix directly in the `.tex`, silently accumulating a list:
- spelling and typos;
- unambiguous grammar slips (agreement, articles, tense) — **only where there is exactly one correct fix**; if a repair requires choosing words, it moves to Phase 2;
- LaTeX/formatting: broken or missing `\cite`/`\ref` syntax, math-mode issues, escaping (`&`, `%`, `_`), inconsistent notation macros, spacing (`~` before `\cite`/`\ref`), quotation marks, obvious environment problems;
- terminology drift against `outline.md` / CLAUDE.md decisions (fI-ProtoMF, fU-ProtoMF, ids/noid, attribute vs feature) — only where the intended term is unambiguous; else flag.

Never "improve" a sentence in this phase. When in doubt, flag instead of fix. After the pass, report the full list of applied fixes in plain language (Matteo verifies via git diff). Preserve his voice absolutely — British/American consistency and comma style follow *his* existing usage, not a style guide.

## Phase 2 — Flagging pass (findings only, no edits)

Work at **passage level, not sentence level**. Many real problems live between sentences: a paragraph whose sentences are individually fine but wrongly ordered, two sentences that should be merged, one overloaded sentence that should be split, a missing linking sentence. Findings may therefore be: split this / merge these / reorder within this passage / a sentence doing X is missing here / this paragraph does two jobs and should be divided — not just "this sentence is awkward".

Checks (adapted from `/thesis-attack`, editor's lens rather than examiner's):

1. **Structure & flow** — sentence- and paragraph-level: overloaded or fragmented sentences, wrong ordering within a paragraph, missing transitions, paragraphs doing two jobs, buried topic sentences.
2. **Red line** — does the section's argument thread run unbroken against the agreed outline and the section's job? Flag paragraphs that drift, duplicate, or leave their contribution to the argument implicit; flag outline paragraphs whose job no prose does yet.
3. **Coverage** — facts/points present in the planning material (protocol entries, scrutiny memo, sample, outline bullets) but absent or garbled in the draft. Report as "missing: <fact> (source: <where>)" — Matteo decides whether the omission was deliberate.
4. **Clarity** — sentences/passages a reader would have to reread; ambiguous referents; undefined-at-first-use terms (under the completed-context assumption below — don't flag what an earlier section properly introduces).
5. **Claim–support & evidence boundary** (hard floor, from `/thesis-attack` Layer 1, severity always `blocking`): scrutiny-phase / working-evidence numbers stated as thesis results (only `replication_report.md` currently qualifies as thesis-grade); `\cite` keys absent from `bibliography.bib`; claims attributed to sources that don't say it; prose contradicting a binding directive.
6. **House style** — DBIS norms (no hedging modals, every fact reasoned or cited, opinion only in Conclusion) and the citation source rule (cite only `Master/literature/` works; a claim needing an outside source stays uncited and flagged).
7. **Cross-referencing convention** (from `/thesis-write`): forward-resting claims missing their short "(see Section X.Y)" pointer; backward refs that are narrative recap rather than reuse of a specific definition/notation/result; re-explained material that belongs elsewhere. Review under the **completed-context assumption** — all other sections exist and do their job well.

**Anti-bias rules (inherited verbatim from `/thesis-attack`):** every finding quotes the offending passage; judge substance, not fluency; never imagine results (missing evidence = "needs a thesis-grade run", named); no finding quota — a clean draft is a reportable outcome. One addition for this skill: **never flag Matteo's phrasing merely for not sounding like AI academic English** — plainer, more direct prose than an LLM would produce is a feature, not a finding.

Deliver findings ranked by severity (blocking / major / minor), each: quoted passage → check → problem → one-line fix direction. Then move straight to Phase 3 — the debrief walks through the top findings one at a time, plain language (standing feedback: no terse digests, no walls).

## Phase 3 — Improvement discussion

Talk through the findings that matter most, one at a time. Matteo decides what to act on; disagreements he doesn't concede are dropped or parked as `SQ-N` in `supervisor_questions.md` (ask before writing).

**Rewordings/restructures on request only.** Default output is fix directions. If Matteo explicitly asks "show me a version" for a specific sentence *or passage* (a restructured paragraph, a merged pair, an inserted linking sentence in context), produce it — but it counts as **sample text**: shown in-conversation and, if substantial, saved to `Master/thesis/samples/<slug>_review_vN.tex`; never pasted into `chapters/*.tex`. Voice-calibrate first (his existing prose, sample-vs-final diffs) and obey the LLM-tell watch ("not X, it's Y" contrastives, "isn't just X — it's Y", rhetorical em-dash escalations are forbidden).

## Close the session

- Write the findings report to `Master/thesis/reviews/<slug>_review_vN.md` (same folder + numbering scheme as attack reports; header: date, scope, git state, list of Phase 1 mechanical fixes applied). Reports and any requested sample snippets are **LLM-usage-disclosure artifacts** — never clean them up; copied into `Master/llm_usage/` at finalization.
- Offer to update the section's status line in `outline.md` (e.g. "drafted by M, reviewed v1: 0 blocking / 2 major").
- If the review surfaced a decision-worthy insight, prompt with the standard `━━━ PROTOCOL ━━━` line (ask before writing, per standing feedback).
- Natural follow-up when the draft matures: `/thesis-attack <slug>` for the hostile-examiner pass.

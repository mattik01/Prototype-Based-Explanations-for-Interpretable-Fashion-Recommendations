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
   - **`Master/thesis/style_notes.md` — the style ledger. Load it every session and apply it without re-litigating**: it holds Matteo's revealed preferences (punctuation, coinage rules, terminology hierarchies, his recurring mechanical slips) harvested from past sessions. It is why pass N+1 must never re-discover what pass N settled.
4. Expect a **live co-editing loop**, not a one-shot review: Matteo redrafts between and *during* passes. The rules below (Live-loop protocol) govern every pass after the first.

## Live-loop protocol (learned 2026-08-21, fi-why v1–v8)

- **Re-read the section from disk immediately before every pass and every edit script** — Matteo edits concurrently; stale text is the default assumption, not the exception.
- **Apply edits tolerantly**: exact-match replacements that miss mean he already changed that passage — skip it, keep his version, report what was skipped. Never let one miss abort the rest, and never re-impose a change he overwrote (an overwritten suggestion is a rejected suggestion).
- **Editor-buffer hazard**: he usually has the `.tex` open. After any disk write, tell him to *reload, not save*. If he reports not seeing a change, or pastes text that differs from disk, suspect an unsaved buffer holding his own edits — the recovery is a merge (his paste/buffer as base, re-apply only the disk-side work he approved), never "save and lose" in either direction.
- **Standing formatting**: rewrap the section's source to 78 chars at the end of every pass (paragraph-preserving; placeholder/`%` lines untouched). Mind that a trailing `%` comment must not wrap into uncommented lines.
- **Report weight scales down**: v1 gets the full report; later rounds get slim delta reports (new findings + carried list + decisions). Ceremony must not make extra passes expensive.

## The pass ladder — iterative by design, focus deepens with maturity

The review is MEANT to run as multiple passes over the same section, each focused one level deeper as the draft matures (fi-why took v1–v8; that is the model, not a failure mode). Default focus per rung — Matteo can name a different focus and that overrides:

1. **First contact** (draft raw): mechanical + coverage vs planning material + red line vs the agreed outline. Is the section's job done at all?
2. **Architecture**: argument order, missing/misplaced moves, paragraph jobs. (fi-why: the five-move restructure, vocabulary-before-asymmetry.)
3. **Coherence & terminology**: self-contradictions, layer/altitude consistency, thesis-wide term alignment, mechanism accuracy.
4. **Sentence flow & word choice**: per-paragraph "would you write anything differently" — overloaded sentences, weak verbs, dangling referents, candidate phrasings included.
5. **Polish**: punctuation sweep, register, redundancy dedup, final mechanical.
6. **Convergence check**: clean-verdict pass; output = the residual to-do list (cites, placeholders, title, deferred items), then hand to `/thesis-attack`.

Rules of the ladder: propose the next rung from the draft's state at pass start, don't re-run the whole checklist every time; **settled rungs are only regression-checked, never re-litigated** (decisions live in the report trail + style ledger); a Matteo rewrite deep in the ladder can legitimately re-open an upper rung — say so explicitly ("this reopens architecture") rather than silently mixing levels; carried findings ride along every report until closed or explicitly dropped.

## Phase 1 — Mechanical pass (in-place edits allowed)

Fix directly in the `.tex`, silently accumulating a list:
- spelling and typos;
- unambiguous grammar slips (agreement, articles, tense) — **only where there is exactly one correct fix**; if a repair requires choosing words, it moves to Phase 2;
- LaTeX/formatting: broken or missing `\cite`/`\ref` syntax, math-mode issues, escaping (`&`, `%`, `_`), inconsistent notation macros, spacing (`~` before `\cite`/`\ref`), quotation marks, obvious environment problems;
- terminology drift against `outline.md` / CLAUDE.md decisions (fI-ProtoMF, fU-ProtoMF, ids/noid, attribute vs feature) — only where the intended term is unambiguous; else flag.

Run **Matteo's recurring-slip checklist** from `style_notes.md` first (German commas before restrictive clauses, Lense/lens, doubled words, missing apostrophes, mid-sentence capitalized common nouns, comma splices, hardcoded section numbers) — these are pre-authorized fixes, not judgment calls. His standing punctuation preference (no colons, semicolons, or em-dashes in prose — sentence breaks or commas instead) is likewise applied as a sweep once per section, not re-asked.

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
7. **Mechanism accuracy** — claims about how models or cited papers work are checked against the code and the papers, not just for prose quality (e.g. "names prototypes through clustering" was false: the paper's read-out is most-similar-items inspection; "grounding the prototypes" vs constraining-the-embedding-table). Anchor facts already verified live in `style_notes.md`. A wrong mechanism claim outranks any style finding.
8. **Cross-referencing convention** (from `/thesis-write`): forward-resting claims missing their short "(see Section X.Y)" pointer; backward refs that are narrative recap rather than reuse of a specific definition/notation/result; re-explained material that belongs elsewhere. Review under the **completed-context assumption** — all other sections exist and do their job well.

**Anti-bias rules (inherited verbatim from `/thesis-attack`):** every finding quotes the offending passage; judge substance, not fluency; never imagine results (missing evidence = "needs a thesis-grade run", named); no finding quota — a clean draft is a reportable outcome. One addition for this skill: **never flag Matteo's phrasing merely for not sounding like AI academic English** — plainer, more direct prose than an LLM would produce is a feature, not a finding.

Deliver findings ranked by severity (blocking / major / minor), each: quoted passage → check → problem → one-line fix direction. Then move straight to Phase 3 — the debrief walks through the top findings one at a time, plain language (standing feedback: no terse digests, no walls).

## Phase 3 — Improvement discussion

Talk through the findings that matter most, one at a time. Matteo decides what to act on; disagreements he doesn't concede are dropped or parked as `SQ-N` in `supervisor_questions.md` (ask before writing).

**Apply-on-instruction (Matteo's explicit override of the no-paste default):** when he says "apply", "write it in", or names a scope ("apply P2 to P7") — apply directly, but: (a) only the named scope, nothing that rode along; (b) **smallest possible diffs** — sentence-level surgery, never wholesale paragraph rebuilds (the move-1 rebuild was reverted; the part-two sentence edits survived); (c) log every AI-worded passage in the report's Phase 3 so the disclosure trail stays accurate; (d) expect partial reverts and treat them as data, not setbacks. When he pastes a full section and asks to merge, his paste is the base and only approved disk-side work is re-applied.

**Sample proximity ladder:** samples must start CLOSE to his draft — his skeleton, his phrasings, minimal intervention ("you are straying a bit too far... try a v3 that is a bit closer"). Only widen the distance when he explicitly asks for a freer version. When he asks "how would I say X", give 2–3 candidates plus a stated pick with one line of reasoning — not an essay of options.

**Cite wiring:** when a flagged claim's source exists in `Master/literature/`, don't leave placeholder ping-pong — read the source, verify the claim as stated, add the `bibliography.bib` entry, and wire the `\cite` in one move (the niknazar2024sleep pattern). Report what the source actually supports if it differs from the claim.

**Rewordings/restructures on request only.** Default output is fix directions. If Matteo explicitly asks "show me a version" for a specific sentence *or passage* (a restructured paragraph, a merged pair, an inserted linking sentence in context), produce it — but it counts as **sample text**: shown in-conversation and, if substantial, saved to `Master/thesis/samples/<sec-num>_<slug>/<slug>_review_vN.tex`; never pasted into `chapters/*.tex`. Voice-calibrate first (his existing prose, sample-vs-final diffs) and obey the LLM-tell watch ("not X, it's Y" contrastives, "isn't just X — it's Y", rhetorical em-dash escalations are forbidden).

## Close the session

- Write the findings report to `Master/thesis/reviews/<slug>_review_vN.md` (same folder + numbering scheme as attack reports; header: date, scope, git state, list of Phase 1 mechanical fixes applied). Reports and any requested sample snippets are **LLM-usage-disclosure artifacts** — never clean them up; copied into `Master/llm_usage/` at finalization.
- **Harvest decisions into `Master/thesis/style_notes.md`**: any punctuation/terminology/register preference, mechanism-accuracy anchor, or newly observed recurring slip that surfaced this session gets appended (dated) so no future pass re-discovers it. This is the single biggest accelerator between sessions.
- Offer to update the section's status line in `outline.md` (e.g. "drafted by M, reviewed v1: 0 blocking / 2 major").
- If the review surfaced a decision-worthy insight, prompt with the standard `━━━ PROTOCOL ━━━` line (ask before writing, per standing feedback).
- Natural follow-up when the draft matures: `/thesis-attack <slug>` for the hostile-examiner pass.

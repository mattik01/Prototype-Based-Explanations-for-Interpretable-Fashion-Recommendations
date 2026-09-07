# /thesis-write <section-slug|free description> — Gated co-writing session for one thesis section (or paragraph)

Work on ONE section (or a single paragraph of it) of the thesis in `Master/thesis/`. The session is **gated**: material briefing → scope discussion → agreed paragraph outline → only then, optionally, sample text. Never skip ahead to text generation.

**Prime directive — the thesis stays Matteo's.** Sample texts are discussion material, NEVER pasted into `chapters/*.tex`. Matteo writes his own prose (usually without AI tooling); it is the only text that enters chapter files. He may also write it here in the conversation for feedback — feedback is fine, ghostwriting is not.

## Phase 0 — Blind-mode check (ask FIRST, before any content discussion)

Ask: does he want **blind mode** for this section — writing his own version *before* seeing any AI-generated text? If yes: phases A–B happen normally (material + outline discussion), but phase C (sample text) is locked until he says his draft exists.

## Phase A — Material briefing

1. Resolve the slug via `Master/thesis/outline.md`; read the target chapter `.tex` (headline comments carry directives).
2. Gather ALL placed material: ledger rows for this slug → read those protocol entries; pull the relevant parts of design docs / scrutiny dossiers / literature `_SUMMARY.md`s the outline names. Additionally, always check `Master/docs/scrutiny/` for a chapter-draft memo covering this section's candidate (e.g. `sc01_chapter_draft.md`) even if the outline doesn't name it — **information only**: mine it for findings, caveats, and pointers, never adopt its structure or prose.
3. Brief in **plain language** (Matteo's standing feedback: no terse technical digests): what material exists, what the binding directives demand, what is still missing (pending runs, open decisions). One point at a time if it gets dense.
4. **Style-preference check (light):** if `samples/` holds earlier samples whose sections Matteo has since written himself, skim the sample-vs-final differences for *scope* lessons (what he tends to cut, compress, or expand) — they inform the Phase B outline granularity, not just Phase C voice.
5. **Guard the evidence boundary**: scrutiny-phase numbers are working evidence only. If the section wants a quantitative claim without a dedicated thesis-grade run behind it, say so explicitly and list what run would be needed — never let a scrutiny number drift into thesis prose.

## Phase B — Scope + paragraph outline (discussion gate)

Discuss what belongs in vs. out; then propose a rough paragraph outline (one line per paragraph: its job + its material). Iterate until Matteo agrees. Write the agreed outline into the section's `.tex` file — that IS the deliverable of most sessions.

**Bullet-first convention (Matteo's method):** sections fill up with bullet points (real `itemize` in the section body, visible in the PDF — not `%` comments) before any prose exists. Bullets may come from Matteo directly, from placed protocol entries, or from the agreed outline. Treat in-body bullets as the drafting layer: never delete them as cruft, never expand them into prose unbidden — prose replaces bullets only when Matteo writes it (or explicitly requests sample text in Phase C).

## Phase C — Sample text (only on explicit request; respects blind mode)

1. Granularity as requested: one paragraph or the whole section.
2. **Voice calibration first:** before generating anything, read samples of Matteo's own academic prose — his already-written paragraphs in `chapters/*.tex`, and if more is needed his finished papers in the KB's `academic/` folder — and match that register.
   **Sample-vs-final diffing (the richest calibration signal):** for any section where both a Phase C sample (`samples/<sec-num>_<slug>/<slug>_vN.tex`) and Matteo's own final prose in the chapter exist, compare them and extract his revealed preferences — what he omitted, what he rephrased and how, what he expanded in more detail, which citations/examples he kept or dropped. Apply those learned preferences to the new sample. Sample text should start in Matteo's voice, not default AI academic English.
3. Obey the house style: DBIS norms (no hedging modals, every fact reasoned or cited, opinion only in Conclusion; bonbon scheme at chapter level), terminology from `outline.md`/CLAUDE.md (fI-ProtoMF, fU-ProtoMF, ids/noid arms, ...), `\cite` keys against `bibliography.bib` (add stubs if missing).
   **Citation source rule (Matteo, 2026-08-20):** cite only works from `Master/literature/` — preferably ones already cited in the protocol entries attached to the section. Do NOT hunt for external/"standard" citations to decorate claims; if a claim seems to need a source outside the literature folder, leave it uncited and flag it for Matteo instead.
   **LLM-tell watch (Matteo, 2026-07-13):** characteristic LLM sentence structures — "not X, it's Y" contrastive punchlines, "isn't just X — it's Y", rhetorical em-dash escalations — are unacceptable in thesis prose. Avoid them in sample text; when summarizing arguments destined for the thesis, mark such phrasings as assistant-flavored so Matteo's rewrite pass targets them.
4. Save to `Master/thesis/samples/<sec-num>_<slug>/<slug>_vN.tex` (one folder per section, named `<compiled-number>_<slug>` from the current `main.aux`, e.g. `5.2_fi-model` — Matteo, 2026-08-21; on renumbering, rename the folder, the slug stays the stable key) as **LaTeX** (samples may need formulas, figures/TikZ, tables, and real `\cite` citations — same notation, packages, and `bibliography.bib` keys as the thesis), with a short comment header (date, granularity, outline version it followed). Never write into `chapters/*.tex`.
5. **Notes pack PDF (on request, or alongside any sample):** compile the section's source notes into `samples/<sec-num>_<slug>/<slug>_notes_vN.pdf` — the section's placed protocol entries in the order their `\vault{}` markers appear in the `.tex`, followed by their one-hop `[[linked]]` vault entries (dated entries only; skip hub/tag notes), each under a header naming the entry and its role (placed/linked). Build via pandoc → xelatex with a full-unicode font (DejaVu Serif); keep the intermediate `.md` beside it. This is Matteo's reading pack for writing the section himself.
6. These samples are LLM-usage-disclosure artifacts: they stay, and at thesis finalization get copied into `Master/llm_usage/`. Never clean them up.

## Standing rule — cross-referencing & completed-context assumption (phases B, C, D)

Write (and outline, and review) every section under the assumption that **all other sections — before and after — exist and do their explanatory job well**. No re-explaining material that belongs elsewhere, no "as will be explained later" filler. Cross-references are asymmetric:

- **Forward refs (later sections) = obligations.** A claim that is supported or elaborated only in a *later* numbered section gets a super-short pointer ("(see Section 5.2)") — the pointer is what licenses the claim for a reader who hasn't seen the material yet. In review passes, flag forward-resting claims missing their pointer.
- **Backward refs (earlier sections) = strict exceptions.** Only when the earlier section is genuinely load-bearing for the current point — crisp test: the text *reuses a specific definition, notation, or numerical result* from there. Never for narrative recap or general connective tissue; the reader has already read it, and habitual back-pointers are noise. In review passes, flag backward refs that fail the test.

## Standing procedure — supervisor questions (any phase)

When a decision is hard to settle or genuinely contested (headline variants, structural forks, terminology), do NOT force it: keep the current-best in the text, record the variants + our thoughts as an `SQ-N` item in `Master/thesis/supervisor_questions.md`, mark the spot in the file (visible bullet or comment referencing the SQ number), and move on. Remind Matteo of open SQ items when he mentions an upcoming supervisor meeting. Resolved items move to the register's Resolved section with the outcome.

## Phase D — Close the session

- If Matteo produced/updated his own prose in the `.tex`: offer a **review pass** (correctness of claims, citation gaps, directive compliance, DBIS style, voice — flag phrasings that read assistant-flavored rather than like Matteo's own register) — comments only, no rewriting.
- Update the section's status line in `outline.md` (e.g. "outline agreed", "drafted by M", "reviewed").
- If the discussion produced a decision or insight worth protocolling, prompt with the standard `━━━ PROTOCOL ━━━` line (ask before writing, per standing feedback).

# /thesis-attack <section-slug | "full" | free description> — Adversarial reviewer pass on written thesis prose

Attack finished prose in `Master/thesis/` the way a **thesis examiner / committee member** would (secondary lens: the DBIS supervisor). The job is to find what a hostile-but-fair reader would find — NOT to help, praise, balance, or rewrite. Inspired by ResearchStudio's audit pattern (2026-07-14): named checks with individual verdicts, mandatory quoted evidence, and a hard-floor layer that judgment cannot talk itself out of.

**Prime directive — findings only, never rewrites.** This skill never edits `chapters/*.tex`, never produces replacement prose, never expands or deletes bullets. Every finding names what is wrong and a one-line *fix direction*; the repair is Matteo's.

**Prose-only rule.** Bullets are the drafting layer (bullet-first convention, see `/thesis-write`) — attacking them is premature and forbidden. Attack only real prose paragraphs. A section that is bullets-only is skipped and reported as "not yet attackable". In mixed sections, attack the prose and ignore the bullets.

## Scope resolution

- **Section slug** → resolve via `Master/thesis/outline.md`, read that chapter `.tex` (headline comments carry binding directives).
- **`full`** → all chapters; run per-section checks on every prose-bearing section, plus the coherence check across chapters. Report skipped bullet-only sections in one line.
- **Free description / pasted paragraph** → attack exactly that text; still load its section's directives and the outline's terminology for context.

Before attacking, gather the claim context: `outline.md` (terminology decisions, directives), the section's placed protocol entries via `placement_ledger.md`, `bibliography.bib` (cite-key existence), and `Master/docs/replication_report.md` (the only currently thesis-grade numbers).

## The checks (each verdicted separately: clear / borderline / triggered)

1. **Claim–support** — every factual or quantitative claim traces to a citation or a thesis-grade experiment. Chase each `\cite` key into `bibliography.bib`; chase each number to its source.
2. **Overclaim / scope** — does a sentence claim more than the evidence shows? ("improves interpretability" vs "improves our interpretability metric on hm_1_month"; "the model learns X" vs "consistent with X"). Includes silent generalization across datasets, arms, or seeds.
3. **Defense-question** — for each core claim in scope, name the single nastiest committee question it invites, then verdict whether the text already contains the answer (answered / partially / unanswered). The Kaggle "feature engineering contribution limited" line is a standing example of the genre.
4. **House-style** — DBIS norms (hedging modals, unreasoned facts, opinion outside the Conclusion, bonbon scheme) + the LLM-tell list ("not X, it's Y" contrastives, "isn't just X — it's Y", rhetorical em-dash escalations) + terminology drift against `outline.md`'s decisions (attribute vs feature, model names, ids/noid vocabulary).
5. **Argument-spine** — does each paragraph demonstrably advance the section's job and, through it, the thesis's central argument — or does it merely report? Paragraphs that only recount (what was run, what a paper says) without carrying the argument forward are findings; so are sections whose connection to the central claim is left implicit.
6. **Coherence** (full mode only) — cross-chapter: promises made early and never cashed, notation/terminology inconsistencies between chapters, the same concept introduced twice under different names, forward references to sections that don't say what the reference claims.

## Verdict layers

**Layer 1 — hard floors (mechanical; severity is always `blocking`, no softening):**
- A scrutiny-phase / working-evidence number stated as a thesis result (standing evidence boundary — thesis quotes require dedicated thesis-grade runs; currently only `replication_report.md` qualifies).
- A `\cite` key absent from `bibliography.bib`, or a claim attributed to a source that does not say it (verify against the literature `_SUMMARY.md` when one exists).
- Prose contradicting a binding directive (the `[directive: ...]` comments in the chapter files / outline).

**Layer 2 — soft judgment:** everything else, severity `major` (an examiner would raise it) or `minor` (polish-level). Judgment may weigh severity but may NOT overturn a check's finding — a borderline stays in the report.

## Anti-bias rules (why the attack stays honest)

- **Every finding quotes the offending sentence verbatim.** A finding with no quoted evidence is not allowed — unquoted findings are vibes, and vibes don't survive a rewrite pass.
- **Judge substance, not fluency.** Polished prose most easily hides an unsound step; short crisp text can be flawless. Never let writing quality soften a content finding, and never manufacture findings on rough-but-correct text just to seem thorough.
- **Never imagine results.** If a claim's truth needs an experiment that hasn't run, the finding is "needs a thesis-grade run" (name which), not a guess about the outcome.
- **No finding quota.** Zero findings on a clean section is a legitimate, reportable outcome; padding with minor nitpicks buries the real signal.

## Report

Write to `Master/thesis/reviews/<slug>_attack_vN.md` (`full_attack_vN.md` for full mode; N = existing reports for that scope + 1; create the folder on first use). Header: date, scope, git state of the attacked files. Body: findings ranked most severe first, each as — quoted sentence → check → severity → fix direction (one line). Then per-check verdict summary, skipped sections, and open defense questions.

Reports are **LLM-usage-disclosure artifacts**: never clean them up; at thesis finalization they get copied into `Master/llm_usage/`.

## Close the session

- Debrief in plain language (standing feedback: no terse technical digests): the two or three findings that matter most, one at a time — not a wall of all of them.
- If Matteo disagrees with a finding and it stays genuinely contested, don't force it: propose parking it as an `SQ-N` item in `Master/thesis/supervisor_questions.md` (standard procedure; ask before writing).
- Offer to update the section's status line in `outline.md` (e.g. "attacked v2: 1 blocking / 3 major").
- If the attack surfaced a decision-worthy insight, prompt with the standard `━━━ PROTOCOL ━━━` line (ask before writing, per standing feedback).

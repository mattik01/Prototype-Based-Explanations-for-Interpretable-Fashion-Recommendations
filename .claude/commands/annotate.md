Produce an ultra-compact PDF margin annotation answering a question from the current conversation context (or from $ARGUMENTS if given).

Rules — all mandatory:
1. **Source:** Answer from what is already established in the conversation; only do a quick code/paper lookup if a fact is missing. No new analysis beyond what the annotation needs.
2. **Brutally short.** Every word must earn its place. Target the absolute minimum that still preserves all distinct content points — compress, don't omit. If the first draft can be halved without losing a content point, halve it.
3. **Line width — two formats, user picks per call.** The user may say `thin`/`narrow` or `wide` in $ARGUMENTS (e.g. `/annotate wide <topic>`):
   - **thin** (default if unspecified): hard-wrap at **13–15 characters per line** whenever possible — that is the width that fits a PDF side margin without overlapping the paper text. Going longer is allowed when a word won't fit or when strict wrapping would make the annotation excessively tall or unreadable. **Never hyphenate/break a word across lines — exceed the limit instead.** Also never leave a line meaningfully below 13 chars (e.g. a single short orphan word) just to avoid overstepping on the previous line; balance the wrap so lines stay near the target.
   - **wide**: hard-wrap at ~80 characters per line — for annotations placed above/below the text or in wide margins. Same total brevity; only the wrapping changes.
   No indentation — every line starts at column 0 in both formats.
4. **No empty lines. Ever.** One contiguous block of text.
5. **No equations or math notation** unless the user explicitly asks for them.
6. **Plain text only** — no markdown bold/italics (PDF annotation tools render them as literal asterisks). **No bullet points, no `-`/`--` list markers, no arrows (`→`).** Structure: first line = topic label ending with a colon; then one point per sentence, each point's super-short motivation attached via `: `; optional final sentence for a cross-cutting takeaway. Points flow as wrapped text, separated by sentence periods.
7. Abbreviations, symbols (≥, vs., `:`), and dropped articles are encouraged.
8. **Output the annotation inside a fenced code block** so line breaks survive copy-paste, with no commentary before or after beyond one short sentence if truly needed.

User-confirmed gold example (thin format) — match this style exactly:

```
ProtoMF penalties
(loss terms,
tuned weights):
R_batch: each
user/item near
≥1 prototype, else
unexplainable
entities. R_proto:
each prototype
near ≥1 real
entity, else dead
prototypes.
Variants: max,
soft (sharper),
incl (balanced,
anti-collapse).
Same reg, opposite
axes. Pure
interpretability
(Rudin), not
accuracy.
```

Create a protocol entry in the Obsidian vault at `Master/protocol_vault/`.

**Scope:** Only log things relevant to the master thesis — research decisions, architecture choices, experiment results, dataset insights, methodology reasoning. Skip pure infrastructure/tooling steps (e.g., SSH setup, GPU config) unless they impact the thesis content.

Steps:
1. Get the current date and time via `date +"%Y-%m-%d"` and `date +"%H%M"`.
2. Ask the user a brief clarifying question if the topic or decision context is unclear. Keep it short. (if necessary only)
3. Create the entry note at `Master/protocol_vault/YYYY-MM-DD_HHMM_short-slug.md` with this format:

```
---
date: YYYY-MM-DD
time: "HH:MM"
phase: <current phase number>
---
# <Short descriptive title>

<For routine decisions: 1-3 sentences — what was done, context, decisions made and their reasoning.
For explanations the user praised or worked through at a gate: see the verbatim-capture rule below.>

[[tag1]] [[tag2]] [[tag3]]
```

4. For tags: reuse existing tags from `Master/protocol_vault/tags/` whenever possible. Only introduce new ones if nothing existing fits. Check the `index.md` for the current tag list.
5. If a new tag is needed, create `Master/protocol_vault/tags/<tag-name>.md` with just `# <tag-name>` and a one-line description.
6. Update `Master/protocol_vault/index.md` to include any new tags.
7. **Conciseness is subordinate to fidelity — the verbatim-capture rule (added 2026-07-11, user directive):**
   Routine decision entries stay concise. BUT when an entry records an explanation
   that was worked through interactively and explicitly praised or deemed valuable
   (gate discussions, "this is gold", thesis-framing moments), do NOT compress it
   into jargon shorthand — jargonified summaries cannot reliably be decoded back
   into the explanation later, and the decoding loses information. Instead:
   - Capture the explanation **close to literally** as it was given in conversation —
     the plain-language build-up, the concrete examples, the analogies, the
     one-liner punchlines. Long entries are explicitly fine for this.
   - Write for a reader (future Matteo, or thesis-drafting Claude) who does NOT
     have the conversation in context and must reconstruct the full argument from
     the entry alone.
   - Test before saving: could this entry be pasted into a thesis draft session
     and regenerate the argument at full strength? If not, expand it.
   - Jargon terms may appear, but only alongside their plain-language rendering,
     never as a substitute for it.

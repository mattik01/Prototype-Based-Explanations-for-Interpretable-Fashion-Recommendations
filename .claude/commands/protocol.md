Create a protocol entry in the Obsidian vault at `Master/protocol_vault/`.

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

<1-3 sentences: what was done, context, and especially any decisions made and their reasoning.>

[[tag1]] [[tag2]] [[tag3]]
```

4. For tags: reuse existing tags from `Master/protocol_vault/tags/` whenever possible. Only introduce new ones if nothing existing fits. Check the `index.md` for the current tag list.
5. If a new tag is needed, create `Master/protocol_vault/tags/<tag-name>.md` with just `# <tag-name>` and a one-line description.
6. Update `Master/protocol_vault/index.md` to include any new tags.
7. Keep entries extremely concise. Prioritize decisions and their reasoning.

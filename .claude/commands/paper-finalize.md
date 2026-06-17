Produce a brief, thesis-relevant mini-summary of a paper — built from its in-PDF annotations + related protocol entries — saved as a markdown file beside the PDF.

Purpose: when a paper has been read (usually **annotated** in a real PDF tool — Adobe/Edge — that saves highlights & comments into the file), distill it into a one-screen, citable, thesis-focused note so the reading pays off later. The annotations mark what the user found important/citable; the protocol entries hold the insights that came out of the reading. Fold both in.

Target file: `<same folder as the PDF>/<pdf_basename>_SUMMARY.md`.

## Procedure

1. **Resolve the paper.** From $ARGUMENTS (a path, an ID like `S1`, or a title fragment) locate the PDF under `Master/literature/papers/`. If ambiguous, list candidates and ask. Note its folder, basename, and library ID (e.g. `S1`, `C6`).

2. **Extract the user's annotations from the PDF — this is the priority input.** Prefer PyMuPDF (`fitz`) if importable, which gives both the highlighted text and the typed comment; fall back to `pypdf` (comments + presence) if not. Example (adapt at runtime, handle errors gracefully):
   ```python
   # PyMuPDF preferred — highlighted text + comments
   import fitz
   doc = fitz.open(PATH)
   for pno, page in enumerate(doc, 1):
       for a in (page.annots() or []):
           t = a.type[1]                      # 'Highlight','Text','FreeText',...
           note = a.info.get('content','')     # user's typed comment
           quad = a.vertices
           hl = ''
           if quad and t == 'Highlight':
               for i in range(0, len(quad), 4):
                   r = fitz.Quad(quad[i:i+4]).rect
                   hl += page.get_textbox(r) + ' '
           print(pno, t, repr(hl.strip()), '||', repr(note))
   ```
   ```python
   # pypdf fallback — at least subtype + typed /Contents
   from pypdf import PdfReader
   for pno, page in enumerate(PdfReader(PATH).pages, 1):
       for ref in (page.get('/Annots') or []):
           o = ref.get_object(); st = str(o.get('/Subtype'))
           if st == '/Link': continue          # publisher hyperlinks — ignore
           print(pno, st, repr(str(o.get('/Contents',''))))
   ```
   Collect the highlighted passages and comments. These are the user's own citable signals — weight them heavily. **But annotations are spontaneous** — written as the user reads, without knowledge of later passages — so an early note (even one saying "must record") can be **superseded by what the full paper or the subsequent discussion reveals**. Weigh each against the complete reading; surface, don't blindly enshrine. If the PDF has **no** user annotations (only `/Link`), say so and proceed from protocol entries + a targeted skim.

3. **Gather related protocol entries.** Grep `Master/protocol_vault/` for the paper's author/ID/key terms and for entries written during its reading session. List the matches by filename; you'll link them.

4. **Targeted read, not cover-to-cover.** Use the annotations + protocol entries to identify the thesis-relevant parts; skim only those in the PDF to get quotes/specifics right. Do not read the whole paper if the annotations already point at what mattered — keep cost low.

5. **Write the summary** at `<folder>/<basename>_SUMMARY.md` using the template below. **Brevity is the hard requirement — target one screen.** A long summary is useless; compress, don't pad. Drop any section that would be empty.

6. **Confirm** with the path. If the annotations surfaced a citable point or insight **not** yet in the protocol vault, name it and offer to `/protocol` it.

## Template

```markdown
# <ID> — <Authors Year, "Title">

Mini-summary, thesis-relevant only. <venue/type>. PDF in this folder. Insights in `protocol_vault/`.

## One line
<the paper's thesis in one sentence>

## Why it matters to us
- <how it connects to ProtoMF / the thesis; link the positioning protocol entry>

## Most citable for (★ = strongest)
- ★ <strongest citable claim — usually what the user highlighted; quote briefly>
- <other citable points; note companion papers if any>

## Coverage map (relevance-flagged)   ← optional, only if the paper has clear parts/sections
<one compact line listing the paper's main parts, bolding the ones relevant to us>

## Our insights from reading it (protocol links)
- <insight> — [[<full protocol basename, no .md>]]

## Caveats when citing
- <scope mismatch, overclaim risks, transfer-of-setting notes>
```

## Rules
- Plain readable markdown; this is a reference note, not a margin annotation (the brevity discipline of `/annotate` applies to length, not to line width — write normal prose bullets).
- Protocol links use the **full basename** `[[YYYY-MM-DD_HHMM_slug]]` so they resolve in Obsidian.
- Never invent citable claims — every ★ point must trace to the paper text or a real annotation.
- One paper per invocation.

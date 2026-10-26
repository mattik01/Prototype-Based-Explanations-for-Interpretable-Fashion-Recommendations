#!/usr/bin/env python3
"""Copy each placed vault note into the thesis chapters as a LaTeX comment block.

For every ``\\vault{name[, name...]}`` link in ``Master/thesis/chapters/*.tex``
the full body of ``Master/protocol_vault/<name>.md`` (YAML frontmatter dropped)
is written as ``%|``-prefixed comment lines directly below the line holding the
link, one block per note:

    % region vault: <name> -- auto-copy, edit the vault entry, not this block
    %| <note body>
    % endregion vault: <name>

The ``% region`` / ``% endregion`` markers sit at column 0 so LaTeX Workshop
can fold each block (its folding provider only matches them there). The PDF is
unchanged: every inserted line is a comment.

The script owns these blocks and nothing else. Every run strips all existing
blocks and regenerates them from the current vault, so copies never go stale
and a removed ``\\vault`` link loses its block. Called by /thesis-place
(step 6, full rebuild steps 2 and 5); safe to run by hand at any time.

Usage:
    python Master/scripts/thesis_vault_copies.py           # strip + regenerate
    python Master/scripts/thesis_vault_copies.py --strip   # remove all blocks
    python Master/scripts/thesis_vault_copies.py --check   # dry run, exit 1 if stale
"""

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHAPTERS = REPO / "Master" / "thesis" / "chapters"
VAULT = REPO / "Master" / "protocol_vault"

START = "% region vault: "
END = "% endregion vault: "
BODY = "%|"
VAULT_LINK = re.compile(r"\\vault\{([^}]*)\}")
UNESCAPED_PCT = re.compile(r"(?<!\\)%")
# LaTeX Workshop's folding provider matches \begin{ / \end{ / \begingroup /
# \endgroup anywhere, comments included; defuse them inside copied text.
FOLD_TOKENS = [("\\begin{", "\\begin {"), ("\\end{", "\\end {"),
               ("\\begingroup", "\\begin group"), ("\\endgroup", "\\end group")]


def strip_blocks(lines, path):
    out, inside = [], None
    for ln, line in enumerate(lines, 1):
        if inside is None and line.startswith(START):
            inside = ln
        elif inside is not None:
            if line.startswith(END):
                inside = None
        else:
            out.append(line)
    if inside is not None:
        sys.exit(f"{path.name}:{inside}: unterminated vault block, file left untouched")
    return out


def note_body(name):
    text = (VAULT / f"{name}.md").read_text(encoding="utf-8")
    if text.startswith("---\n"):
        close = text.find("\n---\n", 4)
        if close != -1:
            text = text[close + 5:]
    for old, new in FOLD_TOKENS:
        text = text.replace(old, new)
    return text.strip("\n").splitlines()


def block(name):
    rows = [f"{START}{name} -- auto-copy, edit the vault entry, not this block"]
    rows += [f"{BODY} {row}".rstrip() for row in note_body(name)]
    rows.append(f"{END}{name}")
    return [row + "\n" for row in rows]


def linked_names(line):
    code = UNESCAPED_PCT.split(line, 1)[0]
    names = []
    for match in VAULT_LINK.finditer(code):
        names += [n.strip().replace("\\_", "_") for n in match.group(1).split(",") if n.strip()]
    return names


def process(path, strip_only, missing, counts):
    original = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines = strip_blocks(original, path)
    if not strip_only:
        out = []
        for line in lines:
            out.append(line if line.endswith("\n") else line + "\n")
            for name in linked_names(line):
                if (VAULT / f"{name}.md").exists():
                    out += block(name)
                    counts["copied"] += 1
                else:
                    missing.append(f"{path.name}: {name}")
        lines = out
    return original, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--strip", action="store_true", help="remove all vault blocks")
    parser.add_argument("--check", action="store_true", help="dry run; exit 1 if any file would change")
    args = parser.parse_args()

    missing, counts, changed = [], {"copied": 0}, []
    for path in sorted(CHAPTERS.glob("*.tex")):
        original, lines = process(path, args.strip, missing, counts)
        if lines != original:
            changed.append(path.name)
            if not args.check:
                path.write_text("".join(lines), encoding="utf-8")

    verb = "would change" if args.check else "changed"
    action = "stripped" if args.strip else f"{counts['copied']} note copies"
    print(f"{action}; {len(changed)} file(s) {verb}: {', '.join(changed) or '-'}")
    for item in missing:
        print(f"MISSING vault note (link kept, no copy): {item}")
    sys.exit(1 if (args.check and changed) else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert a LaTeX file to plain text suitable for TTS synthesis.

- Extracts \\chapter{}, \\section{}, \\subsection{} titles as spoken lines.
- Strips LaTeX commands while keeping their inner text.
- Removes comments, environments like verbatim/figure/table.
- Collapses excess whitespace and blank lines.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


# Environments whose content should be dropped entirely
_DROP_ENVS = {
    "verbatim", "lstlisting", "minted", "figure", "table",
    "tabular", "align", "equation", "math", "tikzpicture",
}


def _drop_environments(src: str) -> str:
    for env in _DROP_ENVS:
        src = re.sub(
            r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}",
            "", src, flags=re.DOTALL,
        )
    return src


def _strip_comments(src: str) -> str:
    # Remove % comments (not escaped \%)
    return re.sub(r"(?<!\\)%.*", "", src)


def _inline_commands(src: str) -> str:
    # Keep inner text of formatting commands
    for _ in range(4):  # iterate to handle nesting
        src = re.sub(
            r"\\(?:textbf|textit|emph|underline|texttt|textrm|textsf|"
            r"textsc|textup|textsl|mbox|hbox|vbox|fbox|"
            r"footnote|marginpar)\{([^{}]*)\}",
            r"\1", src,
        )
    return src


def _section_headings(src: str) -> str:
    # Convert sectioning commands to plain text lines
    src = re.sub(
        r"\\(?:chapter|section|subsection|subsubsection|paragraph)\*?\{([^{}]*)\}",
        r"\n\n\1.\n\n", src,
    )
    return src


def _drop_remaining_commands(src: str) -> str:
    # Drop commands with arguments (keep argument text for simple ones)
    src = re.sub(r"\\(?:label|ref|cite|input|include|includegraphics|url|href)\{[^{}]*\}", "", src)
    # Drop preamble-style commands
    src = re.sub(r"\\(?:documentclass|usepackage|setlength|setcounter|newcommand|renewcommand|def)\b[^\n]*", "", src)
    # Drop \begin / \end for remaining environments (keep content)
    src = re.sub(r"\\(?:begin|end)\{[^{}]*\}", "", src)
    # Drop any remaining \command (with or without braces)
    src = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", "", src)
    # Drop lone backslash-newline (line break command)
    src = re.sub(r"\\\\", " ", src)
    return src


def _normalize_whitespace(src: str) -> str:
    # Collapse inline whitespace
    src = re.sub(r"[ \t]+", " ", src)
    # Trim trailing spaces per line
    lines = [ln.rstrip() for ln in src.splitlines()]
    # Collapse runs of more than two blank lines into one blank line
    out: list[str] = []
    blank = 0
    for ln in lines:
        if ln == "":
            blank += 1
            if blank <= 1 and out:
                out.append("")
        else:
            blank = 0
            out.append(ln)
    return "\n".join(out).strip()


def clean(src: str) -> str:
    src = _strip_comments(src)
    src = _drop_environments(src)
    src = _inline_commands(src)
    src = _section_headings(src)
    src = _drop_remaining_commands(src)
    src = _normalize_whitespace(src)
    return src + "\n"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: clean_tex.py <file.tex>", file=sys.stderr)
        return 2
    src = Path(sys.argv[1]).read_text(encoding="utf-8")
    sys.stdout.write(clean(src))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

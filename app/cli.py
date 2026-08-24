"""Command-line access to the parser.

Useful for inspecting a questionnaire without starting the server, and for
piping the document model into the later pipeline stages.

    python -m app.cli questionnaire.docx --summary
    python -m app.cli questionnaire.docx > parsed.json
"""

from __future__ import annotations

import argparse
import sys

from app.models.document import ParagraphBlock, ParsedDocument, TableBlock
from app.parsing.docx_parser import DocxParseError, parse_docx


def summarise(parsed: ParsedDocument) -> str:
    lines = [
        f"File:       {parsed.source_filename}",
        f"Questions:  {parsed.stats.questions}",
        f"Paragraphs: {parsed.stats.non_empty_paragraphs} non-empty "
        f"({parsed.stats.paragraphs} total)",
        f"Tables:     {parsed.stats.tables}",
        f"Runs:       {parsed.stats.runs} "
        f"({parsed.stats.bold_runs} bold, {parsed.stats.italic_runs} italic)",
    ]

    for warning in parsed.warnings:
        lines.append(f"WARNING:    {warning}")

    lines.append("")
    for question in parsed.questions:
        label = question.label or "(preamble)"
        lines.append(f"{label:<10} {question.title_text}")

        for block in parsed.blocks_for(question):
            if block.index == question.title_block_index:
                continue
            if isinstance(block, TableBlock):
                lines.append(f"{'':<10}   [table {block.n_rows}x{block.n_cols}]")
            elif isinstance(block, ParagraphBlock) and block.text:
                marker = (block.list_info.marker if block.list_info else None) or ""
                prefix = f"{marker} " if marker else ""
                lines.append(f"{'':<10}   {prefix}{block.text}")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli", description="Parse a DOCX questionnaire."
    )
    parser.add_argument("path", help="Path to the .docx questionnaire")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a readable outline instead of JSON",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2)")
    args = parser.parse_args(argv)

    try:
        parsed = parse_docx(args.path)
    except DocxParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.summary:
        print(summarise(parsed))
    else:
        print(parsed.model_dump_json(indent=args.indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

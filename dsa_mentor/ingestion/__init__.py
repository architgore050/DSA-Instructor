"""Ingestion sub-package — parsers for .md, .txt, .pdf corpus sources.

Each parser returns a list of Paragraph nodes with full provenance metadata.
"""

from dsa_mentor.ingestion.md_parser import parse_paragraphs as parse_md_paragraphs
from dsa_mentor.ingestion.txt_parser import parse_paragraphs as parse_txt_paragraphs
from dsa_mentor.ingestion.pdf_parser import parse_paragraphs as parse_pdf_paragraphs

__all__ = [
    "parse_md_paragraphs",
    "parse_txt_paragraphs",
    "parse_pdf_paragraphs",
]

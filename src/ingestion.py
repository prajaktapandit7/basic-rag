from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


@dataclass(frozen=True)
class ExtractedPage:
    source: str
    page: int
    text: str


def extract_pdf_pages(data_dir: str | Path) -> list[ExtractedPage]:
    data_path = Path(data_dir)
    pages: list[ExtractedPage] = []

    for pdf_path in sorted(data_path.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            pages.append(ExtractedPage(source=pdf_path.name, page=i + 1, text=text))

    return pages


def smart_chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to avoid ending mid-word: back up to last space/newline near the end.
        if end < len(text):
            window_start = max(start, end - 100)
            search_area = text[window_start:end]
            last_space = max(search_area.rfind(" "), search_area.rfind("\n"), search_area.rfind("\t"))
            if last_space != -1:
                end = window_start + last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        next_start = max(0, end - overlap)

        # Avoid starting mid-word: move to the next whitespace boundary after next_start.
        if next_start < len(text) and not text[next_start].isspace():
            while next_start < len(text) and not text[next_start].isspace():
                next_start += 1
            while next_start < len(text) and text[next_start].isspace():
                next_start += 1

        start = max(start + 1, next_start) if end <= start else next_start

    return chunks


def chunk_pages(
    pages: Iterable[ExtractedPage],
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Returns a list of dicts compatible with your notebook pipeline:
      {"content": "...", "source": "Doc1.pdf", "page": 1, "chunk_index": 0}
    """
    all_tagged_chunks: list[dict] = []
    for p in pages:
        chunks = smart_chunk_text(p.text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            all_tagged_chunks.append(
                {
                    "content": chunk,
                    "source": p.source,
                    "page": p.page,
                    "chunk_index": idx,
                }
            )
    return all_tagged_chunks


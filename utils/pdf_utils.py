import re
from pypdf import PdfReader


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(file_obj):
    reader = PdfReader(file_obj)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = clean_text(text)
        if text:
            pages.append({
                "page_number": i + 1,
                "text": text
            })

    full_text = "\n\n".join(page["text"] for page in pages)
    return full_text, pages


def chunk_pages(pages, chunk_size=1200, overlap=200):
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        start = 0
        step = max(chunk_size - overlap, 1)

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "text": chunk_text
                })
                chunk_id += 1

            start += step

    return chunks


def build_paper_context(full_text: str, max_chars: int = 18000) -> str:
    if len(full_text) <= max_chars:
        return full_text

    first_part = full_text[:7000]
    middle_start = max((len(full_text) // 2) - 2000, 0)
    middle_part = full_text[middle_start:middle_start + 4000]
    last_part = full_text[-7000:]

    context = (
        "=== BEGINNING OF PAPER ===\n"
        + first_part
        + "\n\n=== MIDDLE OF PAPER ===\n"
        + middle_part
        + "\n\n=== END OF PAPER ===\n"
        + last_part
    )
    return context
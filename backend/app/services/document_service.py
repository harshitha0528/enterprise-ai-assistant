from pathlib import Path

from pypdf import PdfReader
from docx import Document


def extract_text(file_path: str) -> str:
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        reader = PdfReader(file_path)
        # "\n" between pages so the last word of a page is not glued to the first word of the next
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if extension == ".docx":
        document = Document(file_path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    if extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    raise ValueError("Unsupported document format")


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
           break
        start += chunk_size - overlap

    return chunks

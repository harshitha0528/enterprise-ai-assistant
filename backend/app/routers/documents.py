from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.document_service import chunk_text, extract_text
from app.services.rag_service import answer_question
from app.services.vector_store import store_chunks,collection,search_similar_chunks

router = APIRouter(prefix="/documents", tags=["Document Intelligence"])

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


@router.get("/")
def get_documents():
    return {"message": "Document Intelligence module is ready"}


# Plain `def` (not `async def`): FastAPI runs it in a thread pool, so the slow
# OpenAI calls inside do not freeze the whole server.
@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Only PDF, DOCX and TXT files are allowed")

    content = file.file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File size cannot exceed 10 MB")

    stored_filename = f"{uuid4().hex}{extension}"
    file_path = UPLOAD_DIR / stored_filename
    file_path.write_bytes(content)

    text = extract_text(str(file_path))
    if not text.strip():
        file_path.unlink()  # nothing to index (e.g. a scanned PDF without a text layer)
        raise HTTPException(422, "No readable text found. Scanned PDFs need OCR.")

    chunks = chunk_text(text)
    try:
        stored = store_chunks(chunks, stored_filename, file.filename or stored_filename)
    except Exception as error:
        file_path.unlink(missing_ok=True)  # do not keep files that failed to be indexed
        raise HTTPException(500, f"Indexing failed: {error}")

    return {
        "message": "Document uploaded successfully",
        "original_filename": file.filename,
        "stored_filename": stored_filename,
        "file_size": len(content),
        "extracted_characters": len(text),
        "chunks_stored": stored,
    }


@router.post("/ask")
def ask_question(request: QuestionRequest):
    return answer_question(request.question)

@router.post("/debug-search")
def debug_search(request: QuestionRequest):
    results = search_similar_chunks(request.question, n_results=5)
    return {
        "total_chunks_in_db": collection.count(),
        "matches": [
            {
                "distance": round(dist, 3),
                "file": meta["original_filename"],
                "preview": doc[:150],
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ],
    }
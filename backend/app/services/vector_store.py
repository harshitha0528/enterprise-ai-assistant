from pathlib import Path

import chromadb

from app.services.embedding_service import generate_embedding, generate_embeddings

CHROMA_DIR = Path(__file__).resolve().parents[1] / "storage" / "chroma_db"

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# New collection name: these vectors have 384 dims, older Ollama (768) and OpenAI (1536) ones cannot be mixed in
collection = client.get_or_create_collection(
    name="enterprise_documents_local",
    metadata={"hnsw:space": "cosine"},
)


def store_chunks(chunks: list[str], stored_filename: str, original_filename: str) -> int:
    if not chunks:
        return 0

    embeddings = generate_embeddings(chunks)  # one batched call instead of one call per chunk

    collection.add(
        ids=[f"{stored_filename}_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "stored_filename": stored_filename,
                "original_filename": original_filename,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ],
    )
    return len(chunks)


def search_similar_chunks(query: str, n_results: int = 4):
    query_embedding = generate_embedding(query)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

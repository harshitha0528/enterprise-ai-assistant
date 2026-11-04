from app.config import RAG_MAX_DISTANCE, RAG_TOP_K
from app.services.llm_service import generate_answer
from app.services.vector_store import search_similar_chunks

NOT_FOUND = "I could not find this information in the uploaded documents."


def answer_question(question: str) -> dict:
    # 1. Retrieve: find the chunks closest to the question
    results = search_similar_chunks(question, n_results=RAG_TOP_K)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # 2. Filter: drop chunks that are not related enough
    relevant = [
        (doc, meta, dist)
        for doc, meta, dist in zip(documents, metadatas, distances)
        if dist <= RAG_MAX_DISTANCE
    ]

    # No relevant evidence -> do not even call the LLM (saves cost and prevents invented answers)
    if not relevant:
        return {"answer": NOT_FOUND, "sources": []}

    # 3. Augment: build the context that the LLM will read
    context = "\n\n".join(
        f"[Source {i}: {meta['original_filename']}, chunk {meta['chunk_index']}]\n{doc}"
        for i, (doc, meta, _) in enumerate(relevant, start=1)
    )

    # 4. Generate
    answer = generate_answer(question, context)

    sources = [
        {
            "source_number": i,
            "filename": meta["original_filename"],
            "chunk_index": meta["chunk_index"],
            "distance": round(dist, 4),
            "preview": doc[:200],
        }
        for i, (doc, meta, dist) in enumerate(relevant, start=1)
    ]

    return {"answer": answer, "sources": sources}

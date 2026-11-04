from app.services.vector_store import search_similar_chunks

# Ask a question about the uploaded document
question = "What are the key achievements mentioned in the monthly report?"

# Search ChromaDB
results = search_similar_chunks(question)

# Display the results
for index, chunk in enumerate(results["documents"][0]):

    print(f"\nResult {index + 1}")
    print("Document:", results["metadatas"][0][index]["filename"])
    print("Chunk:", chunk)
    print("Distance:", results["distances"][0][index])
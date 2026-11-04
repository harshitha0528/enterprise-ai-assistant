from app.services.embedding_service import (
    generate_embedding,
    cosine_similarity
)

# Our employee's question
question = "What vacation benefits do employees receive?"

# Two sentences from different documents
document_1 = "Employees are entitled to annual leave."
document_2 = "The server database connection failed."

# Generate embeddings
question_embedding = generate_embedding(question)
document_1_embedding = generate_embedding(document_1)
document_2_embedding = generate_embedding(document_2)

# Compare similarities
score_1 = cosine_similarity(
    question_embedding,
    document_1_embedding
)

score_2 = cosine_similarity(
    question_embedding,
    document_2_embedding
)

print("Question:", question)
print("\nAnnual leave similarity:", score_1)
print("Database error similarity:", score_2)

if score_1 > score_2:
    print("\nResult: Annual leave document is more relevant!")
else:
    print("\nResult: Database error document is more relevant!")
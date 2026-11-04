from app.services.document_Service import extract_text, chunk_text
from app.services.vector_store import store_chunks, collection

file_path = "app/storage/uploads/b69cbc4e75394da5bf02e621e85aed6f.docx"

# Extract text from the document
text = extract_text(file_path)

# Split the text into chunks
chunks = chunk_text(text)

# Store chunks in ChromaDB
stored_count = store_chunks(chunks, "monthly_achievement_report.docx")

print("Chunks stored:", stored_count)
print("Total records in ChromaDB:", collection.count())
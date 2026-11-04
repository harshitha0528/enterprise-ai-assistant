from app.config import LLM_MODEL
from app.services.llm_client import get_client

SYSTEM_PROMPT = """You are an enterprise assistant that answers questions about company documents.

Rules:
1. Answer ONLY using the context provided below the question.
2. If the context does not contain the answer, reply exactly:
   "I could not find this information in the uploaded documents."
3. Never invent policies, numbers, names or procedures.
4. Be concise. When useful, mention which source supports your answer, e.g. [Source 2].
"""


def generate_answer(question: str, context: str) -> str:
    response = get_client().chat.completions.create(
        model=LLM_MODEL,
        temperature=0,  # factual answers, no creativity
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content

import math
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once (first run downloads it, about 90 MB)."""
    return SentenceTransformer(EMBEDDING_MODEL)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Convert many texts into vectors, locally."""
    vectors = get_model().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def generate_embedding(text: str) -> list[float]:
    return generate_embeddings([text])[0]


def cosine_similarity(vector_a: list, vector_b: list) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)

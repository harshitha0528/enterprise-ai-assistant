from openai import OpenAI

from app.config import GROQ_API_KEY, LLM_BASE_URL

_client = None


def get_client() -> OpenAI:
    """Create the LLM client once. Same SDK as OpenAI, pointed at Groq."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to the .env file in the project root."
            )
        _client = OpenAI(api_key=GROQ_API_KEY, base_url=LLM_BASE_URL)
    return _client

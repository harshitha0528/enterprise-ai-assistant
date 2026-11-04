import os
from pathlib import Path
from dotenv import load_dotenv

# Project root = enterprise-ai-assistant/ (where .env lives)
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

# Application settings
APP_NAME = os.getenv("APP_NAME", "Enterprise AI Assistant")
APP_ENV = os.getenv("APP_ENV", "development")

# LLM settings (Groq speaks the OpenAI protocol, so we use the openai SDK with a different base_url)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")  # check console.groq.com/docs/models

# Embedding settings (runs locally, no API key needed)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# RAG settings
# Chroma cosine distance: 0 = identical, larger = less related.
# Chunks farther than this are ignored. Tune it using test_search.py.
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.65"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

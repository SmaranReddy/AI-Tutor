# backend/config.py
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file

# --- API KEYS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment/.env")

# You’re now using HuggingFace for embeddings (no key needed if using free models),
# but if you later add one:
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", None)

# --- Model names ---
GROQ_MODEL = "llama-3.1-8b-instant"

# RAG settings
VECTOR_DB_PATH = "vector_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

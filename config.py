"""
Settings shared by the pipeline, API, and evaluation scripts.

Values that depend on the machine running the project can be overridden
with environment variables, so no code change is needed to point at a
different Ollama server or try a different local model.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"

# Retrieval models (downloaded from Hugging Face on first use).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Local language model served by Ollama.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT_SECONDS = float(
    os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120")
)

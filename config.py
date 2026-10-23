"""
Settings shared by the pipeline, API, and evaluation scripts.

Values that depend on the machine running the project can be overridden
with environment variables or a local .env file.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"


def load_env_file(env_path: Path) -> None:
    """Load key-value pairs from an env file without overriding set environment variables."""
    if not env_path.is_file():
        return
    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and (
                    (val[0] == '"' and val[-1] == '"')
                    or (val[0] == "'" and val[-1] == "'")
                ):
                    val = val[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = val


load_env_file(PROJECT_ROOT / ".env")

# Retrieval models (downloaded from Hugging Face on first use).
# Either a Hugging Face model ID or a local folder path.
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
)

# LLM provider configuration ('ollama' or 'groq')
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()

# Local language model served by Ollama.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT_SECONDS = float(
    os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120")
)

# Cloud language model served by Groq.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b").strip()
GROQ_TIMEOUT_SECONDS = float(
    os.environ.get("GROQ_TIMEOUT_SECONDS", "60")
)
GROQ_MAX_COMPLETION_TOKENS = int(
    os.environ.get("GROQ_MAX_COMPLETION_TOKENS", "2048")
)
GROQ_API_URL = os.environ.get(
    "GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"
).strip()
GROQ_WAIT_BUDGET_SECONDS = float(
    os.environ.get("GROQ_WAIT_BUDGET_SECONDS", "60.0")
)
GROQ_REQUEST_DELAY_SECONDS = float(
    os.environ.get("GROQ_REQUEST_DELAY_SECONDS", "8.0")
)

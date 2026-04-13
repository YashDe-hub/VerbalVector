"""
Centralised configuration for VerbalVector.

All API keys and model identifiers are read from environment variables
(or a .env file in the project root via python-dotenv).
No secrets are ever hardcoded here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv  # pip: python-dotenv

# Load .env from project root (no-op if file doesn't exist)
load_dotenv(Path(__file__).parent / ".env")

# ── API Keys ────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY: str = os.environ.get("DEEPGRAM_API_KEY", "")
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
HUME_API_KEY: str = os.environ.get("HUME_API_KEY", "")

# ── Model Identifiers ────────────────────────────────────────────────────────
DEEPGRAM_STT_MODEL: str = os.environ.get("DEEPGRAM_STT_MODEL", "nova-3")
GEMINI_LLM_MODEL: str = os.environ.get("GEMINI_LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MAX_FILE_SIZE_BYTES: int = int(os.environ.get("MAX_FILE_SIZE_BYTES", 50 * 1024 * 1024))

# ── Directory Paths ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
OUTPUT_DIR = ROOT_DIR / "analysis_output"
VECTOR_DB_DIR = ROOT_DIR / "vector_db"

# Ensure runtime directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


def validate():
    """Raise if any required API key is missing."""
    missing = [k for k, v in {
        "DEEPGRAM_API_KEY": DEEPGRAM_API_KEY,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "HUME_API_KEY": HUME_API_KEY,
    }.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your keys."
        )

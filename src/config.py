import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory paths
REPO_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables strictly from repo's .env file
load_dotenv(REPO_ROOT / ".env")
DATA_DIR = REPO_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "posthog_raw" / "contents" / "docs"
LANCEDB_DIR = DATA_DIR / "lancedb"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
RUNS_DIR = REPO_ROOT / "runs"
TRACES_DIR = RUNS_DIR / "traces"
RETRIEVAL_TRACES_PATH = TRACES_DIR / "retrieval_traces.jsonl"

# Chunking parameters
CHUNK_MIN_TOKENS = 100
CHUNK_MAX_TOKENS = 500
OVERLAP_TOKENS = 50

# LanceDB configuration
LANCEDB_TABLE_NAME = "posthog_docs"
TABLE_NAME = LANCEDB_TABLE_NAME

# Model names
GEMINI_FLASH = os.getenv("GEMINI_FLASH_MODEL", "gemini-3.8-flash")
GEMINI_ARGON = os.getenv("GEMINI_ARGON_MODEL", "gemini-4-argon")
GEMINI_PRO = os.getenv("GEMINI_PRO_MODEL", "gemini-3.1-pro-preview")
ANTHROPIC_HAIKU = os.getenv("ANTHROPIC_HAIKU", "claude-haiku-5-5")
ANTHROPIC_SONNET = os.getenv("ANTHROPIC_SONNET", "claude-sonnet-5-5")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Provider configuration
DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "anthropic")
PRIMARY_PROVIDER: str = DEFAULT_LLM_PROVIDER

# Purge any ambient unauthorized keys from shell/OS environment to prevent silent fallback
for forbidden_env in ["GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"]:
    os.environ.pop(forbidden_env, None)

# Personal API Keys (strictly required from .env)
PERSONAL_GEMINI_API_KEY = os.getenv("PERSONAL_GEMINI_API_KEY")
PERSONAL_ANTHROPIC_API_KEY = os.getenv("PERSONAL_ANTHROPIC_API_KEY")


def get_gemini_client():
    """Returns a Google GenAI Client strictly initialized with PERSONAL_GEMINI_API_KEY.
    Ensures ambient GOOGLE_API_KEY or GEMINI_API_KEY are never used.
    """
    for forbidden in ["GOOGLE_API_KEY", "GEMINI_API_KEY"]:
        os.environ.pop(forbidden, None)
    if not PERSONAL_GEMINI_API_KEY or not PERSONAL_GEMINI_API_KEY.strip():
        raise ValueError(
            "PERSONAL_GEMINI_API_KEY is not set in .env! "
            "Using GOOGLE_API_KEY or GEMINI_API_KEY is strictly forbidden."
        )
    from google import genai
    return genai.Client(api_key=PERSONAL_GEMINI_API_KEY)


def get_anthropic_client():
    """Returns an Anthropic Client strictly initialized with PERSONAL_ANTHROPIC_API_KEY.
    Ensures ambient ANTHROPIC_API_KEY is never used.
    """
    os.environ.pop("ANTHROPIC_API_KEY", None)
    if not PERSONAL_ANTHROPIC_API_KEY or not PERSONAL_ANTHROPIC_API_KEY.strip():
        raise ValueError(
            "PERSONAL_ANTHROPIC_API_KEY is not set in .env! "
            "Using generic ANTHROPIC_API_KEY is strictly forbidden."
        )
    import anthropic
    return anthropic.Anthropic(api_key=PERSONAL_ANTHROPIC_API_KEY)


# Budget and runtime limits
SPEND_CAP_USD = float(os.getenv("SPEND_CAP_USD", "40.0"))
MAX_STEPS = 6
RRF_K = 60

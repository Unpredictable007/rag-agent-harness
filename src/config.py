import os
from dotenv import load_dotenv

load_dotenv()

# Model names
GEMINI_FLASH = os.getenv("GEMINI_FLASH_MODEL", "gemini-3.8-flash")
GEMINI_ARGON = os.getenv("GEMINI_ARGON_MODEL", "gemini-4-argon")
GEMINI_PRO = os.getenv("GEMINI_PRO_MODEL", "gemini-3.1-pro-preview")
ANTHROPIC_HAIKU = os.getenv("ANTHROPIC_HAIKU", "claude-haiku-5-5")
ANTHROPIC_SONNET = os.getenv("ANTHROPIC_SONNET", "claude-sonnet-5-5")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

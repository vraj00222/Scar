"""Pinned configuration. Everything is hardcoded on purpose."""
import os
from pathlib import Path

_ENV_PATH = Path(__file__).parent / ".env"


def _load_env():
    if not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ.setdefault(key.strip(), val)


_load_env()

MONGO_URI = os.environ.get("MONGO_URI", "")

# Provider-neutral: any OpenAI-compatible chat-completions endpoint. Currently Novita.
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.novita.ai/v3/openai")

# Optional. Neither Novita nor OpenRouter gave us a usable 1536-dim embeddings
# endpoint; with this set, scar vectors come from OpenAI instead of the local hash.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

DB_NAME = "scar"
MFLIX_DB = "sample_mflix"

# Pinned for the life of the demo. If these change, the demo proves nothing.
AGENT_MODEL = "anthropic/claude-sonnet-4.5"
JUDGE_MODEL = "anthropic/claude-haiku-4.5"
EMBED_MODEL = "text-embedding-3-small"

MAX_STEPS = 12
TOOL_TIMEOUT_MS = 15_000
MAX_DOCS_TO_MODEL = 50

EMBED_DIMS = 1536
VECTOR_INDEX = "scar_vec"
SCAR_TOP_K = 3
SCAR_DEDUPE_COSINE = 0.92
THRASH_LIMIT = 3  # identical pipeline submitted this many times forces a halt


def require_env():
    missing = [k for k, v in (("MONGO_URI", MONGO_URI),
                              ("LLM_API_KEY", LLM_API_KEY)) if not v]
    if missing:
        raise SystemExit(f"missing in .env: {', '.join(missing)}  (see .env.example)")

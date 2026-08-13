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

# Provider-neutral: any OpenAI-compatible chat-completions endpoint. Currently OpenRouter,
# which also serves /embeddings, so chat and vectors run off this one key.
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")

DB_NAME = "scar"
MFLIX_DB = "sample_mflix"

# Pinned for the life of the demo. If these change, the demo proves nothing.
AGENT_MODEL = "anthropic/claude-sonnet-4.5"
JUDGE_MODEL = "anthropic/claude-haiku-4.5"

EMBED_MODEL = "openai/text-embedding-3-small"

MAX_STEPS = 12
TOOL_TIMEOUT_MS = 15_000
MAX_DOCS_TO_MODEL = 50

EMBED_DIMS = 1536
VECTOR_INDEX = "scar_vec"
SCAR_TOP_K = 3
# Measured on text-embedding-3-small against this database's lessons: paraphrases of the
# same lesson land at 0.63-0.80, genuinely different lessons at 0.34-0.53. This sits in
# the gap. The 0.92 from the original spec is far above the paraphrase band, so it would
# have deduped nothing and let four copies of one lesson crowd out the top-3 retrieval.
SCAR_DEDUPE_COSINE = 0.58
THRASH_LIMIT = 3  # identical pipeline submitted this many times forces a halt


def require_env():
    missing = [k for k, v in (("MONGO_URI", MONGO_URI),
                              ("LLM_API_KEY", LLM_API_KEY)) if not v]
    if missing:
        raise SystemExit(f"missing in .env: {', '.join(missing)}  (see .env.example)")

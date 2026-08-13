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

DB_NAME = "scar"
MFLIX_DB = "sample_mflix"

# Pinned for the life of the demo. If these change, the demo proves nothing.
# Both verified live against Novita: the agent emits well-formed tool_calls across
# multiple turns, and the judge returns bare parseable JSON that actually discriminates.
AGENT_MODEL = "deepseek/deepseek-v3.2"
JUDGE_MODEL = "meta-llama/llama-3.1-8b-instruct"

# Novita serves /embeddings even though no embedding model appears in /models.
# bge-m3 is 1024-dim; nothing on this provider offers 1536.
EMBED_MODEL = "baai/bge-m3"

MAX_STEPS = 12
TOOL_TIMEOUT_MS = 15_000
MAX_DOCS_TO_MODEL = 50

EMBED_DIMS = 1024
VECTOR_INDEX = "scar_vec"
SCAR_TOP_K = 3
# Measured on bge-m3 against this database's lessons: a genuine paraphrase of the same
# lesson scores ~0.90, two genuinely different lessons ~0.62. The 0.92 this was
# originally set to sat above the paraphrase band, so nothing ever deduped.
SCAR_DEDUPE_COSINE = 0.88
THRASH_LIMIT = 3  # identical pipeline submitted this many times forces a halt


def require_env():
    missing = [k for k, v in (("MONGO_URI", MONGO_URI),
                              ("LLM_API_KEY", LLM_API_KEY)) if not v]
    if missing:
        raise SystemExit(f"missing in .env: {', '.join(missing)}  (see .env.example)")

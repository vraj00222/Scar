"""1536-dim embeddings for scars.

OpenRouter does not serve an embeddings endpoint, so this reaches OpenAI directly
for `text-embedding-3-small` (natively 1536 dims) when OPENAI_API_KEY is set.

Without that key it falls back to a deterministic character-trigram hash vector so
the system still runs keyless. The fallback is lexical, not semantic: it is good
enough to catch near-duplicate scar text, and noticeably worse at matching a task
to a scar that is worded differently. Pick one backend and stay on it — vectors
from the two are not comparable, so switching means dropping db.scars.
"""
import hashlib
import math
import re

from openai import OpenAI

import config

_client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
BACKEND = f"openai:{config.EMBED_MODEL}" if _client else "local:char-trigram-hash"


def embed(text):
    text = (text or "").strip()
    if not text:
        return [0.0] * config.EMBED_DIMS
    if _client:
        result = _client.embeddings.create(model=config.EMBED_MODEL, input=text[:8000])
        return result.data[0].embedding
    return _hash_embed(text)


def _hash_embed(text):
    vec = [0.0] * config.EMBED_DIMS
    clean = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    grams = clean.split() + [clean[i:i + 3] for i in range(max(0, len(clean) - 2))]
    for gram in grams:
        slot = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big")
        vec[slot % config.EMBED_DIMS] += 1.0
    return _unit(vec)


def _unit(vec):
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

"""Scar embeddings.

OpenRouter serves an OpenAI-compatible /embeddings endpoint, so `text-embedding-3-small`
(natively 1536 dims) runs off the same key as the chat models.

The backend is chosen once, at import. A remote failure raises instead of quietly
returning a hash vector: hash vectors and model vectors live in different spaces,
so mixing them inside db.scars would break retrieval with no visible symptom.
"""
import hashlib
import math
import re

from openai import OpenAI

import config

_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
                 timeout=60.0) if config.LLM_API_KEY else None
BACKEND = f"{config.EMBED_MODEL} ({config.EMBED_DIMS}d)" if _client \
    else f"local:char-trigram-hash ({config.EMBED_DIMS}d)"


def embed(text):
    text = (text or "").strip()
    if not text:
        return [0.0] * config.EMBED_DIMS
    if _client is None:
        return _hash_embed(text)
    last = None
    for _ in range(2):
        try:
            result = _client.embeddings.create(model=config.EMBED_MODEL, input=text[:8000])
            vector = result.data[0].embedding
            if len(vector) != config.EMBED_DIMS:
                raise RuntimeError(
                    f"{config.EMBED_MODEL} returned {len(vector)} dims, "
                    f"config.EMBED_DIMS is {config.EMBED_DIMS}")
            return vector
        except Exception as exc:
            last = exc
    raise RuntimeError(f"embedding failed: {type(last).__name__}: {last}")


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

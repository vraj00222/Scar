"""Mongo handles, indexes, and the non-blocking step writer."""
import queue
import threading
import time
from datetime import datetime, timezone

from pymongo import ASCENDING, MongoClient
from pymongo.operations import SearchIndexModel

import config

if not config.MONGO_URI:
    raise SystemExit("missing MONGO_URI in .env  (see .env.example)")

_client = MongoClient(config.MONGO_URI, appname="scar", serverSelectionTimeoutMS=10_000)

db = _client[config.DB_NAME]
mflix = _client[config.MFLIX_DB]

runs = db.runs
steps = db.steps
scars = db.scars
evals = db.evals


def utcnow():
    return datetime.now(timezone.utc)


def ensure_indexes():
    steps.create_index([("run_id", ASCENDING), ("idx", ASCENDING)])
    runs.create_index([("task_id", ASCENDING)])
    evals.create_index([("run_id", ASCENDING), ("step_idx", ASCENDING)])
    scars.create_index([("status", ASCENDING)])


def _vector_dims(index):
    for field in (index.get("latestDefinition") or {}).get("fields", []):
        if field.get("type") == "vector":
            return field.get("numDimensions")
    return None


def _create_vector_index():
    scars.create_search_index(SearchIndexModel(
        name=config.VECTOR_INDEX,
        type="vectorSearch",
        definition={"fields": [
            {"type": "vector", "path": "embedding",
             "numDimensions": config.EMBED_DIMS, "similarity": "cosine"},
            {"type": "filter", "path": "status"},
        ]},
    ))


def ensure_vector_index():
    """Create the Atlas Vector Search index on scars.embedding. Returns its state.

    Atlas takes ~30s to build it. Retrieval falls back to in-process cosine until
    then, so nothing hard-blocks on this.

    An index whose numDimensions no longer matches config.EMBED_DIMS can never match
    the vectors we write, and Atlas gives no visible symptom, so it is dropped and
    rebuilt rather than reported as fine.
    """
    try:
        existing = {i["name"]: i for i in scars.list_search_indexes()}
    except Exception:
        return "unavailable"

    current = existing.get(config.VECTOR_INDEX)
    if current is not None:
        dims = _vector_dims(current)
        if dims == config.EMBED_DIMS:
            return "ready" if current.get("queryable") else "building"
        try:
            scars.drop_search_index(config.VECTOR_INDEX)
            for _ in range(30):
                time.sleep(2)
                if config.VECTOR_INDEX not in {i["name"] for i in scars.list_search_indexes()}:
                    break
            else:
                return f"stale {dims}d index would not drop; drop {config.VECTOR_INDEX} by hand"
        except Exception as exc:
            return f"failed dropping stale {dims}d index: {exc}"

    try:
        _create_vector_index()
    except Exception as exc:
        return f"failed: {exc}"
    return "building"


class StepWriter:
    """Hands steps to a background thread so the agent loop never waits on Mongo.

    One drain thread keeps insert order stable, which the supervisor's change
    stream relies on.
    """

    def __init__(self, run_id):
        self.run_id = run_id
        self.idx = 0
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def emit(self, kind, content, pipeline=None, error=None):
        doc = {
            "run_id": self.run_id,
            "idx": self.idx,
            "kind": kind,
            "content": content,
            "pipeline": pipeline,
            "error": error,
            "ts": utcnow(),
        }
        self.idx += 1
        self._q.put(dict(doc))
        return doc

    def _drain(self):
        while True:
            doc = self._q.get()
            try:
                if doc is None:
                    return
                steps.insert_one(doc)
            except Exception as exc:
                print(f"[step writer] dropped step: {exc}")
            finally:
                self._q.task_done()

    def close(self):
        self._q.put(None)
        self._q.join()

"""Task bank + deterministic verification.

Verification never asks a model whether an answer looks good. It checks shape,
types, nulls, ranges and ordering in plain Python. A run passes only if the
documents the agent actually produced survive those checks.

The questions deliberately give away nothing about how sample_mflix is shaped.
Every trap in that data has to be discovered by running into it.
"""


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def verify(docs, check):
    """Return (ok, reason). Pure code, no model in the loop.

    `distinct` is only ever applied to grouping keys (genre, decade, director),
    where a repeat means the agent failed to group. It is deliberately NOT applied
    to `title`: sample_mflix genuinely stores some films twice (there are two
    separate "The Shawshank Redemption" documents), so a correct top-N answer can
    legitimately repeat a title.
    """
    if not isinstance(docs, list):
        return False, "result is not a list of documents"
    if not docs:
        return False, "result is empty"

    n = check.get("n_docs")
    if n is not None and len(docs) != n:
        return False, f"expected exactly {n} documents, got {len(docs)}"

    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            return False, f"document {i} is {type(doc).__name__}, not an object"
        for field, kind in check["fields"].items():
            if field not in doc:
                return False, f"document {i} is missing field '{field}'"
            val = doc[field]
            if val is None:
                return False, f"document {i} has null '{field}'"
            if kind == "num" and not _is_num(val):
                return False, (f"document {i} has '{field}'={val!r} "
                               f"({type(val).__name__}), expected a number")
            if kind == "str" and (not isinstance(val, str) or not val.strip()):
                return False, (f"document {i} has '{field}'={val!r} "
                               f"({type(val).__name__}), expected a non-empty string")

    for field, (lo, hi) in check.get("ranges", {}).items():
        for i, doc in enumerate(docs):
            val = doc[field]
            if not (lo <= val <= hi):
                return False, f"document {i} has '{field}'={val}, outside {lo}..{hi}"

    for field, allowed in check.get("allowed", {}).items():
        for i, doc in enumerate(docs):
            if doc[field] not in allowed:
                return False, f"document {i} has '{field}'={doc[field]!r}, not one of {sorted(allowed)}"

    for field in check.get("distinct", []):
        seen = [doc[field] for doc in docs]
        if len(set(seen)) != len(seen):
            return False, f"'{field}' has duplicate values across documents"

    if "sort" in check:
        field, direction = check["sort"]
        vals = [doc[field] for doc in docs]
        pairs = list(zip(vals, vals[1:]))
        if direction == "desc" and any(a < b for a, b in pairs):
            return False, f"documents are not sorted by '{field}' descending: {vals}"
        if direction == "asc" and any(a > b for a, b in pairs):
            return False, f"documents are not sorted by '{field}' ascending: {vals}"

    return True, "all checks passed"


TASK_LIST = [
    {
        "task_id": "comedy_top_rated",
        "family": "top_rated",
        "question": (
            "In the `movies` collection, find the 5 highest-rated Comedy movies by IMDb rating. "
            "Return exactly 5 documents, each with two fields: `title` (string) and `rating` (number)."
        ),
        "check": {
            "n_docs": 5,
            "fields": {"title": "str", "rating": "num"},
            "ranges": {"rating": (1.0, 10.0)},
            "sort": ("rating", "desc"),
        },
    },
    {
        "task_id": "drama_top_rated",
        "family": "top_rated",
        "question": (
            "In the `movies` collection, find the 5 highest-rated Drama movies that have at least "
            "1000 IMDb votes. Return exactly 5 documents, each with two fields: `title` (string) "
            "and `rating` (number)."
        ),
        "check": {
            "n_docs": 5,
            "fields": {"title": "str", "rating": "num"},
            "ranges": {"rating": (1.0, 10.0)},
            "sort": ("rating", "desc"),
        },
    },
    {
        "task_id": "avg_rating_by_decade",
        "family": "avg_group",
        "question": (
            "In the `movies` collection, compute the average IMDb rating of movies for each decade "
            "from the 1970s through the 2000s. Return exactly 4 documents, each with two fields: "
            "`decade` (number, e.g. 1970) and `avg_rating` (number), sorted by `decade` ascending."
        ),
        "check": {
            "n_docs": 4,
            "fields": {"decade": "num", "avg_rating": "num"},
            "ranges": {"decade": (1970, 2000), "avg_rating": (1.0, 10.0)},
            "allowed": {"decade": {1970, 1980, 1990, 2000}},
            "distinct": ["decade"],
            "sort": ("decade", "asc"),
        },
    },
    {
        "task_id": "avg_rating_by_genre",
        "family": "avg_group",
        "question": (
            "In the `movies` collection, compute the average IMDb rating for each of these three "
            "genres: Action, Drama, Comedy. Return exactly 3 documents, each with two fields: "
            "`genre` (string) and `avg_rating` (number)."
        ),
        "check": {
            "n_docs": 3,
            "fields": {"genre": "str", "avg_rating": "num"},
            "ranges": {"avg_rating": (1.0, 10.0)},
            "allowed": {"genre": {"Action", "Drama", "Comedy"}},
            "distinct": ["genre"],
        },
    },
    {
        "task_id": "top_directors",
        "family": "group_count",
        "question": (
            "In the `movies` collection, find the 5 directors who directed the most movies. "
            "Return exactly 5 documents, each with two fields: `director` (string) and "
            "`count` (number), sorted by `count` descending."
        ),
        "check": {
            "n_docs": 5,
            "fields": {"director": "str", "count": "num"},
            "ranges": {"count": (2, 10_000)},
            "distinct": ["director"],
            "sort": ("count", "desc"),
        },
    },
    {
        "task_id": "longest_movies",
        "family": "top_sorted",
        "question": (
            "In the `movies` collection, find the 5 longest movies by runtime. Return exactly 5 "
            "documents, each with two fields: `title` (string) and `runtime` (number), sorted by "
            "`runtime` descending."
        ),
        "check": {
            "n_docs": 5,
            "fields": {"title": "str", "runtime": "num"},
            "ranges": {"runtime": (60, 10_000)},
            "sort": ("runtime", "desc"),
        },
    },
]

TASKS = {t["task_id"]: t for t in TASK_LIST}


def family_of(task_id):
    return TASKS[task_id]["family"]


def ids_in_family(family):
    return [t["task_id"] for t in TASK_LIST if t["family"] == family]

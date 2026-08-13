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

    # Ground truth. Shape checks alone cannot catch an answer that is well-formed and
    # wrong, which is the only kind of failure a competent model does not notice and
    # repair on its own: nothing in a plausible number looks broken.
    expect = check.get("expect")
    if expect:
        key, values = expect["key"], expect["values"]
        got = {doc[key]: doc for doc in docs}
        for want_key in sorted(values):
            if want_key not in got:
                return False, f"no document for '{key}'={want_key}"
            for field, want in values[want_key].items():
                actual = got[want_key][field]
                if isinstance(want, float):
                    if abs(actual - want) > 0.01:
                        return False, (f"'{key}'={want_key}: '{field}' is {actual}, "
                                       f"the correct value is {want}")
                elif actual != want:
                    short = want - actual if _is_num(actual) else None
                    detail = "" if short is None else (
                        f" — {abs(short)} {'too few' if short > 0 else 'too many'}")
                    return False, (f"'{key}'={want_key}: '{field}' is {actual}, "
                                   f"the correct value is {want}{detail}")

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
    # The tasks below deliberately stack several traps in one pipeline. A single
    # visible trap is not enough to fail this agent: it reads the bad rows in the
    # tool result and repairs itself within the run. Stacking them means each
    # rediscovery costs steps, and the 12-step ceiling becomes reachable.
    # Every range here only enforces a constraint the question already states, so
    # nothing is hardcoded from the answer.
    {
        "task_id": "director_avg_rating",
        "family": "group_stats",
        "question": (
            "In the `movies` collection, among directors who have directed at least 10 movies "
            "that have a numeric IMDb rating, find the 5 with the highest average IMDb rating. "
            "Return exactly 5 documents, each with three fields: `director` (string), "
            "`avg_rating` (number) and `movie_count` (number), sorted by `avg_rating` descending."
        ),
        "check": {
            "n_docs": 5,
            "fields": {"director": "str", "avg_rating": "num", "movie_count": "num"},
            "ranges": {"avg_rating": (1.0, 10.0), "movie_count": (10, 10_000)},
            "distinct": ["director"],
            "sort": ("avg_rating", "desc"),
        },
    },
    {
        "task_id": "cast_avg_rating",
        "family": "group_stats",
        "question": (
            "In the `movies` collection, among actors who appear in at least 15 movies that have "
            "a numeric IMDb rating, find the 5 with the highest average IMDb rating. Return "
            "exactly 5 documents, each with three fields: `actor` (string), `avg_rating` (number) "
            "and `movie_count` (number), sorted by `avg_rating` descending."
        ),
        "check": {
            "n_docs": 5,
            "fields": {"actor": "str", "avg_rating": "num", "movie_count": "num"},
            "ranges": {"avg_rating": (1.0, 10.0), "movie_count": (15, 10_000)},
            "distinct": ["actor"],
            "sort": ("avg_rating", "desc"),
        },
    },
    {
        # The counts below are ground truth, computed directly from the collection.
        # 35 documents store `year` as a string ("2010è" and similar), so a numeric
        # range match drops them and returns a total that looks entirely reasonable.
        # Nothing about 866 suggests the answer should have been 870.
        "task_id": "movies_per_year",
        "family": "exact_count",
        "question": (
            "In the `movies` collection, count how many movies were released in each year from "
            "2010 through 2013. Every movie in the collection with one of those release years "
            "must be counted. Return exactly 4 documents, each with two fields: `year` (number, "
            "e.g. 2010) and `count` (number), sorted by `year` ascending."
        ),
        "check": {
            "n_docs": 4,
            "fields": {"year": "num", "count": "num"},
            "allowed": {"year": {2010, 2011, 2012, 2013}},
            "distinct": ["year"],
            "expect": {"key": "year", "values": {
                2010: {"count": 870}, 2011: {"count": 895},
                2012: {"count": 958}, 2013: {"count": 1105}}},
            "sort": ("year", "asc"),
        },
    },
    {
        "task_id": "top_genre_by_decade",
        "family": "cross_group",
        "question": (
            "In the `movies` collection, for each decade from the 1970s through the 2000s, find "
            "the single genre with the highest average IMDb rating among movies released in that "
            "decade, counting only genres with at least 20 rated movies in that decade. Return "
            "exactly 4 documents, each with three fields: `decade` (number, e.g. 1970), "
            "`genre` (string) and `avg_rating` (number), sorted by `decade` ascending."
        ),
        "check": {
            "n_docs": 4,
            "fields": {"decade": "num", "genre": "str", "avg_rating": "num"},
            "ranges": {"decade": (1970, 2000), "avg_rating": (1.0, 10.0)},
            "allowed": {"decade": {1970, 1980, 1990, 2000}},
            "distinct": ["decade"],
            "sort": ("decade", "asc"),
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

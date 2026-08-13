# SCAR

Evals that don't stop at deployment.

Offline evals judge an agent before you ship it. Scar judges it *while it runs*, and turns
every caught failure into a **scar** — a durable lesson that future agents inherit before
their first token. The win condition is transfer: a scar born on task A visibly helps on
unrelated task B, with the same model and the same tools.

## Setup

```bash
python3.11 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env      # fill in MONGO_URI, OPENROUTER_API_KEY, OPENAI_API_KEY
```

Needs the `sample_mflix` sample dataset loaded in the same Atlas cluster.

`OPENAI_API_KEY` is optional but recommended: OpenRouter serves no embeddings endpoint, so
scar vectors come from OpenAI's `text-embedding-3-small`. Without it, `embed.py` falls back
to a local lexical hash that scores a paraphrased scar at ~0.55 cosine instead of ~0.9,
which means near-duplicate scars stop deduping. Vectors from the two backends are not
comparable — if you switch, drop `db.scars`.

## Components

| file | role |
| --- | --- |
| `agent.py` | one task, one run, one tool, max 12 steps |
| `supervisor.py` | watches the step change stream, scores every step, can halt a run |
| `reflector.py` | turns failed runs into scars |
| `compare.py` | the demo: same task, cold vs warm, live, side by side |
| `grind.py` | loops cold runs to accumulate failures |
| `tasks.py` | task bank + the deterministic verifier |
| `embed.py` `db.py` `config.py` | embeddings, Mongo handles, pinned config |

## Running it

Four panes. Left to right, they are the agent making mistakes, the supervisor catching
them, and the reflector turning them into scars.

```bash
# pane 1 — supervise
./.venv/bin/python supervisor.py

# pane 2 — reflect
./.venv/bin/python reflector.py

# pane 3 — make failures
./.venv/bin/python grind.py --limit 8

# pane 4 — the demo, once scars exist
./.venv/bin/python compare.py --task avg_rating_by_genre
```

Single runs:

```bash
./.venv/bin/python agent.py --list
./.venv/bin/python agent.py --task comedy_top_rated
./.venv/bin/python agent.py --task comedy_top_rated --mode warm
./.venv/bin/python reflector.py --backfill    # reflect on past failures instead of watching
```

Tail raw steps as they land:

```bash
./.venv/bin/python -c "
import db
print('watching db.steps ...')
for change in db.steps.watch():
    d = change['fullDocument']
    print(f\"{d['idx']:>3} {d['kind']:<12} {(d.get('content') or '')[:110]}\")
"
```

## The promotion gate

A scar is born as a `candidate`. It becomes `active` in exactly one way: a run that
retrieved it went on to pass, where "pass" is decided by `tasks.verify()` — shape, types,
nulls, ranges and ordering, checked in plain Python. No model is ever asked whether an
answer looks good, and no model can promote a scar. That is the anti-reward-hacking
mechanism; it lives in `agent.promote()`.

Warm retrieval deliberately pulls both `active` and `candidate` scars. A candidate that is
never retrieved can never earn promotion, so gating retrieval on `active` alone would
deadlock the system and no scar would ever activate.

## Correctness

Verification is deterministic, in `tasks.verify()`. It rejects empty results, missing
fields, nulls, wrong types, out-of-range values, duplicates, and bad ordering. The failure
modes in `sample_mflix` are never hardcoded and never hinted at in any prompt — the system
runs into them. For example, `imdb.rating` is an empty string on some documents and BSON
sorts strings above numbers, so `$sort: {"imdb.rating": -1}` puts junk on top. Nothing
tells the agent that; it has to be learned once and inherited thereafter.

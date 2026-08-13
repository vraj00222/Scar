"""Turn failures into scars. Long-running, or --backfill over past failures.

One failed run yields at most one scar, and the scar has to be about the database
rather than about the task. It lands as a candidate; only agent.promote() can make
it active, and only on a run that passed.
"""
import argparse
import json
import re
import sys

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config
import db
import embed

console = Console()

REFLECT_SYSTEM = """A data-analysis agent just failed a task against the MongoDB database `sample_mflix`.
You extract exactly ONE lesson from the wreckage.

The lesson must be domain-general: it describes how this DATABASE behaves and what to do about
it, and it must be useful to a different agent working on a completely different question.

GOOD: "The genres field is an array; match it with $in or $unwind it before grouping, because
       equality against a scalar silently returns nothing."
GOOD: "imdb.rating is sometimes an empty string, and BSON sorts strings above numbers, so
       $sort on it puts junk first unless you $match {$type: 'number'} beforehand."
BAD:  "The comedy query failed."
BAD:  "Remember to filter for Comedy movies before sorting."

Never mention the specific task, the specific question, or this particular run. A reader must
not be able to tell which task produced the lesson.

Reply with JSON only, no prose:
{"text": "the lesson, one or two imperative sentences",
 "trigger": "short condition under which a future agent should recall this",
 "situation": "one sentence describing the class of situation it was learned in"}"""

# Phrases that prove the model wrote about the task instead of the database.
TASK_SPECIFIC = ("this task", "this run", "this query", "the user asked", "in this case",
                 "the previous agent", "the agent failed", "this question", "the above")


def trajectory_text(run_id):
    lines = []
    for step in db.steps.find({"run_id": run_id}).sort("idx", 1):
        if step["kind"] == "tool_call" and step.get("pipeline") is not None:
            body = json.dumps(step["pipeline"], default=str)[:600]
            lines.append(f"[{step['idx']}] pipeline: {body}")
        else:
            lines.append(f"[{step['idx']}] {step['kind']}: {(step.get('content') or '')[:400]}")
    return "\n".join(lines)


def parse_json(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text or "", re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def extract(client, run):
    """Ask for one scar. Returns (scar_fields, rejection_reason)."""
    failure = run.get("verdict_reason") or ""
    last_eval = db.evals.find_one({"run_id": run["_id"], "verdict": {"$in": ["halt", "warn"]}},
                                  sort=[("step_idx", -1)])
    if last_eval:
        failure = f"{failure} supervisor: {last_eval.get('reason', '')}".strip()

    user = (f"TASK THE AGENT WAS GIVEN:\n{run.get('task')}\n\n"
            f"HOW IT WENT WRONG (determined in code, not by opinion):\n{failure or 'the run failed'}\n\n"
            f"FULL TRAJECTORY:\n{trajectory_text(run['_id'])[:8000]}")

    response = client.chat.completions.create(
        model=config.AGENT_MODEL, temperature=0, max_tokens=400,
        messages=[{"role": "system", "content": REFLECT_SYSTEM},
                  {"role": "user", "content": user}])
    parsed = parse_json(response.choices[0].message.content)
    if not parsed or not str(parsed.get("text", "")).strip():
        return None, "model returned no usable lesson"

    text = str(parsed["text"]).strip()
    if len(text) < 20:
        return None, "lesson too short to be a lesson"
    lowered = text.lower()
    for phrase in TASK_SPECIFIC:
        if phrase in lowered:
            return None, f"lesson is task-specific (contains {phrase!r})"
    return {"text": text,
            "trigger": str(parsed.get("trigger", "")).strip()[:300],
            "situation": str(parsed.get("situation", "")).strip()[:300]}, None


def store(scar, run_id):
    """Insert as candidate unless a near-duplicate already exists."""
    vector = embed.embed(scar["text"])
    for existing in db.scars.find({}, {"embedding": 1, "text": 1, "status": 1}):
        score = embed.cosine(vector, existing.get("embedding") or [])
        if score > config.SCAR_DEDUPE_COSINE:
            return None, (existing, score)
    scar_id = db.scars.insert_one({
        "text": scar["text"], "trigger": scar["trigger"], "situation": scar["situation"],
        "embedding": vector, "born_from_run": run_id, "times_used": 0, "times_helped": 0,
        "status": "candidate", "created_at": db.utcnow(),
    }).inserted_id
    return scar_id, None


def reflect(client, run):
    console.print(f"[bright_black]reflecting on failed run {run['_id']} "
                  f"({run.get('task_id')})[/]")
    scar, rejection = extract(client, run)
    if rejection:
        console.print(f"   [yellow]no scar: {rejection}[/]")
        return
    scar_id, duplicate = store(scar, run["_id"])
    if duplicate:
        existing, score = duplicate
        console.print(f"   [bright_black]duplicate of an existing scar (cosine {score:.3f}) — "
                      f"skipped[/]")
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bright_black", justify="right")
    grid.add_column()
    grid.add_row("lesson", Text(scar["text"], style="bold green"))
    grid.add_row("applies when", Text(scar["trigger"]))
    grid.add_row("born from", Text(f"run {run['_id']}  ·  task {run.get('task_id')}",
                                   style="bright_black"))
    grid.add_row("status", Text("candidate — needs a passing run to activate", style="yellow"))
    console.print(Panel(grid, title="[bold green]SCAR BORN[/]", border_style="green"))


def main():
    parser = argparse.ArgumentParser(description="Extract scars from failed runs.")
    parser.add_argument("--backfill", action="store_true",
                        help="process existing failed runs instead of watching")
    parser.add_argument("--limit", type=int, default=10, help="max runs to backfill")
    args = parser.parse_args()

    config.require_env()
    db.ensure_indexes()
    db.ensure_vector_index()
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY, timeout=120.0)

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bright_black", justify="right")
    grid.add_column()
    grid.add_row("model", Text(config.AGENT_MODEL, style="bold white"))
    grid.add_row("embeddings", Text(embed.BACKEND, style="white"))
    grid.add_row("dedupe", Text(f"cosine > {config.SCAR_DEDUPE_COSINE}"))
    console.print(Panel(grid, title="[bold]SCAR · reflector[/]", border_style="bright_black"))

    if args.backfill:
        done = {s["born_from_run"] for s in db.scars.find({}, {"born_from_run": 1})}
        runs = [r for r in db.runs.find({"verdict": "fail"}).sort("started_at", -1)
                if r["_id"] not in done][:args.limit]
        console.print(f"[bright_black]{len(runs)} failed run(s) to reflect on[/]\n")
        for run in runs:
            reflect(client, run)
        return 0

    console.print("[bright_black]watching for failed runs ...[/]\n")
    with db.runs.watch(full_document="updateLookup") as stream:
        for change in stream:
            if change.get("operationType") not in ("update", "replace"):
                continue
            run = change.get("fullDocument")
            if not run or run.get("verdict") != "fail":
                continue
            if db.scars.find_one({"born_from_run": run["_id"]}):
                continue
            reflect(client, run)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[bright_black]reflector stopped[/]")

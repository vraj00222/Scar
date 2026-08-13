"""One task, one run.

Cold mode runs with nothing. Warm mode retrieves scars and injects them into the
system prompt before the first token. The verdict is decided in code by
tasks.verify(), and that verdict is the only thing that can promote a scar.
"""
import argparse
import json
import random
import statistics
import sys
from datetime import datetime

from openai import OpenAI
from pymongo.errors import PyMongoError
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

import config
import db
import embed
import tasks

console = Console()

ALLOWED_COLLECTIONS = {"movies", "comments", "theaters", "users"}

SYSTEM_PROMPT = f"""You are a data analyst agent working against a MongoDB database called `sample_mflix`.

You have exactly one tool: run_pipeline(collection, pipeline_json). It executes a MongoDB
aggregation pipeline and returns the resulting documents, or an error string.

How you are graded:
- The LAST pipeline you run is the one that gets checked. Its documents are your answer.
- Those documents must match the field names and types the task asks for, exactly.
- After that pipeline, reply with a short plain-text answer and no tool call.

Work iteratively. Run a pipeline, read what actually came back, and change your approach if
the result does not answer the question. A pipeline returning without an error does not mean
it returned the right thing.

You have at most {config.MAX_STEPS} steps."""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_pipeline",
        "description": "Run a MongoDB aggregation pipeline against a collection in sample_mflix.",
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Collection name, e.g. 'movies'."},
                "pipeline_json": {"type": "string",
                                  "description": "The aggregation pipeline as a JSON array of stages."},
            },
            "required": ["collection", "pipeline_json"],
        },
    },
}]


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def run_pipeline(collection, pipeline_json):
    """Return (docs, error_string). Exactly one of the two is None."""
    if collection not in ALLOWED_COLLECTIONS:
        return None, f"unknown collection {collection!r}; available: {sorted(ALLOWED_COLLECTIONS)}"
    try:
        pipeline = json.loads(pipeline_json)
    except json.JSONDecodeError as exc:
        return None, f"pipeline_json is not valid JSON: {exc}"
    if not isinstance(pipeline, list):
        return None, "pipeline_json must be a JSON array of aggregation stages"
    try:
        docs = list(db.mflix[collection].aggregate(pipeline, maxTimeMS=config.TOOL_TIMEOUT_MS))
    except PyMongoError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return docs[:config.MAX_DOCS_TO_MODEL], None


def family_median(task_id):
    ids = tasks.ids_in_family(tasks.family_of(task_id))
    prior = [r["steps_taken"] for r in db.runs.find(
        {"task_id": {"$in": ids}, "steps_taken": {"$gt": 0}}, {"steps_taken": 1})]
    return statistics.median(prior) if prior else None


def retrieve_scars(question, k=None):
    """Top-k scars for this task. Candidates are included so they can earn promotion."""
    k = k or config.SCAR_TOP_K
    vector = embed.embed(question)
    wanted = ["active", "candidate"]
    try:
        hits = list(db.scars.aggregate([
            {"$vectorSearch": {
                "index": config.VECTOR_INDEX, "path": "embedding", "queryVector": vector,
                "numCandidates": 100, "limit": k, "filter": {"status": {"$in": wanted}}}},
            {"$project": {"text": 1, "trigger": 1, "situation": 1, "status": 1,
                          "born_from_run": 1, "score": {"$meta": "vectorSearchScore"}}},
        ]))
        return hits, "$vectorSearch"
    except PyMongoError:
        # Index still building, or not an Atlas deployment. The scar set is small.
        pool = list(db.scars.find({"status": {"$in": wanted}}))
        for scar in pool:
            scar["score"] = embed.cosine(vector, scar.get("embedding") or [])
        pool.sort(key=lambda s: -s["score"])
        return pool[:k], "local cosine"


def scar_block(scars):
    lines = ["", "LESSONS FROM PRIOR RUNS",
             "Earlier agents failed against this same database. Apply these before your first query."]
    for scar in scars:
        lines.append(f"- {scar['text']}")
        if scar.get("trigger"):
            lines.append(f"  applies when: {scar['trigger']}")
    return "\n".join(lines)


def promote(scars_used, verdict):
    """candidate -> active, and only ever on a deterministic pass.

    This is the anti-reward-hacking gate. No model opinion reaches this function:
    `verdict` comes from tasks.verify(). A scar that was retrieved by a run which
    then passed has demonstrably survived contact with a real task.
    """
    if not scars_used or verdict != "pass":
        return []
    db.scars.update_many({"_id": {"$in": scars_used}}, {"$inc": {"times_helped": 1}})
    promoted = [s["_id"] for s in db.scars.find(
        {"_id": {"$in": scars_used}, "status": "candidate"}, {"_id": 1})]
    if promoted:
        db.scars.update_many({"_id": {"$in": promoted}}, {"$set": {"status": "active"}})
    return promoted


def execute(task, mode, ui):
    """Run one task to a verdict. `ui` is called as ui(event_name, payload)."""
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY, timeout=120.0)

    run_id = db.runs.insert_one({
        "task": task["question"], "task_id": task["task_id"], "mode": mode,
        "model": config.AGENT_MODEL, "scars_used": [], "steps_taken": 0,
        "verdict": None, "started_at": db.utcnow(), "ended_at": None,
    }).inserted_id

    median = family_median(task["task_id"])
    ui("start", {"run_id": run_id, "task": task, "mode": mode, "median": median})

    system = SYSTEM_PROMPT
    scars_used = []
    if mode == "warm":
        hits, source = retrieve_scars(task["question"])
        ui("scars", {"scars": hits, "source": source})
        if hits:
            scars_used = [s["_id"] for s in hits]
            system += scar_block(hits)
            db.scars.update_many({"_id": {"$in": scars_used}}, {"$inc": {"times_used": 1}})
            db.runs.update_one({"_id": run_id}, {"$set": {"scars_used": scars_used}})

    writer = db.StepWriter(run_id)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": task["question"]}]
    last_docs, finished, steps_taken, crash, halted = None, False, 0, None, None

    try:
        for turn in range(1, config.MAX_STEPS + 1):
            halt = db.evals.find_one({"run_id": run_id, "verdict": "halt"})
            if halt:
                halted = halt.get("reason") or "supervisor halted the run"
                ui("halt", {"reason": halted})
                break

            steps_taken = turn
            ui("turn", {"n": turn, "max": config.MAX_STEPS, "median": median})

            response = client.chat.completions.create(
                model=config.AGENT_MODEL, messages=messages, tools=TOOLS, temperature=0)
            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            text = (message.content or "").strip()

            if text:
                writer.emit("thought" if tool_calls else "final", text)
                ui("thought" if tool_calls else "final", {"text": text})

            if not tool_calls:
                if not text:
                    writer.emit("final", "")
                finished = True
                break

            assistant = {"role": "assistant", "content": message.content or "", "tool_calls": [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            } for tc in tool_calls]}
            messages.append(assistant)

            for tc in tool_calls:
                try:
                    call_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    call_args = {}
                collection = call_args.get("collection", "")
                raw_pipeline = call_args.get("pipeline_json", "")
                try:
                    parsed = json.loads(raw_pipeline)
                except (json.JSONDecodeError, TypeError):
                    parsed = None

                writer.emit("tool_call", f"run_pipeline({collection})", pipeline=parsed)
                ui("tool_call", {"collection": collection, "pipeline": parsed, "raw": raw_pipeline})

                docs, error = run_pipeline(collection, raw_pipeline)
                if error:
                    writer.emit("tool_result", error, error=error)
                    ui("tool_result", {"error": error})
                    payload = f"ERROR: {error}"
                else:
                    last_docs = docs
                    body = json.dumps(jsonable(docs), indent=2)
                    writer.emit("tool_result", body[:2000])
                    ui("tool_result", {"n": len(docs), "docs": docs})
                    payload = body if len(body) <= 12_000 else (
                        body[:12_000] + f"\n... truncated, {len(docs)} documents total")

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": payload})
    except KeyboardInterrupt:
        crash = "interrupted"
    except Exception as exc:
        crash = f"{type(exc).__name__}: {exc}"
    finally:
        writer.close()

    if crash:
        verdict, reason = "fail", crash
    elif halted:
        verdict, reason = "fail", f"halted by supervisor: {halted}"
    elif not finished:
        verdict, reason = "fail", f"hit the {config.MAX_STEPS}-step ceiling with no final answer"
    elif last_docs is None:
        verdict, reason = "fail", "no pipeline ever returned documents"
    else:
        ok, reason = tasks.verify(last_docs, task["check"])
        verdict = "pass" if ok else "fail"

    # verdict_reason is one field beyond the given schema. The reflector needs the
    # deterministic reason a run failed; without it, scars get written from guesswork.
    db.runs.update_one({"_id": run_id}, {"$set": {
        "steps_taken": steps_taken, "verdict": verdict, "verdict_reason": reason,
        "ended_at": db.utcnow()}})
    promoted = promote(scars_used, verdict)

    result = {"run_id": run_id, "verdict": verdict, "reason": reason, "mode": mode,
              "steps_taken": steps_taken, "median": median, "task_id": task["task_id"],
              "scars_used": scars_used, "promoted": promoted}
    ui("verdict", result)
    return result


class ConsoleUI:
    """Full-width single-run rendering for `python agent.py`."""

    def __call__(self, event, payload):
        getattr(self, f"on_{event}", lambda _p: None)(payload)

    def on_start(self, p):
        task, median = p["task"], p["median"]
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bright_black", justify="right")
        grid.add_column()
        grid.add_row("model", Text(config.AGENT_MODEL, style="bold white"))
        grid.add_row("mode", Text(p["mode"], style="bold cyan" if p["mode"] == "warm" else "bold yellow"))
        grid.add_row("task", Text(f"{task['task_id']}  ({task['family']})"))
        grid.add_row("run", Text(str(p["run_id"]), style="bright_black"))
        grid.add_row("median", Text("—  (no prior runs in this family)" if median is None
                                    else f"{median:g} steps", style="bright_black"))
        console.print(Panel(grid, title="[bold]SCAR · agent[/]", border_style="bright_black"))
        console.print(Panel(task["question"], title="[bold]task[/]", border_style="bright_black"))

    def on_scars(self, p):
        if not p["scars"]:
            console.print("[bright_black]no scars retrieved — nothing learned yet[/]\n")
            return
        body = []
        for scar in p["scars"]:
            tag = "active" if scar.get("status") == "active" else "candidate"
            body.append(Text.assemble(
                (f"[{tag}] ", "bold cyan" if tag == "active" else "bold bright_black"),
                (scar["text"], "cyan"),
                (f"\n        born from run {scar.get('born_from_run')}"
                 f"  ·  score {scar.get('score', 0):.3f}", "bright_black")))
        console.print(Panel(Text("\n").join(body),
                            title=f"[bold cyan]{len(p['scars'])} scar(s) injected before first token[/]",
                            subtitle=f"[bright_black]via {p['source']}[/]", border_style="cyan"))

    def on_turn(self, p):
        median = "" if p["median"] is None else f"  ·  median {p['median']:g}"
        console.rule(f"[bright_black]step {p['n']}/{p['max']}{median}[/]", align="left")

    def on_thought(self, p):
        console.print(Text(p["text"], style="bright_black"))

    def on_final(self, p):
        console.print(Text(p["text"], style="white"))

    def on_tool_call(self, p):
        console.print(f"[yellow]→ run_pipeline([bold]{p['collection']}[/])[/]")
        body = json.dumps(p["pipeline"], indent=2) if p["pipeline"] is not None else str(p["raw"])
        console.print(Syntax(body, "json", theme="ansi_dark", background_color="default"))

    def on_tool_result(self, p):
        if p.get("error"):
            console.print(f"[red]✗ {p['error']}[/]")
            return
        console.print(f"[green]✓ {p['n']} document(s)[/]")
        preview = json.dumps(jsonable(p["docs"][:3]), indent=2)
        if p["n"] > 3:
            preview += f"\n... {p['n'] - 3} more"
        console.print(Syntax(preview, "json", theme="ansi_dark", background_color="default"))

    def on_halt(self, p):
        console.print(Panel(p["reason"], title="[bold red]HALTED BY SUPERVISOR[/]", border_style="red"))

    def on_verdict(self, p):
        color = "green" if p["verdict"] == "pass" else "red"
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bright_black", justify="right")
        grid.add_column()
        grid.add_row("verdict", Text(p["verdict"].upper(), style=f"bold {color}"))
        grid.add_row("reason", Text(p["reason"]))
        grid.add_row("steps", Text(f"{p['steps_taken']}" + (
            "" if p["median"] is None else f"  ·  family median {p['median']:g}")))
        if p["scars_used"]:
            grid.add_row("scars used", Text(str(len(p["scars_used"])), style="cyan"))
        if p["promoted"]:
            grid.add_row("promoted", Text(f"{len(p['promoted'])} candidate → active",
                                          style="bold green"))
        grid.add_row("model", Text(config.AGENT_MODEL, style="bold white"))
        grid.add_row("run", Text(str(p["run_id"]), style="bright_black"))
        console.print(Panel(grid, border_style=color))


def main():
    parser = argparse.ArgumentParser(description="Run one Scar task.")
    parser.add_argument("--task", default="random", help="task_id, or 'random' (default)")
    parser.add_argument("--mode", default="cold", choices=["cold", "warm"])
    parser.add_argument("--list", action="store_true", help="list tasks and exit")
    args = parser.parse_args()

    if args.list:
        for t in tasks.TASK_LIST:
            console.print(f"[bold]{t['task_id']:<22}[/] [bright_black]{t['family']}[/]")
        return 0

    config.require_env()
    task = random.choice(tasks.TASK_LIST) if args.task == "random" else tasks.TASKS.get(args.task)
    if task is None:
        console.print(f"[red]unknown task {args.task!r}[/] — try --list")
        return 2

    db.ensure_indexes()
    if args.mode == "warm":
        db.ensure_vector_index()
    result = execute(task, args.mode, ConsoleUI())
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())

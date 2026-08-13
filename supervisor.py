"""Runtime supervision. Long-running.

Watches db.steps for inserts and scores each one with a single cheap judge call.
Writing an eval with verdict "halt" is the flag: agent.py polls for it between
turns and stops itself.
"""
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

console = Console()

JUDGE_SYSTEM = """You supervise a data-analysis agent writing MongoDB aggregation pipelines while it runs.

Score two independent risks, each 0.0 to 1.0:
- immediate_risk: is THIS step itself broken — malformed pipeline, an error, an empty result,
  nulls or wrong types where the task asked for numbers, a claim unsupported by the data.
- drift_risk: has the run stopped serving the ORIGINAL task — resubmitting a pipeline that
  already failed, thrashing between approaches, or wandering onto a different question.

Then pick a verdict:
- "continue": the run is healthy, or is recovering sensibly from a mistake it noticed.
- "warn": real trouble, but still recoverable.
- "halt": the run cannot recover and should be stopped now.

Exploring and correcting course is normal and healthy. Do not punish a first failed attempt.
Reserve "halt" for genuine dead ends.

Reply with JSON only, no prose:
{"immediate_risk": 0.0, "drift_risk": 0.0, "verdict": "continue", "reason": "one short sentence"}"""

_run_cache = {}


def run_info(run_id):
    if run_id not in _run_cache:
        run = db.runs.find_one({"_id": run_id}, {"task": 1, "task_id": 1, "mode": 1}) or {}
        _run_cache[run_id] = run
    return _run_cache[run_id]


def trajectory(run_id, upto_idx, limit=6):
    docs = list(db.steps.find({"run_id": run_id, "idx": {"$lt": upto_idx}},
                              {"idx": 1, "kind": 1, "content": 1})
                .sort("idx", -1).limit(limit))
    docs.reverse()
    lines = [f"[{d['idx']}] {d['kind']}: {(d.get('content') or '')[:300]}" for d in docs]
    return "\n".join(lines) or "(this is the first step of the run)"


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


def thrash_count(run_id, pipeline):
    """How many times this exact pipeline has been submitted in this run. Pure code."""
    if pipeline is None:
        return 0
    key = json.dumps(pipeline, sort_keys=True, default=str)
    seen = 0
    for step in db.steps.find({"run_id": run_id, "kind": "tool_call"}, {"pipeline": 1}):
        if step.get("pipeline") is not None and json.dumps(
                step["pipeline"], sort_keys=True, default=str) == key:
            seen += 1
    return seen


def judge(client, step):
    run = run_info(step["run_id"])
    user = (f"TASK THE AGENT WAS GIVEN:\n{run.get('task', '(unknown)')}\n\n"
            f"RECENT TRAJECTORY:\n{trajectory(step['run_id'], step['idx'])}\n\n"
            f"LATEST STEP (idx {step['idx']}, kind {step['kind']}):\n"
            f"{(step.get('content') or '')[:1500]}")
    if step.get("error"):
        user += f"\n\nTHE TOOL RETURNED AN ERROR: {step['error']}"

    response = client.chat.completions.create(
        model=config.JUDGE_MODEL, temperature=0, max_tokens=220,
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": user}])
    parsed = parse_json(response.choices[0].message.content) or {}

    def risk(key):
        try:
            return max(0.0, min(1.0, float(parsed.get(key, 0.0))))
        except (TypeError, ValueError):
            return 0.0

    verdict = parsed.get("verdict")
    if verdict not in ("continue", "warn", "halt"):
        verdict = "continue"
    return {"immediate_risk": risk("immediate_risk"), "drift_risk": risk("drift_risk"),
            "verdict": verdict, "reason": str(parsed.get("reason", ""))[:300]}


def risk_style(value):
    return "green" if value < 0.34 else ("yellow" if value < 0.67 else "red")


def render(step, result, forced):
    verdict = result["verdict"]
    color = {"continue": "green", "warn": "yellow", "halt": "bold red"}[verdict]
    mark = {"continue": "·", "warn": "!", "halt": "×"}[verdict]
    line = Text()
    line.append(f" {mark} ", style=color)
    line.append(f"{str(step['run_id'])[-6:]} ", style="bright_black")
    line.append(f"step {step['idx']:<3} ", style="bright_black")
    line.append(f"{step['kind']:<12} ", style="white")
    line.append("imm ", style="bright_black")
    line.append(f"{result['immediate_risk']:.2f} ", style=risk_style(result["immediate_risk"]))
    line.append("drift ", style="bright_black")
    line.append(f"{result['drift_risk']:.2f} ", style=risk_style(result["drift_risk"]))
    line.append(f"{verdict.upper():<8} ", style=color)
    line.append(result["reason"], style="bright_black" if verdict == "continue" else "white")
    console.print(line)
    if forced:
        console.print(f"   [bold red]↳ deterministic halt: {forced}[/]")


def main():
    config.require_env()
    db.ensure_indexes()
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY, timeout=60.0)

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bright_black", justify="right")
    grid.add_column()
    grid.add_row("judge", Text(config.JUDGE_MODEL, style="bold white"))
    grid.add_row("watching", Text(f"{config.DB_NAME}.steps change stream"))
    grid.add_row("halt rule", Text(f"same pipeline {config.THRASH_LIMIT}x, or judge says halt"))
    console.print(Panel(grid, title="[bold]SCAR · supervisor[/]", border_style="bright_black"))

    with db.steps.watch(full_document="updateLookup") as stream:
        for change in stream:
            if change.get("operationType") != "insert":
                continue
            step = change["fullDocument"]
            try:
                result = judge(client, step)
            except Exception as exc:
                console.print(f"[red]judge error: {type(exc).__name__}: {exc}[/]")
                continue

            forced = None
            if step["kind"] == "tool_call":
                repeats = thrash_count(step["run_id"], step.get("pipeline"))
                if repeats >= config.THRASH_LIMIT:
                    forced = f"identical pipeline submitted {repeats} times"
                    result.update(verdict="halt", drift_risk=1.0, reason=forced)

            db.evals.insert_one({
                "run_id": step["run_id"], "step_idx": step["idx"],
                "immediate_risk": result["immediate_risk"], "drift_risk": result["drift_risk"],
                "verdict": result["verdict"], "reason": result["reason"], "ts": db.utcnow()})
            render(step, result, forced)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[bright_black]supervisor stopped[/]")

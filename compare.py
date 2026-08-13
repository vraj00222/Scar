"""THE DEMO. Same task, same model, cold vs warm, side by side, live.

The only difference between the two columns is whether scars were injected before
the first token.
"""
import argparse
import json
import random
import sys
import threading

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import agent
import config
import db
import embed
import tasks

console = Console()


class Column:
    """Collects a run's events as lines for one side of the split view."""

    def __init__(self, mode, style):
        self.mode = mode
        self.style = style
        self.lines = []
        self.done = False
        self.result = None
        self.steps = 0
        self.lock = threading.Lock()

    def add(self, text, style=""):
        with self.lock:
            self.lines.append(Text(text, style=style))

    def __call__(self, event, payload):
        getattr(self, f"on_{event}", lambda _p: None)(payload)

    def on_start(self, p):
        self.add(f"run {str(p['run_id'])[-6:]}", "bright_black")

    def on_scars(self, p):
        if not p["scars"]:
            self.add("no scars available", "bright_black")
            return
        self.add(f"{len(p['scars'])} scar(s) injected before first token:", "bold cyan")
        for scar in p["scars"]:
            tag = "active" if scar.get("status") == "active" else "cand"
            self.add(f"  [{tag}] {scar['text']}", "cyan")
            self.add(f"         born run {str(scar.get('born_from_run'))[-6:]}", "bright_black")

    def on_turn(self, p):
        self.steps = p["n"]
        self.add("")
        self.add(f"── step {p['n']}/{p['max']}", "bright_black")

    def on_thought(self, p):
        self.add(_clip(p["text"], 300), "bright_black")

    def on_final(self, p):
        self.add(_clip(p["text"], 400), "white")

    def on_tool_call(self, p):
        self.add(f"→ run_pipeline({p['collection']})", "yellow")
        body = json.dumps(p["pipeline"], separators=(",", ":")) if p["pipeline"] is not None \
            else str(p["raw"])
        self.add(f"  {_clip(body, 400)}", "bright_black")

    def on_tool_result(self, p):
        if p.get("error"):
            self.add(f"✗ {_clip(p['error'], 200)}", "red")
        else:
            self.add(f"✓ {p['n']} document(s)", "green")
            self.add(f"  {_clip(json.dumps(agent.jsonable(p['docs'][:2])), 300)}", "bright_black")

    def on_halt(self, p):
        self.add(f"HALTED: {p['reason']}", "bold red")

    def on_verdict(self, p):
        self.result = p
        color = "bold green" if p["verdict"] == "pass" else "bold red"
        self.add("")
        self.add(f"{p['verdict'].upper()} in {p['steps_taken']} steps", color)
        self.add(_clip(p["reason"], 300), "white")
        if p["promoted"]:
            self.add(f"promoted {len(p['promoted'])} candidate → active", "bold green")

    def render(self, height):
        with self.lock:
            visible = self.lines[-height:]
        head = Text.assemble(("model  ", "bright_black"), (config.AGENT_MODEL, "bold white"),
                             ("\nmode   ", "bright_black"), (self.mode.upper(), self.style))
        body = Group(head, Text("─" * 3, style="bright_black"), *visible)
        border = self.style if self.result is None else (
            "green" if self.result["verdict"] == "pass" else "red")
        return Panel(body, title=f"[{self.style}]{self.mode.upper()}[/]", border_style=border)


def _clip(text, n):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + "…"


def main():
    parser = argparse.ArgumentParser(description="Cold vs warm, side by side.")
    parser.add_argument("--task", default="random", help="task_id, or 'random'")
    args = parser.parse_args()

    config.require_env()
    task = random.choice(tasks.TASK_LIST) if args.task == "random" else tasks.TASKS.get(args.task)
    if task is None:
        console.print(f"[red]unknown task {args.task!r}[/] — try `python agent.py --list`")
        return 2

    db.ensure_indexes()
    db.ensure_vector_index()

    available = db.scars.count_documents({"status": {"$in": ["active", "candidate"]}})
    console.print(Panel(
        Text.assemble((task["question"], "white"),
                      (f"\n\nmodel {config.AGENT_MODEL}  ·  embeddings {embed.BACKEND}"
                       f"  ·  {available} scar(s) in the pool", "bright_black")),
        title=f"[bold]SCAR · compare · {task['task_id']}[/]", border_style="bright_black"))
    if available == 0:
        console.print("[yellow]the scar pool is empty — warm will look exactly like cold. "
                      "Run grind.py and reflector.py first.[/]\n")

    cold = Column("cold", "yellow")
    warm = Column("warm", "cyan")

    def launch(column):
        try:
            agent.execute(task, column.mode, column)
        except Exception as exc:
            column.add(f"crashed: {type(exc).__name__}: {exc}", "bold red")
        finally:
            column.done = True

    threads = [threading.Thread(target=launch, args=(c,), daemon=True) for c in (cold, warm)]

    with Live(console=console, refresh_per_second=8, screen=False) as live:
        for thread in threads:
            thread.start()
        while True:
            height = max(8, console.size.height - 14)
            grid = Table.grid(expand=True, padding=(0, 1))
            grid.add_column(ratio=1)
            grid.add_column(ratio=1)
            grid.add_row(cold.render(height), warm.render(height))
            live.update(grid)
            if cold.done and warm.done:
                break
            for thread in threads:
                thread.join(timeout=0.12)

    summary = Table(show_header=True, header_style="bold", border_style="bright_black")
    summary.add_column("")
    summary.add_column("COLD", style="yellow")
    summary.add_column("WARM", style="cyan")
    for label, get in (("verdict", lambda r: r["verdict"].upper()),
                       ("steps", lambda r: str(r["steps_taken"])),
                       ("scars injected", lambda r: str(len(r["scars_used"]))),
                       ("promoted", lambda r: str(len(r["promoted"]))),
                       ("reason", lambda r: _clip(r["reason"], 90))):
        summary.add_row(label,
                        get(cold.result) if cold.result else "—",
                        get(warm.result) if warm.result else "—")
    console.print(summary)

    if cold.result and warm.result:
        c, w = cold.result, warm.result
        if c["verdict"] == "fail" and w["verdict"] == "pass":
            console.print(Panel(
                Text(f"Same model, same task, same tools. Cold failed and warm passed.\n"
                     f"The only difference was {len(w['scars_used'])} scar(s) injected "
                     f"before the first token.", style="bold green"),
                title="[bold green]TRANSFER[/]", border_style="green"))
        elif c["verdict"] == w["verdict"] and w["steps_taken"] < c["steps_taken"]:
            console.print(Panel(
                Text(f"Both {w['verdict']}, but warm needed {c['steps_taken'] - w['steps_taken']} "
                     f"fewer steps.", style="bold cyan"), border_style="cyan"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[bright_black]compare stopped[/]")

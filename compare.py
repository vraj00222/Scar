"""THE DEMO. Same task, same model, cold vs warm, side by side, live.

The only difference between the two columns is whether scars were injected before
the first token.
"""
import argparse
import json
import math
import random
import sys
import threading
import time

from rich import box
from rich.align import Align
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

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Column:
    """Collects a run's events as lines for one side of the split view."""

    def __init__(self, mode, style):
        self.mode = mode
        self.style = style
        self.lines = []
        self.done = False
        self.result = None
        self.steps = 0
        self.phase = "starting"
        self.t0 = time.time()
        self.t_end = None
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
            self.add("∅ no memory — starting from zero", "bold bright_black")
            return
        self.add(f"🧬 {len(p['scars'])} SCARS INJECTED BEFORE FIRST TOKEN", "bold black on cyan")
        for scar in p["scars"]:
            tag = "active" if scar.get("status") == "active" else "cand"
            self.add(f"  [{tag}] {scar['text']}", "cyan")
            self.add(f"         born run {str(scar.get('born_from_run'))[-6:]}", "bright_black")

    def on_turn(self, p):
        self.steps = p["n"]
        self.phase = "thinking"
        self.add("")
        self.add(f"── step {p['n']}/{p['max']}", "bright_black")

    def on_thought(self, p):
        self.add(_clip(p["text"], 200), "bright_black")

    def on_final(self, p):
        self.phase = "answering"
        self.add(_clip(p["text"], 240), "white")

    def on_tool_call(self, p):
        self.phase = "querying mongo"
        self.add(f"→ run_pipeline({p['collection']})", "yellow")
        body = json.dumps(p["pipeline"], separators=(",", ":")) if p["pipeline"] is not None \
            else str(p["raw"])
        self.add(f"  {_clip(body, 160)}", "bright_black")

    def on_tool_result(self, p):
        self.phase = "thinking"
        if p.get("error"):
            self.add(f"✗ {_clip(p['error'], 160)}", "bold red")
        else:
            self.add(f"✓ {p['n']} document(s)", "green")
            self.add(f"  {_clip(json.dumps(agent.jsonable(p['docs'][:2])), 160)}", "bright_black")

    def on_halt(self, p):
        self.add(f"🛑 HALTED BY SUPERVISOR: {p['reason']}", "bold white on red")

    def on_verdict(self, p):
        self.result = p
        self.t_end = time.time()
        won = p["verdict"] == "pass"
        self.add("")
        self.add(f"  {'✓ PASS' if won else '✗ FAIL'} in {p['steps_taken']} steps  ",
                 "bold white on green" if won else "bold white on red")
        self.add(_clip(p["reason"], 300), "white")
        if p["promoted"]:
            self.add(f"⬆ promoted {len(p['promoted'])} candidate → active", "bold black on green")

    def render(self, budget, width):
        """Keep the pane inside `budget` PHYSICAL rows.

        Counting logical lines is not enough: at half terminal width a single
        pipeline line wraps into many rows, so the panel outgrew the screen and
        made Live scroll instead of repaint.
        """
        with self.lock:
            snapshot = list(self.lines)
        chosen, used = [], 0
        for text in reversed(snapshot):
            cost = max(1, math.ceil(len(text.plain) / width)) if text.plain else 1
            if chosen and used + cost > budget:
                break
            chosen.append(text)
            used += cost
        chosen.reverse()
        head = Text.assemble(("model  ", "bright_black"), (config.AGENT_MODEL, "bold white"),
                             ("\nmode   ", "bright_black"), (self.mode.upper(), self.style))
        body = Group(head, Text("─" * 3, style="bright_black"), *chosen)

        if self.result is None:
            frame = SPINNER[int(time.time() * 10) % len(SPINNER)]
            title = f"[{self.style}]{frame} {self.mode.upper()} · {self.phase}[/]"
            border = self.style
        else:
            won = self.result["verdict"] == "pass"
            title = (f"[bold white on green] ✓ {self.mode.upper()} · PASS [/]" if won
                     else f"[bold white on red] ✗ {self.mode.upper()} · FAIL [/]")
            border = "green" if won else "red"

        elapsed = (self.t_end or time.time()) - self.t0
        bar = "▰" * self.steps + "▱" * (config.MAX_STEPS - self.steps)
        subtitle = (f"[{self.style}]{bar}[/] [bright_black]{self.steps}/{config.MAX_STEPS}"
                    f" · {elapsed:.0f}s[/]")
        return Panel(body, title=title, subtitle=subtitle, border_style=border)


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
        title=f"[bold black on white] ⚡ SCAR [/][bold] COLD vs WARM · {task['task_id']}[/]",
        border_style="bright_black"))
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
            budget = max(8, console.size.height - 12)
            width = max(20, console.size.width // 2 - 6)
            grid = Table.grid(expand=True, padding=(0, 1))
            grid.add_column(ratio=1)
            grid.add_column(ratio=1)
            grid.add_row(cold.render(budget, width), warm.render(budget, width))
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

    for column in (cold, warm):
        for err in (column.result or {}).get("writer_errors", []):
            console.print(f"[red]{column.mode} step writer dropped a step: {err}[/]")

    if cold.result and warm.result:
        c, w = cold.result, warm.result
        if c["verdict"] == "fail" and w["verdict"] == "pass":
            lines = Text()
            lines.append("SAME MODEL · SAME TASK · SAME TOOLS · SAME STEP BUDGET\n\n",
                         style="bold white")
            lines.append("   COLD ", style="bold yellow")
            lines.append(" ✗ FAILED ", style="bold white on red")
            lines.append("        WARM ", style="bold cyan")
            lines.append(" ✓ PASSED ", style="bold white on green")
            lines.append(f"\n\nThe only difference: {len(w['scars_used'])} scars "
                         f"MongoDB remembered,\ninjected before the first token.",
                         style="bold green")
            console.print(Panel(Align.center(lines), box=box.DOUBLE,
                                title="[bold white on green] ⚡ TRANSFER PROVEN ⚡ [/]",
                                border_style="bold green", padding=(1, 4)))
        elif c["verdict"] == w["verdict"] and w["steps_taken"] < c["steps_taken"]:
            console.print(Panel(
                Align.center(Text(
                    f"Both {w['verdict']}, but warm needed "
                    f"{c['steps_taken'] - w['steps_taken']} fewer steps.\n"
                    f"Memory is a shortcut past every trap cold has to rediscover.",
                    style="bold cyan")),
                box=box.DOUBLE, title="[bold cyan] ⚡ FEWER STEPS ⚡ [/]",
                border_style="cyan", padding=(1, 4)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[bright_black]compare stopped[/]")

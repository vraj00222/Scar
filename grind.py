"""Run random tasks cold, forever, so the system has failures to learn from.

Pair it with supervisor.py and reflector.py in other panes: this pane makes the
mistakes, those panes catch them and turn them into scars.
"""
import argparse
import random
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import agent
import config
import db
import tasks

console = Console()


class QuietUI:
    """Swallows step detail. grind prints one line per finished run."""

    def __call__(self, event, payload):
        pass


def scar_counts():
    return (db.scars.count_documents({"status": "active"}),
            db.scars.count_documents({"status": "candidate"}))


def main():
    parser = argparse.ArgumentParser(description="Loop cold runs to accumulate scars.")
    parser.add_argument("--limit", type=int, default=0, help="stop after N runs (0 = forever)")
    parser.add_argument("--sleep", type=float, default=2.0, help="seconds between runs")
    parser.add_argument("--mode", default="cold", choices=["cold", "warm"])
    args = parser.parse_args()

    config.require_env()
    db.ensure_indexes()

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bright_black", justify="right")
    grid.add_column()
    grid.add_row("model", Text(config.AGENT_MODEL, style="bold white"))
    grid.add_row("mode", Text(args.mode, style="bold yellow"))
    grid.add_row("tasks", Text(f"{len(tasks.TASK_LIST)} across "
                               f"{len({t['family'] for t in tasks.TASK_LIST})} families"))
    grid.add_row("stop", Text("ctrl-c" if not args.limit else f"after {args.limit} runs"))
    console.print(Panel(grid, title="[bold]SCAR · grind[/]", border_style="bright_black"))

    total = passed = 0
    started = time.time()
    try:
        while not args.limit or total < args.limit:
            task = random.choice(tasks.TASK_LIST)
            total += 1
            console.print(f"[bright_black]{total:>4}[/] [white]{task['task_id']:<22}[/]",
                          end="")
            try:
                result = agent.execute(task, args.mode, QuietUI())
            except Exception as exc:
                console.print(f" [red]crashed: {type(exc).__name__}: {exc}[/]")
                time.sleep(args.sleep)
                continue

            ok = result["verdict"] == "pass"
            passed += ok
            active, candidate = scar_counts()
            console.print(
                f" [{'green' if ok else 'red'}]{result['verdict'].upper():<5}[/]"
                f" [bright_black]{result['steps_taken']:>2} steps[/]"
                f"  [bright_black]pass {passed}/{total}[/]"
                f"  [cyan]scars {active} active / {candidate} cand[/]"
                f"  [bright_black]{result['reason'][:70]}[/]")
            time.sleep(args.sleep)
    except KeyboardInterrupt:
        pass

    active, candidate = scar_counts()
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bright_black", justify="right")
    summary.add_column()
    summary.add_row("runs", Text(str(total)))
    summary.add_row("passed", Text(f"{passed}/{total}" + (
        f"  ({passed / total:.0%})" if total else "")))
    summary.add_row("scars", Text(f"{active} active · {candidate} candidate", style="cyan"))
    summary.add_row("elapsed", Text(f"{time.time() - started:.0f}s"))
    summary.add_row("model", Text(config.AGENT_MODEL, style="bold white"))
    console.print(Panel(summary, title="[bold]grind stopped[/]", border_style="bright_black"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

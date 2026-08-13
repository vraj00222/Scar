#!/bin/bash
# One-word launchers for each SCAR pane. Run each in its own terminal window.
cd "$(dirname "$0")"
PY=./.venv/bin/python

case "$1" in
  watch)    exec $PY supervisor.py ;;                      # pane 2: judges every step live
  learn)    exec $PY reflector.py ;;                       # pane 3: turns failures into scars
  practice) exec $PY grind.py --limit "${2:-6}" ;;         # optional: accumulate scars
  vs|show)  exec $PY compare.py --task "${2:-top_genre_by_decade}" ;;  # THE DEMO: cold vs warm
  fail)     exec $PY agent.py --task "${2:-top_genre_by_decade}" --mode cold ;;
  pass)     exec $PY agent.py --task "${2:-top_genre_by_decade}" --mode warm ;;
  tasks)    exec $PY agent.py --list ;;
  *)
    echo "usage: ./demo.sh <command>"
    echo "  watch            start the supervisor (keep running in its own window)"
    echo "  learn            start the reflector (keep running in its own window)"
    echo "  practice [n]     run n cold runs to accumulate scars (default 6)"
    echo "  vs [task]        THE DEMO: cold vs warm side by side"
    echo "  fail [task]      one cold run (usually fails on trap tasks)"
    echo "  pass [task]      one warm run (retrieves scars first)"
    echo "  tasks            list all task ids"
    ;;
esac

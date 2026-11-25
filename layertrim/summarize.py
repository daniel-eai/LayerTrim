"""Print benchmark means and sample standard deviations for seeds 0, 1 and 2."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from .models import ROOT
from .data import SOURCES


def table(rows):
    groups = defaultdict(lambda: defaultdict(dict))
    for alias, budget, task, seed, correct, total in rows:
        groups[alias, budget][task][int(seed)] = 100 * int(correct) / int(total)
    print("model | budget | " + " | ".join(SOURCES) + " | average")
    for (alias, budget), tasks in sorted(groups.items()):
        averages, cells = [], []
        for task in SOURCES:
            vals = tasks[task]
            if set(vals) >= {0, 1, 2}:
                runs = [vals[k] for k in (0, 1, 2)]
                averages.append(mean(runs))
                cells.append(f"{mean(runs):.1f} ± {stdev(runs):.1f}")
            else:
                cells.append(f"{len(set(vals) & {0, 1, 2})}/3 seeds")
        avg = f"{mean(averages):.1f}" if len(averages) == len(SOURCES) else "incomplete"
        print(" | ".join([alias, budget, *cells, avg]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", action="store_true")
    p.add_argument("--results", type=Path, default=Path("results"))
    args = p.parse_args()
    if args.reference:
        with (ROOT / "assets/reference_counts.csv").open() as f:
            reader = csv.reader(f)
            next(reader)
            table(reader)
    else:
        rows = []
        for path in args.results.glob("*/*/seed*/score_*.json"):
            alias, budget, seed, _ = path.relative_to(args.results).parts
            r = json.loads(path.read_text())
            rows.append((alias, budget, r["task"], seed[4:], r["n_correct"], r["n_total"]))
        if not rows:
            p.exit(1, f"No completed scores under {args.results}\n")
        table(rows)


if __name__ == "__main__":
    main()

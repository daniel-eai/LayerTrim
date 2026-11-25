"""Prepare benchmark files in the same item and option order used for evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

from .models import ROOT

SOURCES = {
    "aime25": ("MathArena/aime_2025", None, "train", "c94da77eb22bbd6439e62a323bec18493a421302"),
    "aime26": ("MathArena/aime_2026", None, "train", "d2de22f3c656b4f56cf8981212186377d1e23bc3"),
    "gpqa": ("Idavidrein/gpqa", "gpqa_diamond", "train", None),
    "humaneval": ("openai/openai_humaneval", None, "test", None),
    "mbpp": ("evalplus/mbppplus", None, "test", "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"),
}
FILES = {"gpqa": "gpqa_diamond", "mbpp": "mbppplus"}


def digest(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def arrange(task, rows, order):
    if task == "gpqa":
        questions = {digest(r["Question"]): r for r in rows}
        result = []
        for i, (key, *choices) in enumerate(order):
            r = questions[key]
            options = [r["Correct Answer"], *(r[f"Incorrect Answer {j}"] for j in (1, 2, 3))]
            lookup = {digest(o): o for o in options}
            ordered = [lookup[h] for h in choices]
            problem = r["Question"] + "\n\n" + "\n".join(
                f"{letter}. {o}" for letter, o in zip("ABCD", ordered))
            result.append(dict(id=str(i), problem=problem, answer="ABCD"[choices.index(digest(options[0]))]))
        return result
    if task.startswith("aime"):
        keyed = {str(r["problem_idx"]): dict(id=str(r["problem_idx"]), problem=r["problem"],
                                             answer=str(r["answer"])) for r in rows}
    else:
        keys = (["task_id", "prompt", "canonical_solution", "test", "entry_point"]
                if task == "humaneval" else ["task_id", "prompt", "code", "test", "test_list"])
        keyed = {str(r["task_id"]): {k: r[k] for k in keys} for r in rows}
    return [keyed[k] for k in order]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tasks", nargs="*", help="Defaults to all five tasks")
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()
    from datasets import load_dataset
    order = json.loads((ROOT / "assets/benchmark_order.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for task in args.tasks or SOURCES:
        repo, subset, split, revision = SOURCES[task]
        name = FILES.get(task, task)
        rows = load_dataset(repo, subset, split=split, revision=revision)
        prepared = arrange(task, rows, order[name])
        dest = args.out / f"{name}.json"
        dest.write_text(json.dumps(prepared, ensure_ascii=False))
        print(f"{dest}: {len(prepared)} items")


if __name__ == "__main__":
    main()

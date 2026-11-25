"""Score a full set of HumanEval or MBPP+ generations inside the scoring container."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from code_answers import build_program, extract_code


def execute(program, timeout):
    with tempfile.TemporaryDirectory(prefix="code-score-") as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(program)
        with (Path(tmp) / "stderr").open("w+") as err:
            child = subprocess.Popen([sys.executable, "-I", str(path)], cwd=tmp,
                                     start_new_session=True, stdout=subprocess.DEVNULL, stderr=err)
            try:
                child.wait(timeout=timeout)
                status = "pass" if child.returncode == 0 else "fail"
            except subprocess.TimeoutExpired:
                status = "timeout"
            finally:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            err.seek(0)
            return status, err.read(4000)[-800:]


def mbpp_test(problem):
    test = problem["test"]
    if str(problem["task_id"]) in {"737", "787", "794"}:
        needle = "exact_match = exp == (out is not None)"
        if test.count(needle) != 1 or "assert exact_match" in test:
            raise ValueError("MBPP+ snapshot differs from the archived extended test")
        test = test.replace(needle, needle + "\n    assert exact_match")
    if not test.strip():
        raise ValueError("Missing extended MBPP+ test")
    return test


def main():
    task = sys.argv[1]
    if task not in {"humaneval", "mbpp"}:
        raise ValueError(task)
    # The host sends two JSON documents on separate physical lines.
    gen = json.loads(sys.stdin.readline())
    problems = json.loads(sys.stdin.readline())
    if gen["task"] != task or len(gen["items"]) != len(problems) or not problems:
        raise ValueError("Expected one generation per benchmark item")
    timeout = 10 if task == "humaneval" else 60
    rows = []
    for i, (item, problem) in enumerate(zip(gen["items"], problems)):
        if item["i"] != i or str(item["gold"]) != str(problem["task_id"]):
            raise ValueError(f"Generation and problem disagree at item {i}")
        if task == "humaneval":
            program, mode = build_program(item["text"], problem)
        else:
            program = extract_code(item["text"]) + "\n\n" + mbpp_test(problem) + "\n"
            mode = "full_function"
        status, detail = execute(program, timeout)
        rows.append(dict(i=i, task_id=str(problem["task_id"]), ok=status == "pass",
                         exec_status=status, exec_detail=detail, score_mode=mode))
    print(json.dumps(dict(task=task, n_total=len(rows), n_correct=sum(r["ok"] for r in rows),
                          scorer_version="main-code-score-20260908-v1", per_item=rows)))


if __name__ == "__main__":
    main()

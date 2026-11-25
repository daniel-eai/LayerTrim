import re
import textwrap

CLOSE_MARKERS = (
    "</think>",
    "</thinking>",
    "[/THINK]",
    "<|channel|>final",
    "<|start|>assistant<|channel|>final",
    "assistantfinal",
)


def answer_region(text: str) -> str:
    for marker in CLOSE_MARKERS:
        if marker in text:
            return text.rsplit(marker, 1)[1]
    return text


def extract_code(text: str) -> str:
    text = answer_region(text)
    matches = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1]
    matches = re.findall(r"```\n?(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1]
    return text


def has_entry_point_definition(code: str, entry_point: str) -> bool:
    pattern = rf"(?m)^\s*(?:async\s+def|def)\s+{re.escape(entry_point)}\s*\("
    return re.search(pattern, code) is not None


def prompt_prelude(prompt: str, entry_point: str) -> str:
    """Keep imports/constants that appear before the function under test."""
    pattern = rf"(?m)^\s*(?:async\s+def|def)\s+{re.escape(entry_point)}\s*\("
    match = re.search(pattern, prompt)
    if match is None:
        raise ValueError(f"entry point {entry_point!r} not found in HumanEval prompt")
    return prompt[: match.start()]


def build_program(prediction: str, problem: dict) -> tuple[str, str]:
    code = textwrap.dedent(extract_code(prediction)).strip("\n") + "\n"
    if has_entry_point_definition(code, problem["entry_point"]):
        implementation = prompt_prelude(
            problem["prompt"], problem["entry_point"]
        ) + code
        mode = "full_function"
    else:
        body = textwrap.indent(code, "    ")
        implementation = problem["prompt"] + body
        mode = "body_completion"
    program = (
        implementation
        + "\n\n"
        + problem["test"]
        + f"\ncheck({problem['entry_point']})\n"
    )
    return program, mode

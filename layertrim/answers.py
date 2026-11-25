import re

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
LETTER_RE = re.compile(r"([A-Da-d])")
CLOSE_MARKERS = ("</think>", "</thinking>", "[/THINK]", "<|channel|>final", "<|start|>assistant<|channel|>final", "assistantfinal")


def score_gpqa(pred_texts, gold):
    correct = 0
    per = []
    for i, (t, g) in enumerate(zip(pred_texts, gold)):
        raw = extract_boxed(t)
        p = None
        if raw:
            m = LETTER_RE.search(raw.strip())
            if m:
                p = m.group(1).upper()
        ok = (p == g) if p else False
        correct += ok
        per.append({"i": i, "pred": p, "gold": g, "ok": bool(ok),
                    "len": len(t), "close": is_closed(t)})
    return correct, per


def is_closed(text):
    return any(mk in text for mk in CLOSE_MARKERS)


def extract_boxed(text):
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def score_aime(pred_texts, gold):
    correct = 0
    per = []
    for i, (t, g) in enumerate(zip(pred_texts, gold)):
        p = extract_boxed(t)
        try:
            ok = int(str(p).replace(",", "").strip()) == int(str(g).strip()) if p else False
        except (TypeError, ValueError):
            ok = False
        correct += ok
        per.append({"i": i, "pred": p, "gold": g, "ok": bool(ok),
                    "len": len(t), "close": is_closed(t)})
    return correct, per

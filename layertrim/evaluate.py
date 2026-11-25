"""Generate one sample per problem with the physically trimmed decoder or its dense reference."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from .answers import score_aime, score_gpqa
from .data import FILES, SOURCES
from .models import MODELS, checkpoint
from .compact import load_compact
from .prompting import tokenizer_for


def prompt(task, row):
    if task.startswith("aime"):
        return row["problem"] + "\n\nGive your final answer as an integer in \\boxed{}."
    if task == "gpqa":
        return row["problem"] + "\n\nGive only the letter of the correct option in \\boxed{}."
    if task == "humaneval":
        return ("Complete the following Python function. Only output the completed function body "
                "in a ```python``` block.\n\n" + row["prompt"])
    example = row["test_list"][0] if row["test_list"] else ""
    return (row["prompt"] + "\n\nYour code should satisfy this test:\n" + example +
            "\n\nOnly output the completed function in a ```python``` block.")


def save(path, value):
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(value, f)
    Path(f.name).replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("alias", choices=MODELS)
    p.add_argument("budget", choices=["dense", "7", "9"])
    p.add_argument("tasks", nargs="+", choices=SOURCES)
    p.add_argument("--model", help="Override the dense or compact checkpoint directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cuda:0, or cpu")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, GenerationConfig, set_seed
    if args.budget == "dense":
        path = checkpoint(args.alias, args.model)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                      device_map=args.device).eval()
    else:
        path = Path(args.model or f"checkpoints/{args.alias}/n{args.budget}")
        plan = json.loads((path / "layertrim.json").read_text())
        if plan["model"] != args.alias or plan["budget"] != int(args.budget):
            p.error("The checkpoint plan must match the requested model and budget")
        model = load_compact(path, args.device)
    encode, decode = tokenizer_for(args.alias, path)
    limit = min(33792, model.config.max_position_embeddings)
    generation = GenerationConfig(do_sample=True, temperature=0.6, top_p=0.95, top_k=0,
                                  eos_token_id=model.generation_config.eos_token_id,
                                  pad_token_id=model.generation_config.pad_token_id,
                                  use_cache=True)
    folder = args.out / f"seed{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        set_seed(args.seed)
        dataset = args.data / (FILES.get(task, task) + ".json")
        rows = json.loads(dataset.read_text())
        items = []
        for i, row in enumerate(rows):
            ids = torch.tensor([encode(prompt(task, row))], device=model.get_input_embeddings().weight.device)
            cap = min(limit - 512, limit - ids.shape[1])
            if cap < 1:
                raise ValueError(f"Item {i} exceeds the model context length")
            with torch.inference_mode():
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        generation_config=generation, max_new_tokens=cap)
            tokens = output[0, ids.shape[1]:].tolist()
            items.append(dict(i=i, gold=row.get("answer", row.get("task_id")), text=decode(tokens),
                              n_out_tokens=len(tokens)))
        result = dict(task=task, alias=args.alias, budget=args.budget, backend="transformers",
                      parameters=sum(p.numel() for p in model.parameters()),
                      mlp_widths=[l.mlp.down_proj.in_features for l in model.model.layers],
                      torch_version=torch.__version__, transformers_version=transformers.__version__,
                      cuda_version=torch.version.cuda, seed=args.seed, temperature=0.6,
                      top_p=0.95, context=limit, items=items)
        save(folder / f"generations_{task}.json", result)
        if task in ("aime25", "aime26", "gpqa"):
            scorer = score_gpqa if task == "gpqa" else score_aime
            correct, _ = scorer([it["text"] for it in items], [it["gold"] for it in items])
            save(folder / f"score_{task}.json", dict(task=task, n_correct=correct, n_total=len(items)))
            print(f"{task}: {correct}/{len(items)}")


if __name__ == "__main__":
    main()

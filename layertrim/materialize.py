"""Export a compact checkpoint by slicing the selected MLP projections."""
import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .models import MODELS, checkpoint

MLP = re.compile(r"model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|gate_up_proj|down_proj)\.weight")


def trim_tensor(tensor, projection, indices, width, hidden):
    expected = {"down_proj": (hidden, width), "gate_up_proj": (2 * width, hidden),
                "gate_proj": (width, hidden), "up_proj": (width, hidden)}
    if tuple(tensor.shape) != expected[projection]:
        raise ValueError(f"Unexpected {projection} shape {tuple(tensor.shape)}")
    axis = 1 if projection == "down_proj" else 0
    if projection == "gate_up_proj":
        indices = torch.cat([indices, indices + width])
    return tensor.index_select(axis, indices)



def export(source, plan, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError(f"Choose a new output directory: {output}")
    config = json.loads((source / "config.json").read_text())
    width, hidden = plan["intermediate_size"], plan["hidden_size"]
    if (width, hidden) != (config["intermediate_size"], config["hidden_size"]):
        raise ValueError("Plan and checkpoint dimensions differ")
    kept = {int(i): torch.as_tensor(k) for i, k in plan["kept_neurons"].items()}
    if len(kept) != 2 * plan["budget"] or not kept:
        raise ValueError("Expected 2N selected layers")
    for i, k in kept.items():
        if not (0 <= i < config["num_hidden_layers"] and k.ndim == 1
                and k.dtype == torch.long and 0 < len(k) < width and k[0] >= 0
                and k[-1] < width and torch.all(k[1:] > k[:-1])):
            raise ValueError(f"Invalid retained indices for layer {i}")
    index_file = source / "model.safetensors.index.json"
    index = json.loads(index_file.read_text()) if index_file.exists() else None
    shards = sorted(set(index["weight_map"].values())) if index else ["model.safetensors"]
    found = {i: set() for i in kept}
    total_bytes = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "checkpoint"
        stage.mkdir()
        for name in shards:
            tensors = load_file(str(source / name))
            for key, tensor in tensors.items():
                match = MLP.fullmatch(key)
                if match and int(match[1]) in kept:
                    layer, proj = int(match[1]), match[2]
                    tensors[key] = trim_tensor(tensor, proj, kept[layer], width, hidden)
                    found[layer].add(proj)
                value = tensors[key]
                total_bytes += value.numel() * value.element_size()
            save_file(tensors, str(stage / name), metadata={"format": "pt"})
            del tensors
        for i, names in found.items():
            if names not in ({"gate_proj", "up_proj", "down_proj"}, {"gate_up_proj", "down_proj"}):
                raise ValueError(f"Incomplete MLP at layer {i}: {names}")
        if index:
            index.setdefault("metadata", {})["total_size"] = total_bytes
            (stage / index_file.name).write_text(json.dumps(index, indent=2))
        for pattern in ("config.json", "generation_config.json", "*token*.json", "tekken.json",
                        "*token*.model*", "vocab.json", "merges.txt", "*.jinja", "SYSTEM_PROMPT.txt"):
            for file in source.glob(pattern):
                if file.is_file():
                    shutil.copyfile(file, stage / file.name)
        meta = dict(plan, mode="compact", params_trimmed=sum(3 * hidden * (width - len(k)) for k in kept.values()))
        (stage / "layertrim.json").write_text(json.dumps(meta))
        stage.rename(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("--model", help="Local copy of the dense checkpoint")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["revision"] != MODELS[plan["model"]][1]:
        raise ValueError("Unexpected model revision in plan")
    export(checkpoint(plan["model"], args.model), plan, args.out)
    print(args.out)


if __name__ == "__main__":
    main()

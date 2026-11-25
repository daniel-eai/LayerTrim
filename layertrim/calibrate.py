"""Collect dense teacher-forced statistics and select the LayerTrim features."""
import argparse
import gzip
import json
import math
from pathlib import Path

from .models import MODELS, ROOT, checkpoint


def load_traces(alias):
    with gzip.open(ROOT / "calibration" / f"{alias}.json.gz", "rt") as stream:
        return json.load(stream)


def collect(model, records):
    import torch
    layers = model.model.layers
    energy = [torch.zeros(layer.mlp.down_proj.in_features, dtype=torch.float64) for layer in layers]
    cosine = [0.] * len(layers)
    handles = []

    def features(i):
        def capture(module, args):
            energy[i].add_(args[0].float().square().sum(dim=(0, 1)).double().cpu())
        return capture

    def influence(i):
        def capture(module, args, kwargs, output):
            h = args[0] if args else kwargs["hidden_states"]
            y = output[0] if isinstance(output, tuple) else output
            cosine[i] += torch.nn.functional.cosine_similarity(
                h.to(y.device).float(), y.float(), dim=-1).sum().item()
        return capture

    tokens = 0
    try:
        for i, layer in enumerate(layers):
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(features(i)))
            handles.append(layer.register_forward_hook(influence(i), with_kwargs=True))
        with torch.inference_mode():
            for record in records:
                ids = torch.tensor([record["ids"]], device=model.get_input_embeddings().weight.device)
                model.model(ids, use_cache=False)
                tokens += ids.numel()
            scores = []
            for i, layer in enumerate(layers):
                norm = layer.mlp.down_proj.weight.float().square().sum(0).double().cpu()
                scores.append((energy[i] / tokens) * norm)
    finally:
        for hook in handles:
            hook.remove()
    return [1 - c / tokens for c in cosine], scores


def select(bi, scores, budget, width, hidden, attention_parameters):
    import torch
    if not 1 <= 2 * budget <= len(bi):
        raise ValueError("Budget must select between one and all decoder layers")
    if not all(math.isfinite(v) for v in bi):
        raise ValueError("Non-finite block influence")
    keep = round(width - (attention_parameters + 3 * hidden * width) / (6 * hidden))
    if not 0 < keep < width:
        raise ValueError("Half a decoder layer must fit inside one MLP")
    target = sorted(range(len(bi)), key=bi.__getitem__)[:2 * budget]
    result = {}
    for i in sorted(target):
        if scores[i].numel() != width or not scores[i].isfinite().all():
            raise ValueError(f"Invalid feature scores at layer {i}")
        result[str(i)] = torch.topk(scores[i], keep).indices.sort().values.tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alias", choices=MODELS)
    parser.add_argument("--model", help="Local copy of the dense checkpoint")
    parser.add_argument("--budgets", type=int, nargs="+", default=[7, 9])
    parser.add_argument("--device", default="cuda:0", help="Torch device or auto")
    parser.add_argument("--out", type=Path, default=Path("plans"))
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM
    import torch
    path = checkpoint(args.alias, args.model)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                device_map=args.device, attn_implementation="sdpa").eval()
    layer = model.model.layers[0]
    width, hidden = layer.mlp.down_proj.in_features, layer.mlp.down_proj.out_features
    attn = sum(p.numel() for p in layer.self_attn.parameters())
    traces = load_traces(args.alias)
    influence, scores = collect(model, traces)
    folder = args.out / args.alias
    folder.mkdir(parents=True, exist_ok=True)
    for budget in args.budgets:
        kept = select(influence, scores, budget, width, hidden, attn)
        plan = dict(model=args.alias, revision=MODELS[args.alias][1], budget=budget,
                    intermediate_size=width, hidden_size=hidden, block_influence=influence,
                    kept_neurons=kept, n_tokens=sum(len(r["ids"]) for r in traces))
        out = folder / f"n{budget}.json"
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile("w", dir=folder, delete=False) as tmp:
            json.dump(plan, tmp, allow_nan=False)
        Path(tmp.name).replace(out)
        print(out)


if __name__ == "__main__":
    main()

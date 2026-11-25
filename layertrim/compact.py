"""Load a physically sliced checkpoint with its per-layer intermediate widths."""
import json
from pathlib import Path


def load_compact(path, device="cuda:0", dtype=None):
    import torch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
    path = Path(path)
    plan = json.loads((path / "layertrim.json").read_text())
    if plan["mode"] != "compact":
        raise ValueError("Expected a compact LayerTrim checkpoint")
    dtype = torch.bfloat16 if dtype is None else dtype
    config = AutoConfig.from_pretrained(path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
        for i, indices in plan["kept_neurons"].items():
            mlp = model.model.layers[int(i)].mlp
            size, hidden = len(indices), config.hidden_size
            for name, dim_in, dim_out in [("down_proj", size, hidden), ("gate_proj", hidden, size),
                                          ("up_proj", hidden, size), ("gate_up_proj", hidden, 2 * size)]:
                if hasattr(mlp, name):
                    old = getattr(mlp, name)
                    setattr(mlp, name, torch.nn.Linear(dim_in, dim_out, bias=old.bias is not None,
                                                       dtype=old.weight.dtype))
    model.tie_weights()
    placement = "auto" if device == "auto" else {"": device}
    model = load_checkpoint_and_dispatch(
        model, str(path), device_map=placement, dtype=dtype,
        no_split_module_classes=model._no_split_modules, strict=True,
    ).eval()
    if (path / "generation_config.json").is_file():
        model.generation_config = GenerationConfig.from_pretrained(path)
    for i, layer in enumerate(model.model.layers):
        selected = plan["kept_neurons"].get(str(i))
        width = len(selected) if selected is not None else config.intermediate_size
        if tuple(layer.mlp.down_proj.weight.shape) != (config.hidden_size, width):
            raise ValueError(f"Incorrect physical MLP width in layer {i}")
    return model

from pathlib import Path

MODELS = {
    "qwen": ("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18"),
    "phi": ("microsoft/Phi-4-reasoning-plus", "69baf8528e1bcf05f475034d9e5dd32875ed125f"),
    "magistral": ("mistralai/Magistral-Small-2506", "ad2fce5b4113139e1877dcadfd826c9262ad8e8c"),
}

ROOT = Path(__file__).resolve().parents[1]


def checkpoint(alias, local=None):
    if local:
        path = Path(local).resolve()
        if not (path / "config.json").is_file():
            raise ValueError(f"Expected a Hugging Face checkpoint at {path}")
        if any((path / name).is_file() for name in ("prune_meta.json", "layertrim.json")):
            raise ValueError(f"Expected the original dense checkpoint at {path}")
        return path
    from huggingface_hub import snapshot_download
    repo, revision = MODELS[alias]
    return Path(snapshot_download(repo, revision=revision,
                                  ignore_patterns=["consolidated*.safetensors", "*.pth"]))

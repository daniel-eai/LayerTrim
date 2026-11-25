"""Exercise real decoder classes, including the accuracy evaluator's load path."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, MistralConfig, Phi3Config, Qwen3Config

from layertrim.calibrate import collect, select
from layertrim.compact import load_compact
from layertrim.evaluate import main as evaluate
from layertrim.materialize import export


class CompactDecoderTests(unittest.TestCase):
    def check_decoder(self, alias, cls):
        torch.manual_seed(0)
        config = cls(vocab_size=48, hidden_size=16, intermediate_size=40,
                     num_hidden_layers=15, num_attention_heads=2, num_key_value_heads=1,
                     head_dim=8, max_position_embeddings=516, original_max_position_embeddings=516,
                     pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False)
        dense = AutoModelForCausalLM.from_config(config).eval()
        records = [{"ids": [1, 4, 3, 5, 7]}, {"ids": [1, 6, 2]}]
        bi, scores = collect(dense, records)
        attention = sum(p.numel() for p in dense.model.layers[0].self_attn.parameters())
        kept = select(bi, scores, 7, 40, 16, attention)
        plan = dict(model=alias, budget=7, intermediate_size=40, hidden_size=16, kept_neurons=kept)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original, compact = root / "original", root / "compact"
            dense.save_pretrained(original, max_shard_size="100KB")
            export(original, plan, compact)
            physical = load_compact(compact, device="cpu", dtype=torch.float32)
            self.assertEqual(sum(p.numel() for p in dense.parameters()) - sum(p.numel() for p in physical.parameters()),
                             sum(3 * 16 * (40 - len(k)) for k in kept.values()))
            with torch.no_grad():
                for i, indices in kept.items():
                    drop = sorted(set(range(40)) - set(indices))
                    dense.model.layers[int(i)].mlp.down_proj.weight[:, drop] = 0
                ids = torch.tensor([[1, 4, 3, 5]])
                torch.testing.assert_close(physical(ids).logits, dense(ids).logits, rtol=1e-4, atol=1e-5)
            (root / "aime25.json").write_text(json.dumps([dict(problem="Synthetic example", answer="3")]))
            argv = ["evaluate", alias, "7", "aime25", "--model", str(compact), "--data", str(root),
                    "--device", "cpu", "--out", str(root / "results")]
            # Fixed IDs isolate model execution from external tokenizer assets.
            with patch.object(sys, "argv", argv), patch("layertrim.evaluate.tokenizer_for",
                    return_value=(lambda text: [1, 4, 3], lambda ids: str(ids))):
                evaluate()
            report = json.loads((root / "results/seed0/generations_aime25.json").read_text())
            self.assertEqual(report["parameters"], sum(p.numel() for p in physical.parameters()))
            self.assertEqual(report["mlp_widths"], [len(kept[str(i)]) if str(i) in kept else 40 for i in range(15)])
            self.assertGreater(report["items"][0]["n_out_tokens"], 0)

    def test_qwen(self):
        self.check_decoder("qwen", Qwen3Config)

    def test_phi(self):
        self.check_decoder("phi", Phi3Config)

    def test_magistral(self):
        self.check_decoder("magistral", MistralConfig)

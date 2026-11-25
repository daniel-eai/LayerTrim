import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from layertrim.answers import score_aime, score_gpqa
from layertrim.calibrate import load_traces
from layertrim.materialize import export
from layertrim.models import ROOT


class CalibrationTests(unittest.TestCase):
    def test_reference_inputs(self):
        for model, total in {"qwen": 51627, "phi": 56727, "magistral": 55510}.items():
            records = load_traces(model)
            self.assertEqual(len({r["id"] for r in records}), 24)
            self.assertEqual(sum(len(r["ids"]) for r in records), total)
            self.assertTrue(all(0 < len(r["ids"]) - r["plen"] <= 2048 for r in records))

    def test_reference_counts(self):
        with (ROOT / "assets/reference_counts.csv").open() as f:
            rows = list(csv.DictReader(f))
        keys = {(r["model"], r["budget"], r["task"], r["seed"]) for r in rows}
        self.assertEqual(len(rows), 90)
        self.assertEqual(len(keys), 90)
        self.assertTrue(all(0 <= int(r["correct"]) <= int(r["total"]) for r in rows))

    def test_boxed_answer(self):
        self.assertEqual(score_aime([r"\boxed{42} then \boxed{003}"], ["3"])[0], 1)
        self.assertEqual(score_gpqa([r"\boxed{C}"], ["C"])[0], 1)


class ExportTests(unittest.TestCase):
    def check_export(self, fused):
        rng = np.random.default_rng(6)
        hidden, width = 3, 8
        tensors = {}
        for layer in range(3):
            prefix = f"model.layers.{layer}.mlp."
            for name, shape in ({"gate_up_proj": (2 * width, hidden)} if fused else
                                {"gate_proj": (width, hidden), "up_proj": (width, hidden)}).items():
                tensors[prefix + name + ".weight"] = rng.normal(size=shape).astype("float32")
            tensors[prefix + "down_proj.weight"] = rng.normal(size=(hidden, width)).astype("float32")
        tensors["model.layers.0.self_attn.q_proj.weight"] = np.eye(hidden, dtype="float32")
        indices = [1, 4, 7]
        plan = dict(intermediate_size=width, hidden_size=hidden, budget=1,
                    kept_neurons={"0": indices, "2": indices})
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
            mapping = {k: shards[i % 2] for i, k in enumerate(tensors)}
            for shard in shards:
                save_file({k: v for k, v in tensors.items() if mapping[k] == shard}, str(base / shard))
            (base / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map=mapping)))
            (base / "config.json").write_text(json.dumps(dict(intermediate_size=width,
                                                              hidden_size=hidden, num_hidden_layers=3)))
            dest = base / "compact"
            export(base, plan, dest)
            compact = {k: v for shard in shards for k, v in load_file(str(dest / shard)).items()}
            mask = {k: v.copy() for k, v in tensors.items()}
            for k in tensors:
                if ".mlp." not in k or ".layers.1." in k:
                    np.testing.assert_array_equal(compact[k], tensors[k])
                elif "down_proj" in k:
                    mask[k][:, np.setdiff1d(np.arange(width), indices)] = 0
            meta = json.loads((dest / "model.safetensors.index.json").read_text())
            self.assertEqual(meta["metadata"]["total_size"], sum(v.nbytes for v in compact.values()))
            self.assertLess(meta["metadata"]["total_size"], sum(v.nbytes for v in tensors.values()))
            x = rng.normal(size=(4, hidden)).astype("float32")
            prefix = "model.layers.2.mlp."
            def forward(t):
                if fused:
                    gate, up = np.split(x @ t[prefix+"gate_up_proj.weight"].T, 2, axis=-1)
                else:
                    gate = x @ t[prefix+"gate_proj.weight"].T
                    up = x @ t[prefix+"up_proj.weight"].T
                return ((gate / (1 + np.exp(-gate))) * up) @ t[prefix+"down_proj.weight"].T
            np.testing.assert_array_equal(compact[prefix+"down_proj.weight"],
                                          tensors[prefix+"down_proj.weight"][:, indices])
            np.testing.assert_allclose(forward(mask), forward(compact), rtol=1e-5, atol=1e-6)

    def test_separate(self):
        self.check_export(False)

    def test_fused(self):
        self.check_export(True)


if __name__ == "__main__":
    unittest.main()

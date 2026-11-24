"""Tests for the standalone BEAC full-model path and log-only evaluation."""

from __future__ import annotations

import importlib.machinery
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "runtime_site"))

# 该检查不运行 NLP 抽取或训练，只需令训练模块的可选运行时符号可导入。
if "nltk" not in sys.modules:
    nltk_stub = types.ModuleType("nltk")
    nltk_stub.__spec__ = importlib.machinery.ModuleSpec("nltk", loader=None)
    nltk_stub.pos_tag = lambda tokens: []
    nltk_stub.word_tokenize = lambda text: text.split()
    nltk_stub.data = SimpleNamespace(path=[])
    sys.modules["nltk"] = nltk_stub

if "transformers.optimization" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.__path__ = []
    transformers_stub.__spec__ = importlib.machinery.ModuleSpec(
        "transformers", loader=None, is_package=True
    )
    optimization_stub = types.ModuleType("transformers.optimization")
    optimization_stub.AdamW = torch.optim.AdamW
    sys.modules["transformers"] = transformers_stub
    sys.modules["transformers.optimization"] = optimization_stub

from beac_full_model import pipeline, support_alignment, train_impl
from beac_full_model._repro_compat import advance_legacy_disabled_init_rng


class BEACFullModelTests(unittest.TestCase):
    def test_removed_modules_have_no_cli_switches(self) -> None:
        parser = train_impl.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertFalse(
            destinations
            & {
                "disable_support_gate",
                "disable_lite_rqg",
                "use_hard_negative_routing",
                "hard_negative_loss_weight",
                "hard_negative_temperature",
            }
        )
        self.assertIn("log_only", destinations)

    def test_initialization_rng_matches_removed_historical_layers(self) -> None:
        feature_dim = 512
        torch.manual_seed(2024)
        advance_legacy_disabled_init_rng(feature_dim)
        actual_state = torch.random.get_rng_state()

        torch.manual_seed(2024)
        nn.Linear(feature_dim, feature_dim)
        nn.Linear(feature_dim, feature_dim)
        nn.Linear(feature_dim * 2, feature_dim)
        nn.Linear(feature_dim, feature_dim)
        expected_state = torch.random.get_rng_state()

        self.assertTrue(torch.equal(actual_state, expected_state))

    def test_no_gate_ibet_transport_matches_historical_neutral_prior(self) -> None:
        torch.manual_seed(12)
        similarity = torch.rand(4, 49)
        temperature = 0.05
        steps = 32
        evidence_mass = 0.75
        background_logit = -0.20

        scores = similarity.detach().float()
        gate = scores.new_full((scores.shape[1],), 0.5)
        background_scores = background_logit + torch.log1p(-gate)
        logits = torch.cat([scores, background_scores[None, :]], dim=0)
        kernel = torch.exp((logits - logits.max()) / temperature).clamp_min(1e-8)
        row_mass = scores.new_full((scores.shape[0] + 1,), evidence_mass / scores.shape[0])
        row_mass[-1] = 1.0 - evidence_mass
        column_mass = scores.new_full((scores.shape[1],), 1.0 / scores.shape[1])
        left = scores.new_ones((scores.shape[0] + 1,))
        right = scores.new_ones((scores.shape[1],))
        for _ in range(steps):
            left = row_mass / (kernel @ right).clamp_min(1e-8)
            right = column_mass / (kernel.transpose(0, 1) @ left).clamp_min(1e-8)
        historical_plan = left[:, None] * kernel * right[None, :]

        clean_plan = support_alignment.information_balanced_evidence_transport(
            similarity,
            temperature=temperature,
            steps=steps,
            evidence_mass=evidence_mass,
            background_logit=background_logit,
        )
        self.assertTrue(torch.allclose(clean_plan, historical_plan, atol=1e-7, rtol=1e-6))

    def test_log_only_evaluation_does_not_write_score_artifacts(self) -> None:
        runner = object.__new__(train_impl.BEACRunner)
        runner.evaluation = lambda _loader: (
            torch.ones(1, 1).numpy(),
            torch.ones(1, 1).numpy(),
            torch.ones(1, 1).numpy(),
            torch.ones(1, 1).numpy(),
        )
        runner.itm_eval = lambda *_args: {"r_mean": 60.0, "i2t_r1": 1.0, "t2i_r1": 1.0}
        runner.model = SimpleNamespace(
            candidate_context_head=SimpleNamespace(parameters_record=lambda: {})
        )
        old_config = pipeline.config

        with tempfile.TemporaryDirectory() as temp_dir:
            score_dir = Path(temp_dir) / "scores"
            pipeline.config = SimpleNamespace(
                log_only=True,
                evaluation_score_dir=str(score_dir),
                method_name="BEAC-Full-Model",
                ranking_mode="legacy_cch",
                global_score_weight=0.85,
                use_candidate_context=True,
            )
            old_print = pipeline.Tools.print
            pipeline.Tools.print = lambda *_args, **_kwargs: None
            try:
                dataset = SimpleNamespace(txt2img={}, img2txt={})
                result = runner._evaluate_and_save(None, dataset, "init", "test")
            finally:
                pipeline.Tools.print = old_print
                pipeline.config = old_config

            self.assertEqual(result["r_mean"], 60.0)
            self.assertFalse(score_dir.exists())
            self.assertEqual(list(Path(temp_dir).iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

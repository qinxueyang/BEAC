"""CPU checks for the shared local scorer and IBET evidence mass."""
from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


CODE_ROOT = Path(__file__).resolve().parents[1]


def load_local_scorer():
    """Execute the source functions without loading a backbone or using a GPU."""
    path = CODE_ROOT / "beac_full_model/support_alignment.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    settings = {}
    namespace = {
        "torch": torch,
        "F": F,
        "Sequence": Sequence,
        "cfg": lambda name, default: settings.get(name, default),
    }
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("DEFAULT_") or name == "EPSILON":
                namespace[name] = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name in (
            "pad_nouns", "batched_information_balanced_evidence_transport"
        ):
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "SupportAlignmentModel":
            nodes.extend(child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "local_scores")
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["local_scores"], settings


def verify_local_mean():
    scorer, settings = load_local_scorer()
    settings["local_scoring_mode"] = "uniform_mean"
    nouns = [torch.tensor([[0.2, math.sqrt(0.96)]], dtype=torch.float32)]
    globals_ = torch.zeros(1, 2)
    for count in (1, 4, 49):
        patches = torch.tensor([[[1.0, 0.0]] * count], dtype=torch.float32)
        actual = scorer(None, patches, nouns, globals_)
        torch.testing.assert_close(actual, torch.full((1, 1), 0.2), atol=1e-6, rtol=1e-6)
        print(f"PASS constant cosine=0.2 patches={count} local_score={actual.item():.6f}")

    generator = torch.Generator().manual_seed(2024)
    for count in (1, 4, 49):
        for noun_counts in ((1,), (1, 3), (2, 4, 1)):
            patches = torch.randn(2, count, 8, generator=generator)
            nouns = [torch.randn(n, 8, generator=generator) for n in noun_counts]
            actual = scorer(None, patches, nouns, torch.zeros(len(nouns), 8))
            expected = torch.empty_like(actual)
            for image_index in range(patches.shape[0]):
                for text_index, noun in enumerate(nouns):
                    pair_scores = F.normalize(patches[image_index], dim=-1) @ F.normalize(noun, dim=-1).t()
                    expected[image_index, text_index] = pair_scores.mean()
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    print("PASS mixed valid noun counts and padding; all patch/noun pairs averaged")

    settings.update({
        "local_scoring_mode": "ibet_ot",
        "use_information_balance": True,
        "information_balance_temperature": 0.10,
        "information_balance_steps": 24,
        "information_balance_evidence_mass": 0.35,
        "information_balance_score_mode": "background_aware",
        "local_inference_aggregation": "transport_mean",
    })
    patches = torch.tensor([[[1.0, 0.0]] * 49], dtype=torch.float32)
    nouns = [torch.tensor([[0.2, math.sqrt(0.96)]], dtype=torch.float32)]
    actual = scorer(None, patches, nouns, torch.zeros(1, 2))
    torch.testing.assert_close(actual, torch.full((1, 1), 0.07), atol=1e-6, rtol=1e-6)
    print(f"PASS IBET real evidence mass=0.35 remains local_score={actual.item():.6f}")


if __name__ == "__main__":
    verify_local_mean()
    print("ALL_CHECKS_PASSED shared uniform normalization and IBET scoring")

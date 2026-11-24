"""CPU-only checks for all datasets and the common BEAC launch process."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("beac_launcher", ROOT / "scripts" / "train.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def arguments(dataset: str, assets: Path, condition: str = "full", run_id: str = "verify"):
    return SimpleNamespace(
        dataset=dataset, condition=condition, gpu=0, run_id=run_id,
        python=sys.executable, image_root=None,
        dataset_root=str(assets / "annotations"), data_root=str(assets / "images"),
        model_path=str(assets / "model.bin"), nltk_data=str(assets / "nltk"),
    )


def training_parser():
    """Load the actual CLI definition without importing the CUDA training stack."""
    path = ROOT / "beac_full_model" / "train_impl.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "build_parser")
    namespace = {
        "argparse": argparse,
        "dataset_config": SimpleNamespace(DATASETS=dict.fromkeys(launcher.DATASETS)),
        "OUTPUT_ROOT": ROOT / "outputs", "SCORE_ROOT": ROOT / "scores",
        "ABLATION_DESCRIPTIONS": {"c3": "CCH3"},
        "GLOBAL_LOSS_WEIGHT": 1.0, "IBET_LOSS_WEIGHT": 1.0,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["build_parser"]()


def resolve_command(command: list[str]):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--project-root", required=True)
    bootstrap.add_argument("--gpu", required=True)
    bootstrap.add_argument("--no-save-checkpoint", action="store_true")
    _, remaining = bootstrap.parse_known_args(command[3:])
    return training_parser().parse_args(remaining)


class LauncherTests(unittest.TestCase):
    def test_reference_configurations_and_all_conditions(self):
        expected = {
            "ucm": (60.29, 1.0, .75, .05, .85, .25, .75, 32, .75),
            "rsitmd": (51.24, .25, .75, .20, .95, .60, .75, 24, .35),
            "rsicd": (38.30, 1.0, .50, .20, .85, .60, .50, 32, .25),
        }
        with tempfile.TemporaryDirectory() as temp:
            assets = Path(temp)
            for dataset, values in expected.items():
                full = launcher.build_plan(arguments(dataset, assets))
                parsed = resolve_command(full["command"])
                actual = (
                    full["reference"]["mR"], parsed.global_loss_weight,
                    parsed.triplet_loss_weight, parsed.candidate_loss_weight,
                    parsed.global_score_weight, parsed.noun_patch_weight,
                    parsed.candidate_initial_direction_weight,
                    parsed.information_balance_steps, parsed.information_balance_evidence_mass,
                )
                self.assertEqual(actual, values)
                self.assertEqual(parsed.epochs, 10)
                self.assertEqual(parsed.scheduler_epochs, 10)
                self.assertEqual(parsed.seed, 2024)
                self.assertEqual(parsed.max_train_batch_size, 256)
                self.assertEqual(parsed.lr, .00075)
                self.assertEqual(parsed.ablation, "c3")
                self.assertEqual(parsed.candidate_initial_mutual_weight, 0)
                self.assertIn("--no-save-checkpoint", full["command"])
                self.assertFalse(full["reference_result_reused"])
                for condition, switches in launcher.CONDITIONS.items():
                    with self.subTest(dataset=dataset, condition=condition):
                        plan = launcher.build_plan(arguments(dataset, assets, condition))
                        parsed = resolve_command(plan["command"])
                        ibet, cch_train, cch_infer = switches
                        self.assertEqual(parsed.use_information_balance, ibet)
                        self.assertEqual(parsed.local_scoring_mode, "ibet_ot" if ibet else "uniform_mean")
                        self.assertEqual(parsed.candidate_loss_weight, values[3] if cch_train else 0)
                        self.assertEqual(parsed.cch_training_enabled is not False, cch_train)
                        self.assertEqual(parsed.cch_inference_enabled is not False, cch_infer)
                        self.assertEqual(plan["reference_applies_to_condition"], condition == "full")
                        self.assertFalse(Path(plan["experiment_root"]).exists())

    def test_cli_dry_run_for_three_datasets(self):
        with tempfile.TemporaryDirectory() as temp:
            for dataset in launcher.DATASETS:
                process = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "scripts" / "train.py"),
                     "--dataset", dataset, "--run-id", "dryrun_test", "--dry-run",
                     "--dataset-root", temp, "--data-root", temp,
                     "--model-path", str(Path(temp) / "model.bin"), "--nltk-data", temp],
                    text=True, encoding="utf-8", capture_output=True, timeout=30, check=False,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                plan = json.loads(process.stdout)
                self.assertEqual(plan["dataset"], dataset)
                self.assertEqual(Path(plan["code_root"]), ROOT)
                self.assertFalse(Path(plan["experiment_root"]).exists())

    def test_invalid_run_and_gpu(self):
        args = arguments("rsitmd", ROOT)
        args.run_id = "../escape"
        with self.assertRaises(ValueError):
            launcher.build_plan(args)
        args.run_id = "valid"
        args.gpu = -1
        with self.assertRaises(ValueError):
            launcher.build_plan(args)

    def test_foreground_process_and_asset_checks(self):
        with tempfile.TemporaryDirectory(prefix="beac_launch_test_") as temp:
            fixture = Path(temp) / "BEAC"
            for component in ["scripts", "beac_full_model", "configs"]:
                (fixture / component).mkdir(parents=True)
            for dataset in launcher.DATASETS:
                shutil.copy2(ROOT / "configs" / (dataset + ".json"), fixture / "configs")
            (fixture / "beac_full_model" / "train_impl.py").write_text("# isolated fixture\n", encoding="utf-8")
            (fixture / "scripts" / "capture_test_selection.py").write_text(
                "import os, sys\n"
                "assert os.environ['BEAC_FULL_MODEL_MODEL_NAME'] == 'ViT-B/32'\n"
                "print('SMOKE_FORWARD_OK', flush=True)\n"
                "sys.exit(int(os.environ.get('BEAC_TEST_EXIT_CODE', '0')))\n",
                encoding="utf-8",
            )
            assets = Path(temp) / "assets"
            for directory in [assets / "annotations" / "data" / "finetune", assets / "images", assets / "nltk"]:
                directory.mkdir(parents=True)
            (assets / "model.bin").write_bytes(b"fixture, not model weights")
            with patch.object(launcher, "ROOT", fixture), patch.dict(
                os.environ, {"BEAC_FULL_MODEL_IMAGE_ROOT_OVERRIDE": ""}
            ):
                for dataset in launcher.DATASETS:
                    config = json.loads((fixture / "configs" / (dataset + ".json")).read_text(encoding="utf-8"))
                    image_root = assets / "images" / config["data"]["image_subdirectory"]
                    image_root.mkdir(parents=True)
                    for key in ["train_file", "val_file", "test_file"]:
                        path = assets / "annotations" / config["data"][key]
                        path.write_text("[]", encoding="utf-8")
                    plan = launcher.build_plan(arguments(dataset, assets))
                    launcher.validate_assets(plan)
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(launcher.execute(plan), 0)
                    run = Path(plan["experiment_root"])
                    self.assertIn("SMOKE_FORWARD_OK", (run / "training.log").read_text(encoding="utf-8"))
                    self.assertTrue((run / "run_config.json").is_file())
                    self.assertTrue((run / "effective_command.txt").is_file())
                    with self.assertRaises(FileExistsError):
                        launcher.execute(plan)
                    annotation = Path(plan["annotation_files"][0])
                    annotation.unlink()
                    with self.assertRaises(ValueError):
                        launcher.validate_assets(plan)
                    annotation.write_text("[]", encoding="utf-8")
                plan = launcher.build_plan(arguments("rsitmd", assets, run_id="failed_child"))
                with patch.dict(os.environ, {"BEAC_TEST_EXIT_CODE": "7"}):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(launcher.execute(plan), 7)
                args = arguments("rsitmd", assets, run_id="override")
                args.image_root = str(assets / "custom_images")
                Path(args.image_root).mkdir()
                plan = launcher.build_plan(args)
                launcher.validate_assets(plan)
                self.assertEqual(plan["environment"]["BEAC_FULL_MODEL_IMAGE_ROOT_OVERRIDE"], args.image_root)

    def test_all_python_sources_compile(self):
        for path in ROOT.rglob("*.py"):
            if "__pycache__" not in path.parts:
                compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")


if __name__ == "__main__":
    unittest.main(verbosity=2)

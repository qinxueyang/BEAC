#!/usr/bin/env python3
"""Launch a recorded BEAC configuration for UCM, RSITMD or RSICD."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ucm", "rsitmd", "rsicd")
ASSET_ENVIRONMENT = {
    "dataset_root": "BEAC_FULL_MODEL_DATASET_ROOT",
    "data_root": "BEAC_FULL_MODEL_DATA_ROOT",
    "model_path": "BEAC_FULL_MODEL_MODEL_PATH",
    "nltk_data": "BEAC_FULL_MODEL_NLTK_DATA",
}
CONDITIONS = {
    "full": (True, True, True),
    "no_ibet_no_cch": (False, False, False),
    "ibet_only": (True, False, False),
    "cch_only": (False, True, True),
    "ibet_cch_training_only": (True, True, False),
}
CCH_SWITCHES = {"cch_training_enabled", "cch_inference_enabled"}


def condition_settings(training: dict, condition: str) -> dict:
    """Apply the original five component conditions to a dataset configuration."""
    result = dict(training)
    if condition == "full":
        return result
    ibet, cch_train, cch_inference = CONDITIONS[condition]
    result.update({
        "use_information_balance": ibet,
        "local_scoring_mode": "ibet_ot" if ibet else "uniform_mean",
        "cch_training_enabled": cch_train,
        "cch_inference_enabled": cch_inference,
        "candidate_loss_weight": training["candidate_loss_weight"] if cch_train else 0.0,
    })
    return result


def training_arguments(settings: dict) -> list[str]:
    """Convert explicit configuration values to the existing training CLI."""
    arguments = []
    for name, value in settings.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("invalid training option: " + name)
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                arguments.append(flag)
            elif name in CCH_SWITCHES:
                arguments.append("--no-" + name.replace("_", "-"))
        elif isinstance(value, (str, int, float)):
            arguments.extend([flag, str(value)])
        else:
            raise ValueError("invalid value for training option: " + name)
    return arguments


def build_plan(args: argparse.Namespace) -> dict:
    """Build one run while keeping dataset-specific scientific settings explicit."""
    config_path = ROOT / "configs" / (args.dataset + ".json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["dataset"] != args.dataset:
        raise ValueError("dataset does not match configuration: " + str(config_path))
    settings = condition_settings(config["training"], args.condition)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("repro_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in (".", ".."):
        raise ValueError("run-id must contain only letters, digits, dots, underscores or hyphens")
    if args.gpu < 0:
        raise ValueError("gpu must be a non-negative integer")
    tag = f"BEAC_{args.dataset.upper()}_seed{settings['seed']}_{args.condition}_{run_id}"
    experiment_root = ROOT / "runs" / tag
    command = [
        args.python, "-u", "scripts/capture_test_selection.py",
        "--project-root", str(ROOT), "--no-save-checkpoint",
        "--gpu", str(args.gpu), "--dataset", args.dataset,
        "--output-root", str(experiment_root / "outputs"),
        "--score-root", str(experiment_root / "eval_scores"), "--tag", tag,
    ]
    command.extend(training_arguments(settings))
    environment = {
        "BEAC_FULL_MODEL_PROJECT_ROOT": str(ROOT),
        "BEAC_FULL_MODEL_MODEL_NAME": "ViT-B/32",
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT / "runtime_site"), str(ROOT)]
            + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
        ),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
    }
    for attribute, variable in ASSET_ENVIRONMENT.items():
        value = getattr(args, attribute) or os.environ.get(variable)
        if not value:
            raise ValueError("set --" + attribute.replace("_", "-") + " or " + variable)
        environment[variable] = str(Path(value).expanduser().resolve())
    image_override = args.image_root or os.environ.get("BEAC_FULL_MODEL_IMAGE_ROOT_OVERRIDE")
    if image_override:
        image_root = Path(image_override).expanduser().resolve()
        environment["BEAC_FULL_MODEL_IMAGE_ROOT_OVERRIDE"] = str(image_root)
    else:
        image_root = Path(environment["BEAC_FULL_MODEL_DATA_ROOT"]) / config["data"]["image_subdirectory"]
    environment["NLTK_DATA"] = environment["BEAC_FULL_MODEL_NLTK_DATA"]
    annotation_root = Path(environment["BEAC_FULL_MODEL_DATASET_ROOT"])
    return {
        "dataset": args.dataset,
        "condition": args.condition,
        "reference": config["reference"],
        "reference_applies_to_condition": args.condition == "full",
        "reference_result_reused": False,
        "config_path": str(config_path),
        "code_root": str(ROOT),
        "experiment_root": str(experiment_root),
        "image_root": str(image_root),
        "annotation_files": [str(annotation_root / config["data"][name])
                             for name in ("train_file", "val_file", "test_file")],
        "environment": environment,
        "command": command,
    }


def validate_assets(plan: dict) -> None:
    """Check input files and directories before creating a run directory."""
    if not shutil.which(plan["command"][0]):
        raise ValueError("Python executable not found: " + plan["command"][0])
    code_root = Path(plan["code_root"])
    required = [
        code_root / "scripts" / "capture_test_selection.py",
        code_root / "beac_full_model" / "train_impl.py",
        Path(plan["environment"]["BEAC_FULL_MODEL_MODEL_PATH"]),
    ]
    required.extend(Path(item) for item in plan["annotation_files"])
    for path in required:
        if not path.is_file():
            raise ValueError("required file missing: " + str(path))
    directories = [Path(plan["image_root"])]
    directories.extend(Path(plan["environment"][variable])
                       for variable in ("BEAC_FULL_MODEL_DATASET_ROOT", "BEAC_FULL_MODEL_DATA_ROOT",
                                        "BEAC_FULL_MODEL_NLTK_DATA"))
    for path in directories:
        if not path.is_dir():
            raise ValueError("required directory missing: " + str(path))


def execute(plan: dict) -> int:
    """Run in the foreground and preserve the command, configuration and log."""
    validate_assets(plan)
    root = Path(plan["experiment_root"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "run_config.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "effective_command.txt").write_text(
        shlex.join(plan["command"]) + "\n", encoding="utf-8"
    )
    environment = os.environ.copy()
    environment.update(plan["environment"])
    with (root / "training.log").open("w", encoding="utf-8") as log:
        print("BEAC_START " + str(root), flush=True)
        with subprocess.Popen(
            plan["command"], cwd=plan["code_root"], env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        ) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                return process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), default="full")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--image-root", default=None)
    for attribute in ASSET_ENVIRONMENT:
        parser.add_argument("--" + attribute.replace("_", "-"), default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print one command without importing Torch, creating runs or training")
    args = parser.parse_args()
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        return execute(plan)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("BEAC_ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

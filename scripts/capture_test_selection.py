"""Run the historical test-selection protocol with optional checkpoint capture.

This is a thin runner around ``beac_full_model.train_impl``. ``--no-save-checkpoint``
keeps only the test-mR/epoch trace in logs while retaining the same initial-plus-
epoch-wise test selection protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--project-root", type=Path, required=True)
    bootstrap.add_argument("--checkpoint-dir", type=Path, default=None)
    bootstrap.add_argument("--no-save-checkpoint", action="store_true")
    bootstrap.add_argument("--gpu", required=True)
    bootstrap.add_argument("--embedding-output-dir", type=Path, default=None)
    bootstrap.add_argument(
        "--embedding-stages",
        default="epoch_000,epoch_004,epoch_009",
        help="Comma-separated internal stages to export; epoch_000/004/009 are human epochs 1/5/10.",
    )
    boot, remaining_args = bootstrap.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(boot.gpu)
    os.environ["BEAC_FULL_MODEL_PROJECT_ROOT"] = str(boot.project_root.resolve())
    os.chdir(boot.project_root)
    sys.path.insert(0, str(boot.project_root))

    from beac_full_model import train_impl
    import torch

    parser = train_impl.build_parser()
    # The bootstrap-only arguments must not be passed to train_impl's parser.
    args = parser.parse_args(remaining_args)
    if args.strict_validation_protocol:
        raise SystemExit("This capture script requires historical test selection; do not pass --strict-validation-protocol")
    if not boot.no_save_checkpoint and boot.checkpoint_dir is None:
        raise SystemExit("--checkpoint-dir is required unless --no-save-checkpoint is set")
    train_impl.configure(args)
    runner = train_impl.BEACRunner()
    checkpoint_dir = None if boot.checkpoint_dir is None else boot.checkpoint_dir.resolve()
    if checkpoint_dir is not None and not boot.no_save_checkpoint:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    embedding_output_dir = (
        boot.embedding_output_dir.resolve()
        if boot.embedding_output_dir is not None
        else None
    )
    embedding_stages = {
        item.strip()
        for item in str(boot.embedding_stages).split(",")
        if item.strip()
    }

    @torch.no_grad()
    def export_embeddings(stage: str) -> None:
        if embedding_output_dir is None or stage not in embedding_stages:
            return
        embedding_output_dir.mkdir(parents=True, exist_ok=True)
        runner.model.eval()
        device = next(runner.model.parameters()).device

        image_features = []
        for images, _ in runner.create_dataset.test_loader:
            _, global_features = runner.model.get_vis_emb(images.to(device))
            image_features.append(global_features.detach().cpu().float().numpy())
        image_features = np.concatenate(image_features, axis=0)

        texts = runner.create_dataset.test_dataset.text
        text_features = []
        text_batch_size = int(getattr(train_impl.pipeline.config, "batch_size_test_text", 128))
        for start in range(0, len(texts), text_batch_size):
            text_batch = texts[start : start + text_batch_size]
            text_input = runner.tokenize(text_batch).to(device)
            _, global_features = runner.model.get_txt_emb(
                text_input, return_word_feats=False
            )
            text_features.append(global_features.detach().cpu().float().numpy())
        text_features = np.concatenate(text_features, axis=0)

        output_path = embedding_output_dir / f"{stage}_embeddings.npz"
        np.savez_compressed(
            output_path,
            image_embeddings=image_features,
            text_embeddings=text_features,
            image_ids=np.arange(image_features.shape[0], dtype=np.int64),
            text_ids=np.arange(text_features.shape[0], dtype=np.int64),
            stage=np.asarray(stage),
            normalization=np.asarray("l2_normalized"),
        )
        print(
            "EMBEDDINGS_SAVED",
            json.dumps(
                {
                    "stage": stage,
                    "path": str(output_path),
                    "image_shape": list(image_features.shape),
                    "text_shape": list(text_features.shape),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    checkpoint_path = (
        None if boot.no_save_checkpoint else checkpoint_dir / "historical_test_best.pth"
    )
    metadata_path = (
        None if boot.no_save_checkpoint else checkpoint_dir / "historical_test_best.json"
    )
    best_score = float("-inf")
    best_stage = ""
    best_result = None

    original_test = runner.test

    def test_and_capture():
        nonlocal best_score, best_stage, best_result
        result = original_test()
        export_embeddings(str(getattr(runner, "last_test_stage", "unknown")))
        score = float(result["r_mean"])
        if score > best_score:
            best_score = score
            best_stage = str(getattr(runner, "last_test_stage", "unknown"))
            best_result = result
            if not boot.no_save_checkpoint:
                payload = {
                    "model": {name: value.detach().cpu() for name, value in runner.model.state_dict().items()},
                    "epoch_stage": best_stage,
                    "test_result": result,
                    "protocol": "historical_test_split_per_epoch_selection",
                }
                temporary = checkpoint_path.with_suffix(".tmp")
                torch.save(payload, temporary)
                os.replace(temporary, checkpoint_path)
                metadata_path.write_text(json.dumps({
                    "checkpoint": str(checkpoint_path),
                    "epoch_stage": best_stage,
                    "best_test_r_mean": best_score,
                    "test_result": best_result,
                    "protocol": "historical_test_split_per_epoch_selection",
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            print("HISTORICAL_TEST_BEST_UPDATED", json.dumps({
                "checkpoint_saved": not boot.no_save_checkpoint,
                "stage": best_stage,
                "best_test_r_mean": best_score,
                "test_result": best_result,
            }, ensure_ascii=False), flush=True)
        return result

    runner.test = test_and_capture
    result = runner.train()
    print("HISTORICAL_TEST_CAPTURE_FINAL", json.dumps({
        "final_result": result,
        "checkpoint_saved": not boot.no_save_checkpoint,
        "best_checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "best_stage": best_stage,
        "best_test_r_mean": best_score,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

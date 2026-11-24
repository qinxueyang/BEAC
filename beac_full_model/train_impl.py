"""BEAC Full Model training implementation.

The retained path is OpenCLIP + Adapter, global alignment, Robust Triplet,
IBET noun-patch evidence alignment, and Candidate Context Head (CCH).

本文件是主入口唯一执行的实现，不依赖历史兼容实现或历史排序器。
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    from transformers.optimization import AdamW
except ImportError:
    from torch.optim import AdamW

from beac_full_model import candidate_context, dataset_config, pipeline, support_alignment


PROJECT_ROOT = Path(
    os.environ.get("BEAC_FULL_MODEL_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
OUTPUT_ROOT = PROJECT_ROOT / "runs/outputs"
SCORE_ROOT = PROJECT_ROOT / "runs/eval_scores"
METHOD_NAME = "BEAC-Full-Model"

CANDIDATE_INITIAL_CONTEXT_WEIGHT = 1.0
CANDIDATE_INITIAL_MUTUAL_WEIGHT = 0.0
GLOBAL_LOSS_WEIGHT = 1.0
IBET_LOSS_WEIGHT = 1.0
CANDIDATE_LOSS_WEIGHT = 0.20
CANDIDATE_STUDENT_TEMPERATURE = 0.07
CANDIDATE_MUTUAL_TEMPERATURE = 0.07
CANDIDATE_MUTUAL_SCALE = 0.10
ABLATION_DESCRIPTIONS = {
    "m0": "global_only",
    "m1": "global_local",
    "m3": "global_local_cch",
    "c0": "cch_without_rank_context",
    "c1": "cch_without_direction_context",
    "c2": "cch_without_context_mixture",
    "c3": "cch_without_mutual_agreement",
    "c4": "cch_full",
}


def runtime_float(name: str, default: float) -> float:
    return float(getattr(pipeline.config, name, default))


class BEACModel(support_alignment.SupportAlignmentModel):
    """只包含论文主方法的模型：support alignment 与 CCH。"""

    def __init__(self) -> None:
        super().__init__()
        self.candidate_context_head = candidate_context.CandidateContextHead()

    def candidate_training_scores(
        self,
        patches: torch.Tensor,
        image_global: torch.Tensor,
        text_global: torch.Tensor,
        nouns: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        global_weight, local_weight = support_alignment.pairwise_score_weights(self)
        global_scores = image_global @ text_global.t()
        if local_weight <= 0.0:
            return global_scores
        local_scores = self.local_scores(patches, nouns, text_global)
        return global_weight * global_scores + local_weight * local_scores.detach()

    def candidate_scores(self, base_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.candidate_context_head(
            base_scores,
            mutual_temperature=runtime_float(
                "candidate_mutual_temperature", CANDIDATE_MUTUAL_TEMPERATURE
            ),
            mutual_scale=runtime_float("candidate_mutual_scale", CANDIDATE_MUTUAL_SCALE),
        )

    def forward(self, image, text_ids, raw_texts, idx=None, label=None):
        patches, image_global, text_global, nouns, fine_loss = self.training_features(
            image, text_ids, raw_texts, idx=idx
        )
        contrastive = runtime_float("global_loss_weight", GLOBAL_LOSS_WEIGHT) * self.get_contrastive_loss(
            image_global, text_global, idx
        )
        triplet = self.get_triplet_loss(image_global, text_global, idx)
        if idx is None or not bool(getattr(
            pipeline.config,
            "use_candidate_context_training",
            getattr(pipeline.config, "use_candidate_context", True),
        )):
            return contrastive, triplet, fine_loss

        base_scores = self.candidate_training_scores(
            patches, image_global, text_global, nouns
        )
        scores_i2t, scores_t2i = self.candidate_scores(base_scores)
        candidate_loss = candidate_context.symmetric_soft_cross_entropy(
            scores_i2t,
            scores_t2i,
            idx,
            runtime_float("candidate_student_temperature", CANDIDATE_STUDENT_TEMPERATURE),
        )
        contrastive = contrastive + runtime_float(
            "candidate_loss_weight", CANDIDATE_LOSS_WEIGHT
        ) * candidate_loss
        return contrastive, triplet, fine_loss


def compact_img2txt(img2txt: dict[int, list[int]]) -> np.ndarray:
    return np.asarray([img2txt[key] for key in sorted(img2txt)], dtype=np.int64)


def compact_txt2img(txt2img: dict[int, int]) -> np.ndarray:
    return np.asarray([txt2img[key] for key in sorted(txt2img)], dtype=np.int64)


class BEACRunner(support_alignment.SupportAlignmentRunner):
    """主方法训练器；评估只保留 raw 与 CCH 两条路径。"""

    def __init__(self) -> None:
        self.eval_index = 0
        self.device = torch.device(pipeline.config.device)
        self.model = BEACModel().to(self.device)
        self.model.candidate_context_head.configure_components(
            **pipeline.config.cch_components
        )
        self.model.candidate_context_head.reset(
            float(pipeline.config.candidate_initial_rank_weight),
            float(pipeline.config.candidate_initial_direction_weight),
            float(pipeline.config.candidate_initial_context_weight),
            float(pipeline.config.candidate_initial_mutual_weight),
        )
        self.tokenize = self.model.tokenize
        self.set_trainable(self.model)
        self.create_dataset = pipeline.CreateDataset()
        self.optimizer = AdamW(
            filter(lambda parameter: parameter.requires_grad, self.model.parameters()),
            lr=pipeline.config.lr,
            weight_decay=pipeline.config.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.98),
        )
        schedule_epochs = int(
            getattr(pipeline.config, "scheduler_epochs", pipeline.config.epochs)
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, schedule_epochs * len(self.create_dataset.train_loader)
        )
        pipeline.Tools.print(
            f"learnable parameter num = {self.count_trainable_parameters()}",
            txt_path=pipeline.config.log_filename,
        )

    @staticmethod
    def set_trainable(model: BEACModel) -> None:
        support_alignment.SupportAlignmentRunner.set_trainable(model)
        if bool(getattr(
            pipeline.config,
            "use_candidate_context_training",
            getattr(pipeline.config, "use_candidate_context", True),
        )):
            model.candidate_context_head.train()
            for parameter in model.candidate_context_head.parameters():
                parameter.requires_grad = True
        else:
            model.candidate_context_head.eval()
            for parameter in model.candidate_context_head.parameters():
                parameter.requires_grad = False

    @torch.no_grad()
    def evaluation(self, data_loader):
        raw_i2t, raw_t2i, *_ = self.evaluation_artifacts(data_loader)
        if (
            str(getattr(pipeline.config, "ranking_mode", "legacy_cch")) == "raw"
            or not bool(getattr(
                pipeline.config,
                "use_candidate_context_inference",
                getattr(pipeline.config, "use_candidate_context", True),
            ))
        ):
            return raw_i2t, raw_t2i, raw_i2t, raw_t2i
        base_scores = torch.from_numpy(raw_i2t).to(self.device)
        final_i2t, final_t2i = self.model.candidate_scores(base_scores)
        return raw_i2t, raw_t2i, final_i2t.cpu().numpy(), final_t2i.cpu().numpy()

    def _evaluate_and_save(self, data_loader, dataset, stage: str, split: str):
        raw_i2t, raw_t2i, final_i2t, final_t2i = self.evaluation(data_loader)
        raw_result = self.itm_eval(raw_i2t, raw_t2i, dataset.txt2img, dataset.img2txt)
        result = self.itm_eval(final_i2t, final_t2i, dataset.txt2img, dataset.img2txt)
        path = None
        score_dir = Path(pipeline.config.evaluation_score_dir)
        if not bool(getattr(pipeline.config, "log_only", False)):
            score_dir.mkdir(parents=True, exist_ok=True)
            prefix = "" if split == "test" else f"{split}_"
            path = score_dir / f"{prefix}{stage}_scores.npz"
            np.savez_compressed(
                path,
                raw_score_i2t=np.asarray(raw_i2t, dtype=np.float32),
                raw_score_t2i=np.asarray(raw_t2i, dtype=np.float32),
                score_i2t=np.asarray(final_i2t, dtype=np.float32),
                score_t2i=np.asarray(final_t2i, dtype=np.float32),
                txt2img=compact_txt2img(dataset.txt2img),
                img2txt=compact_img2txt(dataset.img2txt),
                raw_r_mean=np.asarray(float(raw_result["r_mean"]), dtype=np.float32),
                r_mean=np.asarray(float(result["r_mean"]), dtype=np.float32),
            )
        record = {
            "method": getattr(pipeline.config, "method_name", METHOD_NAME),
            "engine": "beac_full_model_pure_main",
            "stage": stage,
            "split": split,
            "path": None if path is None else str(path),
            "raw_result": raw_result,
            "result": result,
            "ranking_mode": str(getattr(pipeline.config, "ranking_mode", "legacy_cch")),
            "raw_pairwise_score": {
                "global_weight": float(pipeline.config.global_score_weight),
                "local_weight": 1.0
                - float(pipeline.config.global_score_weight),
            },
            "inference": "candidate_context_head" if bool(getattr(
                pipeline.config,
                "use_candidate_context_inference",
                getattr(pipeline.config, "use_candidate_context", True),
            )) else "raw_pairwise_score",
            "head": self.model.candidate_context_head.parameters_record(),
        }
        if not bool(getattr(pipeline.config, "log_only", False)):
            with (score_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("BEAC_FULL_MODEL_RESULT", json.dumps(record, ensure_ascii=False), flush=True)
        pipeline.Tools.print(f"Start evaluating {split}_result={result}")
        return result

    def test(self):
        if bool(getattr(pipeline.config, "strict_validation_protocol", False)):
            stage = f"selected_epoch_{int(getattr(pipeline.config, 'selected_epoch', -1)):03d}"
        else:
            stage = "init" if self.eval_index == 0 else f"epoch_{self.eval_index - 1:03d}"
        self.eval_index += 1
        self.last_test_stage = stage
        return self._evaluate_and_save(
            self.create_dataset.test_loader,
            self.create_dataset.test_dataset,
            stage,
            "test",
        )

    def validation(self):
        if not bool(getattr(pipeline.config, "evaluate_validation", False)):
            return None
        stage = getattr(self, "last_test_stage", "init")
        return self._evaluate_and_save(
            self.create_dataset.val_loader,
            self.create_dataset.val_dataset,
            stage,
            "val",
        )


def apply_fixed_initialization(config: Any) -> dict[str, Any]:
    defaults, calibration = dataset_config.build_fixed_defaults()
    for key, value in defaults.items():
        setattr(config, key, value)
    config.candidate_initial_context_weight = CANDIDATE_INITIAL_CONTEXT_WEIGHT
    config.candidate_initial_mutual_weight = CANDIDATE_INITIAL_MUTUAL_WEIGHT
    return {"calibration": calibration, "defaults": defaults}


def apply_ablation(config: Any, ablation: str) -> dict[str, Any]:
    ablation = ablation.lower()
    if ablation not in ABLATION_DESCRIPTIONS:
        raise ValueError(f"Unknown ablation {ablation!r}; expected one of {sorted(ABLATION_DESCRIPTIONS)}")
    if ablation.startswith("m"):
        config.use_local_alignment = ablation != "m0"
        config.use_candidate_context = ablation == "m3"
        components = {"use_rank_context": True, "use_direction_context": True,
                      "use_context_mixture": True, "use_mutual_agreement": True}
    else:
        config.use_local_alignment = True
        config.use_candidate_context = True
        components = {
            "use_rank_context": ablation != "c0",
            "use_direction_context": ablation != "c1",
            "use_context_mixture": ablation != "c2",
            "use_mutual_agreement": ablation != "c3",
        }
    config.ablation = ablation
    config.ablation_description = ABLATION_DESCRIPTIONS[ablation]
    config.cch_components = components
    config.use_candidate_context_training = config.use_candidate_context
    config.use_candidate_context_inference = config.use_candidate_context
    return {
        "name": ablation,
        "description": ABLATION_DESCRIPTIONS[ablation],
        "use_local_alignment": config.use_local_alignment,
        "use_candidate_context": config.use_candidate_context,
        "use_candidate_context_training": config.use_candidate_context_training,
        "use_candidate_context_inference": config.use_candidate_context_inference,
        "cch_components": components,
    }


def parse_adapter_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise ValueError(f"Invalid --adapter-layers value: {value!r}") from exc
    if not layers or any(layer < 0 or layer > 11 for layer in layers):
        raise ValueError("--adapter-layers must contain Transformer indices in [0, 11]")
    return layers


def configure(args: argparse.Namespace) -> dict[str, Any]:
    spec = dataset_config.DATASETS[args.dataset]
    tag = args.tag or f"beac_full_model_{args.dataset}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    output_dir = Path(args.output_root) / args.dataset / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    config = spec.config_cls()
    dataset_config.configure_dataset(config, args.dataset, spec, output_dir, args.epochs)
    if args.lr is not None:
        config.lr = float(args.lr)
    config.scheduler_epochs = int(args.scheduler_epochs or args.epochs)
    config.seed = int(args.seed)
    config.setup_seed(config.seed)
    config.num_workers = int(os.environ.get("BEAC_FULL_MODEL_NUM_WORKERS", "4"))
    config.batch_size_train = min(int(config.batch_size_train), int(args.max_train_batch_size))
    if args.eval_batch_size > 0:
        config.batch_size_test = int(args.eval_batch_size)
    if args.eval_text_batch_size > 0:
        config.batch_size_test_text = int(args.eval_text_batch_size)

    fixed = apply_fixed_initialization(config)
    ablation = apply_ablation(config, args.ablation)
    if args.cch_training_enabled is not None:
        config.use_candidate_context_training = bool(args.cch_training_enabled)
    if args.cch_inference_enabled is not None:
        config.use_candidate_context_inference = bool(args.cch_inference_enabled)
    config.method_name = METHOD_NAME
    config.evaluation_score_dir = str(Path(args.score_root) / tag)
    config.evaluation_image_chunk_size = int(args.eval_image_chunk_size)
    config.adapter_layers = parse_adapter_layers(args.adapter_layers)
    config.vision_adapter_down_dim = int(args.vision_adapter_down_dim)
    config.text_adapter_down_dim = int(args.text_adapter_down_dim)
    config.vision_adapter_attention_mode = str(args.vision_adapter_attention)
    config.grouped_caption_batches = bool(args.grouped_caption_batches)

    config.global_score_weight = 0.95 if args.global_score_weight is None else float(args.global_score_weight)
    config.noun_patch_weight = 0.60 if args.noun_patch_weight is None else float(args.noun_patch_weight)
    config.patch_mask_weight = 0.40 if args.patch_mask_weight is None else float(args.patch_mask_weight)
    config.global_loss_weight = float(args.global_loss_weight)
    config.ibet_loss_weight = float(args.ibet_loss_weight)
    config.spatial_aggregation = str(args.spatial_aggregation)
    config.local_inference_aggregation = str(args.local_inference_aggregation)

    config.use_information_balance = bool(args.use_information_balance)
    config.information_balance_mode = str(args.information_balance_mode)
    config.information_balance_temperature = 0.05 if args.information_balance_temperature is None else float(args.information_balance_temperature)
    config.information_balance_steps = 24 if args.information_balance_steps is None else int(args.information_balance_steps)
    config.information_balance_evidence_mass = 0.35 if args.information_balance_evidence_mass is None else float(args.information_balance_evidence_mass)
    config.information_balance_background_logit = 0.0 if args.information_balance_background_logit is None else float(args.information_balance_background_logit)
    config.information_balance_score_mode = str(args.information_balance_score_mode)
    config.local_scoring_mode = str(args.local_scoring_mode)

    config.candidate_loss_weight = CANDIDATE_LOSS_WEIGHT if args.candidate_loss_weight is None else float(args.candidate_loss_weight)
    config.candidate_student_temperature = CANDIDATE_STUDENT_TEMPERATURE
    config.candidate_mutual_temperature = CANDIDATE_MUTUAL_TEMPERATURE
    config.candidate_mutual_scale = CANDIDATE_MUTUAL_SCALE
    config.candidate_loss_warmup_epochs = 0
    config.candidate_loss_scale = 1.0
    config.candidate_initial_rank_weight = 1.00 if args.candidate_initial_rank_weight is None else float(args.candidate_initial_rank_weight)
    config.candidate_initial_direction_weight = 0.50 if args.candidate_initial_direction_weight is None else float(args.candidate_initial_direction_weight)
    config.candidate_initial_context_weight = CANDIDATE_INITIAL_CONTEXT_WEIGHT if args.candidate_initial_context_weight is None else float(args.candidate_initial_context_weight)
    config.candidate_initial_mutual_weight = CANDIDATE_INITIAL_MUTUAL_WEIGHT if args.candidate_initial_mutual_weight is None else float(args.candidate_initial_mutual_weight)

    config.use_robust_triplet = bool(args.use_robust_triplet)
    config.robust_triplet_temperature = float(args.robust_triplet_temperature)
    config.robust_triplet_margin = float(args.robust_triplet_margin)
    config.robust_triplet_margin_min_scale = float(args.robust_triplet_margin_min_scale)
    config.robust_triplet_margin_max_scale = float(args.robust_triplet_margin_max_scale)
    config.robust_triplet_reliability_min = float(args.robust_triplet_reliability_min)
    config.robust_triplet_reliability_max = float(args.robust_triplet_reliability_max)
    config.triplet_reduction = str(args.triplet_reduction)
    config.triplet_loss_weight = 0.75 if args.triplet_loss_weight is None else float(args.triplet_loss_weight)
    config.local_loss_warmup_epochs = 0
    config.ranking_mode = str(args.ranking_mode)
    config.evaluate_validation = bool(args.evaluate_validation)
    config.strict_validation_protocol = bool(args.strict_validation_protocol)
    config.log_only = bool(args.log_only)

    if config.ranking_mode not in {"legacy_cch", "raw"}:
        raise ValueError("pure main package accepts only --ranking-mode legacy_cch or raw")
    if not 0.0 <= config.global_score_weight <= 1.0:
        raise ValueError("--global-score-weight must be in [0, 1]")
    loss_weights = (
        config.global_loss_weight,
        config.triplet_loss_weight,
        config.ibet_loss_weight,
        config.candidate_loss_weight,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in loss_weights):
        raise ValueError("loss weights must be finite and non-negative")
    triplet_parameters = (
        config.robust_triplet_margin,
        config.robust_triplet_temperature,
        config.robust_triplet_margin_min_scale,
        config.robust_triplet_margin_max_scale,
        config.robust_triplet_reliability_min,
        config.robust_triplet_reliability_max,
    )
    if not all(math.isfinite(value) for value in triplet_parameters):
        raise ValueError("robust triplet parameters must be finite")
    if config.robust_triplet_margin < 0.0:
        raise ValueError("--robust-triplet-margin must be non-negative")
    if config.robust_triplet_temperature <= 0.0:
        raise ValueError("--robust-triplet-temperature must be positive")
    if (
        config.robust_triplet_margin_min_scale < 0.0
        or config.robust_triplet_margin_max_scale < config.robust_triplet_margin_min_scale
    ):
        raise ValueError("margin scales must satisfy 0 <= min <= max")
    if not (
        0.0
        <= config.robust_triplet_reliability_min
        <= config.robust_triplet_reliability_max
        <= 1.0
    ):
        raise ValueError("reliability bounds must satisfy 0 <= min <= max <= 1")
    if config.information_balance_mode != "background_sinkhorn":
        raise ValueError("the BEAC full model implementation supports background_sinkhorn IBET only")

    pipeline.set_config(config)
    record = {
        "method": config.method_name,
        "dataset": args.dataset,
        "tag": tag,
        "backbone": os.environ.get("BEAC_FULL_MODEL_MODEL_NAME", "ViT-B/32"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "seed": config.seed,
        "epochs": config.epochs,
        "scheduler_epochs": config.scheduler_epochs,
        "num_workers": config.num_workers,
        "ablation": ablation,
        "available_components": ["global_alignment", "robust_triplet", "IBET", "CCH"],
        "loss_weights": {
            "global_alignment": config.global_loss_weight,
            "robust_triplet": config.triplet_loss_weight,
            "IBET": config.ibet_loss_weight,
            "CCH": config.candidate_loss_weight,
        },
        "adapter_layout": {"layers": list(config.adapter_layers), "image_adapter_count": len(config.adapter_layers), "text_adapter_count": len(config.adapter_layers)},
        "raw_pairwise_score": {"global_weight": config.global_score_weight, "local_weight": 1.0 - config.global_score_weight},
        "information_balance": {"enabled": config.use_information_balance, "mode": config.information_balance_mode, "temperature": config.information_balance_temperature, "steps": config.information_balance_steps, "evidence_mass": config.information_balance_evidence_mass, "score_mode": config.information_balance_score_mode},
        "aggregation": {"spatial_training": config.spatial_aggregation, "local_inference": config.local_inference_aggregation},
        "triplet": {
            "robust": config.use_robust_triplet,
            "temperature": config.robust_triplet_temperature,
            "base_margin": config.robust_triplet_margin,
            "margin_min_scale": config.robust_triplet_margin_min_scale,
            "margin_max_scale": config.robust_triplet_margin_max_scale,
            "reliability_min": config.robust_triplet_reliability_min,
            "reliability_max": config.robust_triplet_reliability_max,
            "confidence_is_detached": True,
            "loss_weight": config.triplet_loss_weight,
        },
        "candidate_context": {
            "enabled": bool(config.use_candidate_context_training or config.use_candidate_context_inference),
            "training_enabled": bool(config.use_candidate_context_training),
            "inference_enabled": bool(config.use_candidate_context_inference),
            "loss_weight": config.candidate_loss_weight,
            "components": config.cch_components,
        },
        "training_batching": {"grouped_caption_batches": config.grouped_caption_batches, "positive_definition": "same_image_id_captions_in_same_batch"},
        "strict_validation_protocol": config.strict_validation_protocol,
        "main_implementation": "beac_full_model.train_impl + beac_full_model.pipeline_core",
    }
    if not config.log_only:
        with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
    print("BEAC_FULL_MODEL_RUN_CONFIG", json.dumps(record, ensure_ascii=False), flush=True)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the pure BEAC main method.")
    parser.add_argument("--dataset", required=True, choices=sorted(dataset_config.DATASETS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--scheduler-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--score-root", default=str(SCORE_ROOT))
    parser.add_argument("--eval-image-chunk-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--eval-text-batch-size", type=int, default=0)
    parser.add_argument("--max-train-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--ablation", choices=sorted(ABLATION_DESCRIPTIONS), default="c3")
    parser.add_argument(
        "--cch-training-enabled", action=argparse.BooleanOptionalAction, default=None,
        help="Enable or disable CCH candidate-loss training independently of inference.",
    )
    parser.add_argument(
        "--cch-inference-enabled", action=argparse.BooleanOptionalAction, default=None,
        help="Enable or disable CCH inference independently of training.",
    )
    parser.add_argument("--ranking-mode", choices=("legacy_cch", "raw"), default="legacy_cch")
    parser.add_argument("--adapter-layers", default="0,11")
    parser.add_argument("--grouped-caption-batches", action="store_true")
    parser.add_argument("--vision-adapter-down-dim", type=int, default=128)
    parser.add_argument("--text-adapter-down-dim", type=int, default=128)
    parser.add_argument("--vision-adapter-attention", choices=("standard", "differential"), default="standard")
    parser.add_argument("--global-score-weight", type=float, default=None)
    parser.add_argument("--global-loss-weight", type=float, default=GLOBAL_LOSS_WEIGHT)
    parser.add_argument("--ibet-loss-weight", type=float, default=IBET_LOSS_WEIGHT)
    parser.add_argument("--candidate-loss-weight", type=float, default=None)
    parser.add_argument("--noun-patch-weight", type=float, default=None)
    parser.add_argument("--patch-mask-weight", type=float, default=None)
    parser.add_argument(
        "--spatial-aggregation",
        choices=("mean", "max"),
        default="mean",
        help="Aggregation used to form the spatial mask supervision map.",
    )
    parser.add_argument(
        "--local-inference-aggregation",
        choices=("transport_mean", "max"),
        default="transport_mean",
        help="Aggregation used by the IBET local score at inference.",
    )
    parser.add_argument("--use-information-balance", action="store_true")
    parser.add_argument("--information-balance-mode", choices=("background_sinkhorn",), default="background_sinkhorn")
    parser.add_argument("--information-balance-temperature", type=float, default=None)
    parser.add_argument("--information-balance-steps", type=int, default=None)
    parser.add_argument("--information-balance-evidence-mass", type=float, default=None)
    parser.add_argument("--information-balance-background-logit", type=float, default=None)
    parser.add_argument("--information-balance-score-mode", choices=("evidence_only", "background_aware"), default="evidence_only")
    parser.add_argument("--local-scoring-mode", choices=("uniform_mean", "ibet_ot"), default="uniform_mean")
    parser.add_argument("--use-robust-triplet", action="store_true")
    parser.add_argument("--robust-triplet-temperature", type=float, default=0.07)
    parser.add_argument("--robust-triplet-margin", type=float, default=0.1)
    parser.add_argument("--robust-triplet-margin-min-scale", type=float, default=0.5)
    parser.add_argument("--robust-triplet-margin-max-scale", type=float, default=1.5)
    parser.add_argument("--robust-triplet-reliability-min", type=float, default=0.5)
    parser.add_argument("--robust-triplet-reliability-max", type=float, default=1.0)
    parser.add_argument("--triplet-reduction", choices=("sum", "mean"), default="sum")
    parser.add_argument("--triplet-loss-weight", type=float, default=None)
    parser.add_argument("--candidate-initial-rank-weight", type=float, default=None)
    parser.add_argument("--candidate-initial-direction-weight", type=float, default=None)
    parser.add_argument("--candidate-initial-context-weight", type=float, default=None)
    parser.add_argument("--candidate-initial-mutual-weight", type=float, default=None)
    parser.add_argument("--evaluate-validation", action="store_true")
    parser.add_argument("--strict-validation-protocol", action="store_true")
    parser.add_argument("--log-only", action="store_true", help="Write training/evaluation records to logs only; do not emit run-config or score artifacts.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 0:
        raise ValueError("--epochs must be >= 0")
    if args.scheduler_epochs is not None and args.scheduler_epochs <= 0:
        raise ValueError("--scheduler-epochs must be > 0")
    if args.max_train_batch_size <= 0:
        raise ValueError("--max-train-batch-size must be > 0")
    os.chdir(PROJECT_ROOT)
    configure(args)
    result = BEACRunner().train()
    print("FINAL_RESULT", json.dumps(result, ensure_ascii=False), flush=True)

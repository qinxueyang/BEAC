"""Dataset paths and fixed paper-method initialization for BEAC."""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import pipeline


# 项目根目录用于保存 runs；数据集和模型权重通过独立环境变量指定。
PROJECT_ROOT = Path(
    os.environ.get("BEAC_FULL_MODEL_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
# DATA_ROOT 存放图像；DATASET_ROOT（由 pipeline.py 读取）存放标注 JSON。
DATA_ROOT = Path(
    os.environ.get("BEAC_FULL_MODEL_DATA_ROOT", str(PROJECT_ROOT / "runtime/data"))
).resolve()
MODEL_PATH = os.environ.get("BEAC_FULL_MODEL_MODEL_PATH", "")

@dataclass(frozen=True)
class DatasetSpec:
    config_cls: type
    image_root: str


# RSITMD D050 defaults used by the runtime configuration and CCH c3 path.
FIXED_INITIALIZATION = {
    "global_score_weight": 0.95,
    "candidate_initial_rank_weight": 1.00,
    "candidate_initial_direction_weight": 0.50,
}


DATASETS: dict[str, DatasetSpec] = {
    "rsitmd": DatasetSpec(pipeline.Config_RSITMD_ViT, str(DATA_ROOT / "organized/rsitmd")),
    "rsicd": DatasetSpec(pipeline.Config_RSICD_ViT, str(DATA_ROOT / "organized/rsicd")),
    "ucm": DatasetSpec(pipeline.Config_UCM_Caption_ViT, str(DATA_ROOT / "raw/UCM_rar/imgs")),
}


def build_fixed_defaults() -> tuple[dict[str, Any], dict[str, Any]]:
    # 返回副本，避免后续配置修改污染全局常量，并同时保留审计记录。
    """Return shared CCH initialization values used before runtime overrides."""
    calibration = {
        "profile": "beac_full_model_initialization",
        "rule": "shared_constant_values",
        "values": FIXED_INITIALIZATION,
    }
    return dict(FIXED_INITIALIZATION), calibration


def configure_dataset(
    config: Any,
    dataset: str,
    spec: DatasetSpec,
    output_dir: Path,
    epochs: int,
) -> None:
    # 将数据集 spec 和 launcher 环境变量写入具体 Config 实例，供 pipeline 使用。
    config.pretrain_path_open_clip = MODEL_PATH
    config.image_root = os.environ.get("BEAC_FULL_MODEL_IMAGE_ROOT_OVERRIDE", spec.image_root)
    config.output_dir = str(output_dir)
    config.log_filename = str(output_dir / f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}-log.txt")
    config.epochs = epochs
    config.dataset_name = dataset

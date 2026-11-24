"""论文主方法使用的数据、训练基类和数据集配置。

这里直接连接 ``pipeline_core``。历史实验实现只在显式兼容入口中存在。
"""

from __future__ import annotations

from . import pipeline_core as _core

CreateDataset = _core.CreateDataset
OpenCLIPRetrievalBase = _core.OpenCLIPRetrievalBase
TrainingRunner = _core.TrainingRunner
Tools = _core.Tools
Config_RSITMD_ViT = _core.Config_RSITMD_ViT
Config_RSICD_ViT = _core.Config_RSICD_ViT
Config_UCM_Caption_ViT = _core.Config_UCM_Caption_ViT
config = None


def set_config(value) -> None:
    """同步主方法配置到 public facade 和 core module。"""
    global config
    config = value
    _core.config = value


__all__ = [
    "CreateDataset",
    "OpenCLIPRetrievalBase",
    "TrainingRunner",
    "Tools",
    "Config_RSITMD_ViT",
    "Config_RSICD_ViT",
    "Config_UCM_Caption_ViT",
    "config",
    "set_config",
]

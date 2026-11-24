# BEAC

[English](README.md) | **简体中文**

**面向遥感图文检索的平衡证据分配与双向候选上下文校准**

## 摘要

BEAC 面向遥感图文检索，结合全局语义对齐、平衡局部证据分配和双向候选上下文校准，在 UCM-Caption、RSITMD 和 RSICD 三个数据集上开展实验。

## 方法

BEAC 采用 OpenCLIP ViT-B/32 及图像、文本适配器。平衡证据分配（BEA）利用背景槽建模名词与图像 patch 的关联，候选上下文校准（CCH）利用候选上下文调整检索分数。

## 下载

下载预训练 [OpenCLIP ViT-B/32 权重](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)。

准备 UCM-Caption、RSITMD 和 RSICD，标注文件放在 `data/finetune/` 下。各数据集的路径与参数见 [configs/](configs/)。

## 环境

Linux、Python 3.10.20、PyTorch 2.5.1+cu124（CUDA 12.4 构建）。

```bash
python -m pip install -r requirements.txt
```

在 NLTK 数据目录中准备英文词性标注资源 `averaged_perceptron_tagger_eng`。

## 训练

设置资源路径，在仓库根目录运行：

```bash
export BEAC_FULL_MODEL_DATASET_ROOT=/path/to/local_datasets
export BEAC_FULL_MODEL_DATA_ROOT=/path/to/local_datasets/runtime/data
export BEAC_FULL_MODEL_MODEL_PATH=/path/to/open_clip_pytorch_model.bin
export BEAC_FULL_MODEL_NLTK_DATA=/path/to/nltk_data

python scripts/train.py --dataset ucm --gpu 0 --run-id repro_01
python scripts/train.py --dataset rsitmd --gpu 0 --run-id repro_01
python scripts/train.py --dataset rsicd --gpu 0 --run-id repro_01
```

添加 `--dry-run` 可预览启动命令。配置分别保存在 `configs/ucm.json`、`configs/rsitmd.json` 和 `configs/rsicd.json`。

## 结果

| 数据集 | mR（%） |
|---|---:|
| UCM-Caption | 60.29 |
| RSITMD | 51.24 |
| RSICD | 38.30 |

## 致谢

感谢 [FGVLA](https://github.com/Ji-Haoyang/FGVLA) 提供的公开代码。

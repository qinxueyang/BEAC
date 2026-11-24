# BEAC

**English** | [简体中文](README_zh.md)

**Balanced Evidence Allocation and Bidirectional Candidate-Context Calibration for Remote Sensing Image–Text Retrieval**

## Abstract

BEAC addresses remote sensing image–text retrieval by combining global semantic alignment with balanced local evidence allocation and bidirectional candidate-context calibration. Experiments are conducted on UCM-Caption, RSITMD and RSICD.

## Method

BEAC uses OpenCLIP ViT-B/32 with image and text adapters. Balanced Evidence Allocation (BEA) models noun–patch associations with a background slot, while Candidate-Context Calibration (CCH) refines retrieval scores using candidate context.

## Download

Download the pretrained [OpenCLIP ViT-B/32 weights](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K).

Prepare UCM-Caption, RSITMD and RSICD with annotations under `data/finetune/`. Dataset-specific paths and parameters are defined in [configs/](configs/).

## Environment

Linux, Python 3.10.20 and PyTorch 2.5.1+cu124 (CUDA 12.4 build).

```bash
python -m pip install -r requirements.txt
```

Prepare the NLTK English tagger resource `averaged_perceptron_tagger_eng` in your NLTK data directory.

## Train

Set the resource paths and run from the repository root:

```bash
export BEAC_FULL_MODEL_DATASET_ROOT=/path/to/local_datasets
export BEAC_FULL_MODEL_DATA_ROOT=/path/to/local_datasets/runtime/data
export BEAC_FULL_MODEL_MODEL_PATH=/path/to/open_clip_pytorch_model.bin
export BEAC_FULL_MODEL_NLTK_DATA=/path/to/nltk_data

python scripts/train.py --dataset ucm --gpu 0 --run-id repro_01
python scripts/train.py --dataset rsitmd --gpu 0 --run-id repro_01
python scripts/train.py --dataset rsicd --gpu 0 --run-id repro_01
```

Add `--dry-run` to preview a command. Configurations are stored in `configs/ucm.json`, `configs/rsitmd.json` and `configs/rsicd.json`.

## Results

| Dataset | mR (%) |
|---|---:|
| UCM-Caption | 60.29 |
| RSITMD | 51.24 |
| RSICD | 38.30 |

## Acknowledgements

We thank [FGVLA](https://github.com/Ji-Haoyang/FGVLA) for its public code.

"""主方法所需的数据、基础损失、训练循环和数据集配置。

当前主方法通过 ``train.py`` 继承这里的 Dataset、Runner 和评估工具。
本文件只提供主方法所需的 OpenCLIP、数据集、检索评估和训练基类；局部对齐
和推理逻辑位于 support_alignment.py。
"""

import math
import os
import re
import json
import torch
import random
import datetime
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
try:
    from transformers.optimization import AdamW
except ImportError:  # compatible with newer Transformers releases
    from torch.optim import AdamW
from beac_full_model.data.randaugment import RandomAugment
from torchvision.transforms import InterpolationMode
from beac_full_model.utils import MetricLogger, SmoothedValue
from beac_full_model.backbone import CreateModel
from nltk import pos_tag, word_tokenize
import nltk
_nltk_data = os.environ.get("BEAC_FULL_MODEL_NLTK_DATA") or os.environ.get("NLTK_DATA")
if _nltk_data:
    nltk.data.path.append(_nltk_data)
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("BEAC_FULL_MODEL_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
DATASET_ROOT = Path(
    os.environ.get("BEAC_FULL_MODEL_DATASET_ROOT", str(PROJECT_ROOT))
).resolve()
IMAGE_DATA_ROOT = Path(
    os.environ.get("BEAC_FULL_MODEL_DATA_ROOT", str(PROJECT_ROOT / "runtime/data"))
).resolve()
DEFAULT_MODEL_PATH = os.environ.get("BEAC_FULL_MODEL_MODEL_PATH", "")




# Set by beac_full_model.train_impl.configure before dataset/model construction.
config = None



def multi_positive_triplet_loss(
    image_feat: torch.Tensor,
    text_feat: torch.Tensor,
    idx: torch.Tensor | None = None,
    margin: float = 0.1,
    reduction: str = "sum",
) -> torch.Tensor:
    """Symmetric triplet ranking loss with all same-image captions positive.

    ``idx`` identifies the source image of every paired image/text training
    item.  Captions with the same id are positives for one another, while only
    different ids participate as negatives.  ``reduction='sum'`` retains the
    historical triplet scale; ``reduction='mean'`` divides by the number of
    valid positive-negative pairs and is stable as the batch composition
    changes.  When ``idx`` is omitted, the function retains the diagonal-only
    behavior for callers without multi-caption labels.
    """
    reduction = str(reduction).lower()
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"Unsupported triplet reduction: {reduction!r}")

    # image_feat/text_feat 分别是 [B,D]，先得到 batch 内图文相似度 [B,B]。
    scores = image_feat @ text_feat.t()
    image_count, text_count = scores.shape
    if idx is None:
        # 没有 image_id 时只把对角线视为正样本。
        positive_mask = torch.eye(image_count, text_count, device=scores.device, dtype=torch.bool)
    else:
        # 同一 image_id 的多条 caption 都是正样本，只有不同 image_id 才是负样本。
        pair_ids = idx.reshape(-1).to(device=scores.device)
        if pair_ids.numel() != image_count or image_count != text_count:
            raise ValueError(
                "Multi-positive triplet loss requires one image and one text item "
                f"per batch entry, got scores={tuple(scores.shape)} and idx={pair_ids.numel()}"
            )
        positive_mask = pair_ids[:, None].eq(pair_ids[None, :])

    def directional_loss(direction_scores: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        # 对每个 anchor、positive、negative 组成 triplet：
        # [margin + negative_score - positive_score]_+。
        negatives = ~positives
        # [anchor, positive, negative]: [margin + s(anchor, negative) - s(anchor, positive)]_+
        violations = F.relu(
            float(margin)
            + direction_scores[:, None, :]
            - direction_scores[:, :, None]
        )
        valid_pairs = positives[:, :, None] & negatives[:, None, :]
        if not valid_pairs.any():
            return direction_scores.sum() * 0.0
        selected = violations.masked_select(valid_pairs)
        return selected.mean() if reduction == "mean" else selected.sum()

    return 0.5 * (
        directional_loss(scores, positive_mask)
        + directional_loss(scores.t(), positive_mask.t())
    )


def robust_multi_positive_triplet_loss(
    image_feat: torch.Tensor,
    text_feat: torch.Tensor,
    idx: torch.Tensor | None = None,
    margin: float = 0.1,
    reduction: str = "sum",
    semantic_temperature: float = 0.07,
    margin_min_scale: float = 0.5,
    margin_max_scale: float = 1.5,
    reliability_min: float = 0.5,
    reliability_max: float = 1.0,
) -> torch.Tensor:
    """Reliability-aware triplet ranking for weak/noisy RS captions.

    This is a train-only robust variant inspired by self-paced noisy-
    correspondence learning: the batch-local positive-vs-hard-negative
    semantic gap controls both the margin and the anchor reliability.  Easy,
    semantically coherent pairs receive a larger margin; ambiguous pairs
    receive a softer margin and a smaller contribution.  ``idx`` still makes
    every caption of the same image a positive, so no same-image caption is
    treated as a negative.

    The gap, margin and reliability are detached.  They therefore act as
    pseudo-label statistics for the current training batch, while gradients
    flow only through the ranking violation itself.  No validation/test
    scores enter this loss.
    """
    reduction = str(reduction).lower()
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"Unsupported triplet reduction: {reduction!r}")
    numeric_parameters = (
        float(margin),
        float(semantic_temperature),
        float(margin_min_scale),
        float(margin_max_scale),
        float(reliability_min),
        float(reliability_max),
    )
    if not all(math.isfinite(value) for value in numeric_parameters):
        raise ValueError("robust triplet parameters must be finite")
    if semantic_temperature <= 0.0:
        raise ValueError("semantic_temperature must be positive")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if margin_min_scale < 0.0 or margin_max_scale < margin_min_scale:
        raise ValueError("margin scales must satisfy 0 <= min <= max")
    if not 0.0 <= reliability_min <= reliability_max <= 1.0:
        raise ValueError("reliability bounds must satisfy 0 <= min <= max <= 1")

    scores = image_feat @ text_feat.t()
    image_count, text_count = scores.shape
    if idx is None:
        positive_mask = torch.eye(
            image_count, text_count, device=scores.device, dtype=torch.bool
        )
    else:
        pair_ids = idx.reshape(-1).to(device=scores.device)
        if pair_ids.numel() != image_count or image_count != text_count:
            raise ValueError(
                "Robust multi-positive triplet loss requires one image and one "
                f"text item per batch entry, got scores={tuple(scores.shape)} "
                f"and idx={pair_ids.numel()}"
            )
        positive_mask = pair_ids[:, None].eq(pair_ids[None, :])

    def directional_loss(
        direction_scores: torch.Tensor, positives: torch.Tensor
    ) -> torch.Tensor:
        negatives = ~positives
        positive_count = positives.sum(dim=1, keepdim=True).clamp_min(1)
        positive_score = (
            direction_scores.masked_fill(~positives, 0.0).sum(dim=1, keepdim=True)
            / positive_count
        )
        hard_negative = direction_scores.masked_fill(~negatives, -torch.inf).max(
            dim=1, keepdim=True
        ).values
        has_negative = negatives.any(dim=1, keepdim=True)
        gap = torch.where(
            has_negative, positive_score - hard_negative, torch.zeros_like(positive_score)
        ).detach()
        confidence = torch.sigmoid(gap / float(semantic_temperature)).detach().squeeze(1)
        # Defaults preserve the previous [0.5, 1.5] margin scale and [0.5, 1.0]
        # reliability range; both can now be configured from the CLI.
        dynamic_margin = float(margin) * (
            float(margin_min_scale)
            + (float(margin_max_scale) - float(margin_min_scale)) * confidence
        )
        reliability = (
            float(reliability_min)
            + (float(reliability_max) - float(reliability_min)) * confidence
        ).detach()

        violations = F.relu(
            dynamic_margin[:, None, None]
            + direction_scores[:, None, :]
            - direction_scores[:, :, None]
        )
        valid_pairs = positives[:, :, None] & negatives[:, None, :]
        if not valid_pairs.any():
            return direction_scores.sum() * 0.0
        weighted = violations * reliability[:, None, None]
        selected = weighted.masked_select(valid_pairs)
        return selected.mean() if reduction == "mean" else selected.sum()

    return 0.5 * (
        directional_loss(scores, positive_mask)
        + directional_loss(scores.t(), positive_mask.t())
    )


class RETrainDataset(Dataset):

    def __init__(self, ann_file, transform, image_root, max_words=30):
        # ann_file 是训练 JSON 文件列表；多个文件的标注会拼接到同一训练集。
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            # 将原始 image_id 映射成连续整数，供多正样本损失使用。
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
                pass
            pass
        # 保留每张图对应的全部 caption 索引，供严格 multi-positive batch 使用。
        self.caption_groups = {}
        for index, ann in enumerate(self.ann):
            self.caption_groups.setdefault(ann['image_id'], []).append(index)
        pass

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # 一个训练样本同时返回图像、清洗后的 caption、连续 image index 和原始 label。
        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.pre_caption(ann['caption'], self.max_words)
        label = torch.tensor(ann['label'])
        return image, caption, self.img_ids[ann['image_id']], label

    @staticmethod
    def pre_caption(caption, max_words):
        # 与 OpenCLIP tokenizer 对齐前，先统一大小写、标点和空白，并截断词数。
        caption = re.sub(r"([,.'!?\"()*#:;~])", '', caption.lower(),).replace(
            '-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(r"\s{2,}", ' ', caption)
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])
        if not len(caption):
            raise ValueError("pre_caption yields invalid text")
        return caption

    pass


class GroupedCaptionBatchSampler(Sampler[list[int]]):
    """Yield batches containing complete caption groups for each image.

    The global and triplet losses define all captions sharing an ``image_id``
    as positives.  A normal shuffled caption sampler rarely places those
    captions in the same batch, silently reducing the intended multi-positive
    objective to a mostly diagonal objective.  This sampler keeps each
    image's caption group intact and packs complete groups into a batch.
    """

    def __init__(self, dataset: RETrainDataset, batch_size: int, drop_last: bool = True):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.groups = list(dataset.caption_groups.values())
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

    def __iter__(self):
        groups = list(self.groups)
        random.shuffle(groups)
        batch: list[int] = []
        for group in groups:
            if len(group) > self.batch_size:
                raise ValueError(
                    f"caption group of size {len(group)} exceeds batch_size={self.batch_size}"
                )
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            batch.extend(group)
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        # Dataset annotations use fixed five-caption groups in the supported
        # benchmarks; the exact packing is still computed for robustness.
        count = 0
        current = 0
        for group in self.groups:
            if current and current + len(group) > self.batch_size:
                count += 1
                current = 0
            current += len(group)
        if current and not self.drop_last:
            count += 1
        return count


class REEvalDataset(Dataset):

    def __init__(self, ann_file, transform, image_root, max_words=30):
        # 评估标注通常是一张图像对应多条 caption，需要展开 text 和建立双向映射。
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                # txt2img 用于 T2I；img2txt 保存图像的全部正确 caption，用于 I2T。
                self.text.append(RETrainDataset.pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
            pass
        pass

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

    pass


class CreateDataset(object):

    def __init__(self):
        # config 由 train.configure() 写入 pipeline.config；这里据此创建变换和路径。
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                              (0.26862954, 0.26130258, 0.27577711))
        # pretrain_transform 是历史路径保留的预训练增强；当前训练主路径使用 train_transform。
        self.pretrain_transform = transforms.Compose([
            transforms.RandomResizedCrop(config.image_res, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(), self.normalize])
        # 训练使用随机裁剪、翻转和 RandomAugment，增加遥感图像的外观变化。
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config.image_res, scale=(0.5, 1.0),
                                         interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(), self.normalize])
        # 测试只做确定性的 resize 和归一化，保证评估可复现。
        self.test_transform = transforms.Compose([transforms.Resize((config.image_res, config.image_res),
                                                                    interpolation=InterpolationMode.BICUBIC),
                                                  transforms.ToTensor(), self.normalize])

        self.train_dataset = RETrainDataset(config.train_file, self.train_transform, config.image_root)
        self.test_dataset = REEvalDataset(config.test_file, self.test_transform, config.image_root) 
        self.val_dataset = REEvalDataset(config.val_file, self.test_transform, config.image_root)

        loader_num_workers = int(getattr(config, "num_workers", os.environ.get("BEAC_FULL_MODEL_NUM_WORKERS", 4)))
        # grouped_caption_batches 开启后，每个训练 batch 都保留完整 image_id
        # caption group，使 global/Triplet multi-positive 定义真正生效。
        if bool(getattr(config, "grouped_caption_batches", False)):
            train_sampler = GroupedCaptionBatchSampler(
                self.train_dataset,
                batch_size=config.batch_size_train,
                drop_last=True,
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_sampler=train_sampler,
                num_workers=loader_num_workers,
                pin_memory=True,
            )
        else:
            # Legacy control: caption-level random sampling.
            self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size_train,
                                           num_workers=loader_num_workers, pin_memory=True, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=config.batch_size_test,
                                      num_workers=loader_num_workers, pin_memory=True, shuffle=False, drop_last=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size_test,
                                     num_workers=loader_num_workers, pin_memory=True, shuffle=False, drop_last=False)
        pass

    pass


class OpenCLIPRetrievalBase(nn.Module):

    def __init__(self, enable_adapters: bool = True):
        super().__init__()
        # 创建带 Adapter 的 OpenCLIP ViT-B/32；真正的冻结/解冻在 Runner.set_trainable 中完成。
        create_model = CreateModel()
        model_name = os.environ.get("BEAC_FULL_MODEL_MODEL_NAME", "ViT-B/32")
        self.model = create_model.create_model_and_transforms(
            model_name,
            pretrained=config.pretrain_path_open_clip,
            enable_adapters=enable_adapters,
            vision_adapter_down_dim=int(getattr(config, "vision_adapter_down_dim", 128)),
            text_adapter_down_dim=int(getattr(config, "text_adapter_down_dim", 128)),
            vision_adapter_attention_mode=str(
                getattr(config, "vision_adapter_attention_mode", "standard")
            ),
            adapter_layers=tuple(getattr(config, "adapter_layers", (0, 11))),
        )
        print(f"Model class: {self.model.__class__}")
        self.tokenize = create_model.tokenize


    def extract_nouns(self, text):
        # NLTK 只负责 POS 名词筛选；support_alignment.py 会继续做 BPE span 对齐。
        tokens = word_tokenize(text, preserve_line=True)
        tagged = pos_tag(tokens)
        nouns = [word for word, tag in tagged if tag.startswith('NN')]
        return nouns if nouns else [tokens[0]]

    def get_vis_emb(self, image):
        # 当前 ViT 返回 (patch_features, global_feature)。
        features = self.model.encode_image(image, normalize=True)
        if isinstance(features, tuple):
            patch_feats, global_feat = features
            return patch_feats, global_feat
        else:
            return None, features


    def get_txt_emb(self, text_ids, return_word_feats=False):
        # 局部对齐需要 word/BPE 特征；纯全局实验可以只返回 sentence feature。
        if return_word_feats:
            word_feats, sent_feat = self.model.encode_text(text_ids, normalize=True, return_word_feats=True)
            return word_feats, sent_feat
        else:
            sent_feat = self.model.encode_text(text_ids, normalize=True, return_word_feats=False)
            return None, sent_feat  


    # The old standalone noun-patch forward/loss lived here in the prototype
    # implementation.  The paper path is implemented by
    # ``support_alignment.SupportAlignmentModel`` and its
    # ``sample_fine_losses`` method; keeping a second local-loss definition
    # would make the reproducibility contract ambiguous.

    def get_contrastive_loss(self, image_feat, text_feat, idx=None):
        # 使用归一化全局特征的图文相似度；idx 存在时构造多正样本 soft target。
        # 注意：logit_scale 在 OpenCLIP 主干中以 log(scale) 参数保存。
        # 这里保留历史最高分实现的原始参数乘法，以保证与历史记录严格同协议；
        # 改为 exp() 会改变优化轨迹，必须作为独立实验而不能静默替换。
        pairwise_scores = image_feat @ text_feat.t()
        logits = pairwise_scores

        logits = self.model.logit_scale * logits
        if idx is None:
            labels = torch.arange(image_feat.shape[0], device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
        else:
            idx = idx.view(-1, 1)
            pos_idx = torch.eq(idx, idx.t()).float()
            labels = pos_idx / pos_idx.sum(dim=1, keepdim=True)
            per_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1)
            per_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1)
            loss_i2t = per_i2t.mean()
            loss_t2i = per_t2i.mean()
        loss = (loss_i2t + loss_t2i) / 2.0
        return loss

    def get_triplet_loss(self, image_feat, text_feat, idx=None, margin=0.1):
        # 统一调用多正样本双向 Triplet；鲁棒变体只使用当前训练 batch。
        if bool(getattr(config, "use_robust_triplet", False)):
            return robust_multi_positive_triplet_loss(
                image_feat,
                text_feat,
                idx=idx,
                margin=float(getattr(config, "robust_triplet_margin", margin)),
                reduction=str(getattr(config, "triplet_reduction", "sum")),
                semantic_temperature=float(
                    getattr(config, "robust_triplet_temperature", 0.07)
                ),
                margin_min_scale=float(
                    getattr(config, "robust_triplet_margin_min_scale", 0.5)
                ),
                margin_max_scale=float(
                    getattr(config, "robust_triplet_margin_max_scale", 1.5)
                ),
                reliability_min=float(
                    getattr(config, "robust_triplet_reliability_min", 0.5)
                ),
                reliability_max=float(
                    getattr(config, "robust_triplet_reliability_max", 1.0)
                ),
            )
        return multi_positive_triplet_loss(
            image_feat,
            text_feat,
            idx=idx,
            margin=margin,
            reduction=str(getattr(config, "triplet_reduction", "sum")),
        )

    pass


class TrainingRunner(object):

    def __init__(self):
            # 这是当前 SupportAlignmentRunner/BEACRunner 继承的通用训练基类。
            self.device = torch.device(config.device)

            self.model = OpenCLIPRetrievalBase()
            self.tokenize = self.model.tokenize
            self.model = self.model.to(self.device)

            self.set_trainable(self.model)
            Tools.print(f"learnable parameter num = {self.count_trainable_parameters()}", txt_path=config.log_filename)

            self.create_dataset = CreateDataset()

            self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr,
                                   weight_decay=config.weight_decay, eps=1e-8, betas=(0.9, 0.98))
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, config.epochs * len(self.create_dataset.train_loader))

            self.best_weights = None
            self.best_val_score = 0
            self.warm_up_epochs = 2

    def set_trainable(model):
            # 默认基类只打开名称包含 adapter 的模块；当前支持分支在子类中补充打开。
            for name, module in model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
                pass
            for name, module in model.named_modules():
                if 'adapter' in name:
                    module.train()
                    for param in module.parameters():
                        param.requires_grad = True
                        # Tools.print(name)
                        pass
                    pass
                pass
            pass

    def load_checkpoint(model, checkpoint):
            if checkpoint != '-1':
                checkpoint_value = torch.load(checkpoint, map_location='cpu')
                state_dict = checkpoint_value['model'] if 'model' in checkpoint_value.keys() else checkpoint_value
                msg = model.load_state_dict(state_dict, strict=True)
                print("CHECKPOINT_LOAD strict=True missing=[] unexpected=[]", flush=True)
                pass
            pass

    def count_trainable_parameters(self):
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(self):
            # 新论文实验使用严格协议：只在验证集选单一 checkpoint，训练期间
            # 不读取测试集，最后加载选中的 checkpoint 后只评估一次测试集。
            if bool(getattr(config, "strict_validation_protocol", False)):
                return self._train_validation_selected()

            # 兼容旧实验：记录未训练模型的 test 结果，再进行逐 epoch 训练和 test。
            Tools.print("Start training ")
            best_result = self.test()
            if hasattr(self, "validation"):
                self.validation()
            for epoch in range(0, config.epochs):
                train_stats = self.train_one_epoch(self.create_dataset.train_loader, epoch)
                test_result = self.test()
                val_result = self.validation() if hasattr(self, "validation") else None
            
                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_result.items()},
                    'epoch': epoch,
                }
                if val_result is not None:
                    log_stats.update({f'val_{k}': v for k, v in val_result.items()})
                Tools.print(json.dumps(log_stats), txt_path=config.log_filename)
            
                if test_result['r_mean'] > best_result['r_mean']:
                    best_result = test_result
                
            return best_result

    def _strict_checkpoint_path(self):
            return os.path.join(str(config.output_dir), "best_validation_checkpoint.pth")

    def _save_validation_checkpoint(self, epoch, val_result):
            path = self._strict_checkpoint_path()
            payload = {
                "model": self.model.state_dict(),
                "epoch": int(epoch),
                "validation_result": dict(val_result),
                "selection_metric": "validation_r_mean",
            }
            torch.save(payload, path)
            return path

    def _load_validation_checkpoint(self, path):
            payload = torch.load(path, map_location="cpu")
            state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            message = self.model.load_state_dict(state_dict, strict=True)
            Tools.print("CHECKPOINT_LOAD strict=True missing=[] unexpected=[]")
            return payload

    def _train_validation_selected(self):
            """Train without test leakage and return one final test evaluation."""
            Tools.print("Start training with strict validation-only checkpoint selection")
            if not hasattr(self, "validation"):
                raise RuntimeError("Strict validation protocol requires a validation() implementation")

            self.last_test_stage = "init"
            best_val_result = self.validation()
            if best_val_result is None or "r_mean" not in best_val_result:
                raise RuntimeError("Strict validation protocol requires validation metrics with r_mean")
            best_val_score = float(best_val_result["r_mean"])
            best_epoch = -1
            best_path = self._save_validation_checkpoint(best_epoch, best_val_result)

            for epoch in range(0, config.epochs):
                train_stats = self.train_one_epoch(self.create_dataset.train_loader, epoch)
                # SupportAlignmentRunner.validation() uses this label when persisting
                # the validation score matrix; no test evaluation occurs here.
                self.last_test_stage = f"epoch_{epoch:03d}"
                val_result = self.validation()
                if val_result is None:
                    raise RuntimeError("Strict validation protocol returned no validation result")
                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"val_{k}": v for k, v in val_result.items()},
                    "epoch": epoch,
                    "selection_metric": "validation_r_mean",
                    "selected_epoch_so_far": best_epoch,
                }
                Tools.print(json.dumps(log_stats), txt_path=config.log_filename)
                current_val_score = float(val_result["r_mean"])
                if current_val_score > best_val_score:
                    best_val_score = current_val_score
                    best_epoch = epoch
                    best_val_result = val_result
                    best_path = self._save_validation_checkpoint(best_epoch, best_val_result)

            self._load_validation_checkpoint(best_path)
            config.selected_epoch = int(best_epoch)
            config.selected_checkpoint = str(best_path)
            Tools.print(
                json.dumps(
                    {
                        "STRICT_SELECTION": True,
                        "selected_epoch": best_epoch,
                        "validation_r_mean": best_val_score,
                        "checkpoint": best_path,
                        "test_evaluations_during_training": 0,
                    },
                    ensure_ascii=False,
                ),
                txt_path=config.log_filename,
            )
            # This is the only test evaluation in the strict run.
            return self.test()

    def train_one_epoch(self, data_loader, epoch):
            # 每个 batch 的 model.forward() 返回三项：全局对比（可能含 Lcc）、Triplet、局部损失。
            metric_logger = MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('loss_contrastive', SmoothedValue(window_size=1, fmt='{value:.4f}'))
            metric_logger.add_meter('loss_triplet', SmoothedValue(window_size=1, fmt='{value:.4f}'))
            metric_logger.add_meter('loss_fine', SmoothedValue(window_size=1, fmt='{value:.4f}'))


            header = 'Train Epoch: [{}]'.format(epoch)

            self.model.train()
            candidate_warmup_epochs = int(getattr(config, "candidate_loss_warmup_epochs", 0))
            if candidate_warmup_epochs > 0:
                config.candidate_loss_scale = min(
                    1.0, float(epoch + 1) / float(candidate_warmup_epochs)
                )
            else:
                config.candidate_loss_scale = 1.0
            for i, (image, text, idx, label) in enumerate(metric_logger.log_every(data_loader, 50, header)):
                # DataLoader 返回的 caption 仍是字符串，进入 OpenCLIP 前由 tokenizer 转成 [B,77]。
                image = image.to(self.device, non_blocking=True)
                idx = idx.to(self.device, non_blocking=True)
                text_input = self.tokenize(text).to(self.device)

                loss_contrastive, loss_triplet, loss_fine = self.model(
                    image, text_input,text, idx=idx, label=label)
                # 总损失：lambda_global*Lglobal + Lcc + lambda_tri*Ltri + lambda_ibet*Libet。
                # 默认 loss weights preserve the existing RSITMD D050 training behavior.
                triplet_weight = float(getattr(config, "triplet_loss_weight", 0.75))
                if triplet_weight < 0.0:
                    raise ValueError("triplet_loss_weight must be non-negative")
                ibet_weight = float(getattr(config, "ibet_loss_weight", 1.0))
                if ibet_weight < 0.0:
                    raise ValueError("ibet_loss_weight must be non-negative")
                warmup_epochs = int(getattr(config, "local_loss_warmup_epochs", 0))
                local_loss_scale = (
                    min(1.0, float(epoch + 1) / float(warmup_epochs))
                    if warmup_epochs > 0
                    else 1.0
                )
                loss = (
                    loss_contrastive
                    + triplet_weight * loss_triplet
                    + ibet_weight * local_loss_scale * loss_fine
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                metric_logger.update(loss_contrastive=loss_contrastive.item())
                metric_logger.update(loss_triplet=loss_triplet.item())
                metric_logger.update(loss_fine=loss_fine.item())
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])


            return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def test(self):
            score_test_i2t, score_test_t2i = self.evaluation(self.create_dataset.test_loader)
            test_result = self.itm_eval(score_test_i2t, score_test_t2i, self.create_dataset.test_dataset.txt2img,
                                        self.create_dataset.test_dataset.img2txt)
            Tools.print(f"Start evaluating test_result={test_result}")
            return test_result

    def itm_eval(self, scores_i2t, scores_t2i, txt2img, img2txt):
            # scores_i2t 形状 [N_img,N_txt]；scores_t2i 形状 [N_txt,N_img]。
            # 分别计算图到文和文到图的 Recall@1/5/10，并求六项平均 mR。
            # Images->Text
            ranks = np.zeros(scores_i2t.shape[0])
            for index, score in enumerate(scores_i2t):
                inds = np.argsort(score)[::-1]
                # Score
                rank = 1e20
                for i in img2txt[index]:
                    tmp = np.where(inds == i)[0][0]
                    if tmp < rank:
                        rank = tmp
                ranks[index] = rank
                pass

            # Compute metrics
            tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
            tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
            tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

            # Text->Images
            ranks = np.zeros(scores_t2i.shape[0])

            for index, score in enumerate(scores_t2i):
                inds = np.argsort(score)[::-1]
                ranks[index] = np.where(inds == txt2img[index])[0][0]
                pass

            # Compute metrics
            ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
            ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
            ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

            tr_mean = (tr1 + tr5 + tr10) / 3
            ir_mean = (ir1 + ir5 + ir10) / 3
            r_mean = (tr_mean + ir_mean) / 2

            eval_result = {'txt_r1': round(tr1, 2),
                           'txt_r5': round(tr5, 2),
                           'txt_r10': round(tr10, 2),
                           'img_r1': round(ir1, 2),
                           'img_r5': round(ir5, 2),
                           'img_r10': round(ir10, 2),
                           'r_mean': round(r_mean, 2)}
            return eval_result


class ConfigCommon(object):

    def __init__(self):
        # 数据集子类继承这里的 GPU、随机种子、优化器和 batch 基础配置。
        self.clean_gpu()
        self.gpu_id=0
        torch.cuda.set_device(self.gpu_id)
        #torch.cuda.set_device(self.get_gpu_id())
        self.setup_seed(2024)

        self.device = "cuda"
        self.epochs = 10
        self.lr = 0.001
        self.weight_decay = 0.01

        ################ Model setting  ###########################################################
        # Vision encoder setting
        self.image_res = 224  # no need modify
        self.patch_size = 32  # if use swin, set the patch_size to 32, else 16
        ############################################################################################

        ################ Training setting #########################################################
        self.pretrain_path_open_clip = DEFAULT_MODEL_PATH
        self.batch_size_train = 256
        self.batch_size_test = 128
        self.batch_size_test_text = 128
        ############################################################################################
        pass

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        pass

    @staticmethod
    def get_gpu_id():
        """
        torch.cuda.set_device(get_gpu_id())
        """
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_id, free = 0, 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            now_free = (info.free // 1048576) / 1024  # info.total, info.free, info.used
            if now_free > free:
                free = now_free
                gpu_id = i
            pass
        pynvml.nvmlShutdown()
        return gpu_id

    @staticmethod
    def clean_gpu(o=None):
        if o is not None:
            del o
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        pass

    pass


class Config_RSITMD_ViT(ConfigCommon):

    def __init__(self):
        super().__init__()

        self.output_dir = os.path.join(
            "./outputs",
            f"bootstrap_RSITMD_{os.getpid()}_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}",
        )
        self.log_filename = os.path.join(self.output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-log.txt"))

        self.model = 'vit'
        self.lr = 0.001

        ############## The train & val & test set root ############################################
        self.image_root = str(IMAGE_DATA_ROOT / 'organized/rsitmd')
        self.train_file = [str(DATASET_ROOT / 'data/finetune/rsitmd_train.json')]
        self.val_file = str(DATASET_ROOT / 'data/finetune/rsitmd_val.json')
        self.test_file = str(DATASET_ROOT / 'data/finetune/rsitmd_test.json')
        ############################################################################################
        pass

    pass


class Config_RSICD_ViT(ConfigCommon):

    def __init__(self):
        super().__init__()

        self.output_dir = os.path.join(
            "./outputs",
            f"bootstrap_RSICD_{os.getpid()}_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}",
        )
        self.log_filename = os.path.join(self.output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-log.txt"))

        self.model = 'vit'
        self.lr = 0.001

        ############## The train & val & test set root ############################################
        self.image_root = str(IMAGE_DATA_ROOT / 'organized/rsicd')
        self.train_file = [str(DATASET_ROOT / 'data/finetune/rsicd_train.json')]
        self.val_file = str(DATASET_ROOT / 'data/finetune/rsicd_val.json')
        self.test_file = str(DATASET_ROOT / 'data/finetune/rsicd_test.json')
        ############################################################################################
        pass

    pass


class Config_UCM_Caption_ViT(ConfigCommon):

    def __init__(self):
        super().__init__() 

        self.output_dir = os.path.join(
            "./outputs",
            f"bootstrap_UCM_{os.getpid()}_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}",
        )
        self.log_filename = os.path.join(self.output_dir, 
                                         datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-log.txt"))
        self.model = 'vit' 
        self.image_root = str(IMAGE_DATA_ROOT / 'raw/UCM_rar/imgs')
        converted_json_dir = DATASET_ROOT / 'data/finetune'
        self.train_file = [str(converted_json_dir / 'ucm_caption_train_converted.json')]
        self.val_file = str(converted_json_dir / 'ucm_caption_val_converted.json')
        self.test_file = str(converted_json_dir / 'ucm_caption_test_converted.json')
    
        pass 

    pass

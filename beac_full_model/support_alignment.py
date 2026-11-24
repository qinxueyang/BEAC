"""BEAC noun-patch evidence alignment and local retrieval scoring.

本文件连接了四个关键阶段：
1. caption 名词到 OpenCLIP BPE span 的对齐；
2. IBET 背景槽传输与 noun-patch / spatial-mask 损失；
3. 推理阶段全图像-全文本的局部分数矩阵计算。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.optimization import AdamW
except ImportError:
    from torch.optim import AdamW

from . import pipeline
from ._repro_compat import advance_legacy_disabled_init_rng as _advance_legacy_disabled_init_rng
from .third_party.open_clip import tokenizer as open_clip_tokenizer


EPSILON = 1e-8
DEFAULT_EVALUATION_IMAGE_CHUNK_SIZE = 4
# RSITMD D050 uses 0.95 global + 0.05 local score fusion.
DEFAULT_GLOBAL_SCORE_WEIGHT = 0.95
DEFAULT_NOUN_PATCH_WEIGHT = 0.60
DEFAULT_PATCH_MASK_WEIGHT = 0.40
DEFAULT_INFORMATION_BALANCE_TEMPERATURE = 0.05
DEFAULT_INFORMATION_BALANCE_STEPS = 24
DEFAULT_INFORMATION_BALANCE_EVIDENCE_MASS = 0.35
DEFAULT_INFORMATION_BALANCE_BACKGROUND_LOGIT = 0.0
DEFAULT_LOCAL_SCORING_MODE = "uniform_mean"
DEFAULT_INFORMATION_BALANCE_SCORE_MODE = "evidence_only"


def cfg(name: str, default):
    # pipeline.config 在 train.configure() 中写入；这里用 getattr 兼容单元测试
    # 或独立调用时没有完整配置对象的情况。
    return getattr(getattr(pipeline, "config", None), name, default)


def pairwise_score_weights(model: nn.Module | None = None) -> tuple[float, float]:
    # 这是检索分数的全局-局部融合比例。
    learnable_logit = None if model is None else getattr(model, "learnable_fusion_logit", None)
    if learnable_logit is not None:
        global_weight = float(torch.sigmoid(learnable_logit.detach()).item())
    else:
        global_weight = float(cfg("global_score_weight", DEFAULT_GLOBAL_SCORE_WEIGHT))
    if not 0.0 <= global_weight <= 1.0:
        raise ValueError("global_score_weight must be in [0, 1]")
    return global_weight, 1.0 - global_weight


def pad_nouns(nouns: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_count = max(noun.shape[0] for noun in nouns)
    feature_dim = nouns[0].shape[-1]
    values = nouns[0].new_zeros(len(nouns), max_count, feature_dim)
    mask = torch.zeros(len(nouns), max_count, device=values.device, dtype=torch.bool)
    for index, noun in enumerate(nouns):
        values[index, : noun.shape[0]] = noun
        mask[index, : noun.shape[0]] = True
    return values, mask


def information_balanced_evidence_transport(
    similarity: torch.Tensor,
    temperature: float = DEFAULT_INFORMATION_BALANCE_TEMPERATURE,
    steps: int = DEFAULT_INFORMATION_BALANCE_STEPS,
    evidence_mass: float = DEFAULT_INFORMATION_BALANCE_EVIDENCE_MASS,
    background_logit: float = DEFAULT_INFORMATION_BALANCE_BACKGROUND_LOGIT,
) -> torch.Tensor:
    """Assign noun evidence with an explicit background/null transport row.

    ``similarity`` has shape ``[M, P]`` for M noun features and P image
    patches.  The first M rows are real noun evidence and the last row is a
    background sink.  Only ``evidence_mass`` of the total mass is allowed to
    flow through real nouns; the remaining mass can be explained by the
    background row.  Uniform patch capacity prevents all nouns from collapsing
    onto the same patch.

    The logits and the returned plan are detached on purpose.  This is a
    pseudo-target construction: the alignment loss learns from the balanced
    assignment, but cannot improve by moving its own target. The background
    row is assigned a fixed zero score; ``background_logit`` remains accepted for config compatibility.

    The return value has shape ``[M + 1, P]``.  Its last row is background
    mass, while ``transport[:-1]`` is the noun-to-patch evidence plan.
    """
    if similarity.ndim != 2:
        raise ValueError("similarity must have shape [num_nouns, num_patches]")
    noun_count, patch_count = similarity.shape
    if noun_count <= 0 or patch_count <= 0:
        return similarity.new_zeros((noun_count + 1, patch_count))
    if temperature <= 0.0:
        raise ValueError("information_balance_temperature must be positive")
    if steps <= 0:
        raise ValueError("information_balance_steps must be positive")
    if not 0.0 < float(evidence_mass) < 1.0:
        raise ValueError("information_balance_evidence_mass must be in (0, 1)")

    scores = similarity.detach().float()
    # The background sink is neutral and always receives a fixed zero score.
    background_scores = scores.new_zeros((patch_count,))
    logits = torch.cat([scores, background_scores[None, :]], dim=0)
    kernel = torch.exp((logits - logits.max()) / float(temperature)).clamp_min(EPSILON)

    row_mass = scores.new_full((noun_count + 1,), float(evidence_mass) / noun_count)
    row_mass[-1] = 1.0 - float(evidence_mass)
    column_mass = scores.new_full((patch_count,), 1.0 / patch_count)
    left = scores.new_ones((noun_count + 1,))
    right = scores.new_ones((patch_count,))
    for _ in range(int(steps)):
        left = row_mass / (kernel @ right).clamp_min(EPSILON)
        right = column_mass / (kernel.transpose(0, 1) @ left).clamp_min(EPSILON)
    transport = left[:, None] * kernel * right[None, :]
    return transport.to(dtype=similarity.dtype)


def batched_information_balanced_evidence_transport(
    similarity: torch.Tensor,
    noun_mask: torch.Tensor,
    temperature: float = DEFAULT_INFORMATION_BALANCE_TEMPERATURE,
    steps: int = DEFAULT_INFORMATION_BALANCE_STEPS,
    evidence_mass: float | torch.Tensor = DEFAULT_INFORMATION_BALANCE_EVIDENCE_MASS,
    background_logit: float = DEFAULT_INFORMATION_BALANCE_BACKGROUND_LOGIT,
) -> torch.Tensor:
    """Vectorized information-balanced noun/patch transport for inference.

    Args:
        similarity: ``[B, T, M, P]`` noun-patch cosine similarities.
        noun_mask: ``[T, M]`` marking valid noun rows after padding.

    Returns:
        ``[B, T, M + 1, P]`` transport plans.  The last row is the
        background/null sink.  Unlike the historical local score, this uses
        all 49 patches and all valid nouns; no hard Top-k is applied.

    The implementation is deliberately batched because the evaluation matrix
    covers every image-caption pair.  The marginal constraints are identical
    to :func:`information_balanced_evidence_transport`, but the computation
    avoids a Python loop over gallery pairs.
    """
    if similarity.ndim != 4:
        raise ValueError("similarity must have shape [batch, texts, nouns, patches]")
    if noun_mask.ndim != 2:
        raise ValueError("noun_mask must have shape [texts, nouns]")
    batch_size, text_count, noun_count, patch_count = similarity.shape
    if noun_mask.shape != (text_count, noun_count):
        raise ValueError("noun_mask shape is inconsistent with similarity")
    if batch_size == 0 or text_count == 0 or noun_count == 0 or patch_count == 0:
        return similarity.new_zeros((batch_size, text_count, noun_count + 1, patch_count))
    if temperature <= 0.0:
        raise ValueError("information_balance_temperature must be positive")
    if steps <= 0:
        raise ValueError("information_balance_steps must be positive")

    device = similarity.device
    dtype = similarity.dtype
    valid_nouns = noun_mask.to(device=device, dtype=torch.bool)
    noun_counts = valid_nouns.sum(dim=-1).clamp_min(1).to(dtype=similarity.dtype)
    if isinstance(evidence_mass, torch.Tensor):
        mass = evidence_mass.to(device=device, dtype=torch.float32).reshape(1, text_count)
    else:
        mass = similarity.new_full((1, text_count), float(evidence_mass), dtype=torch.float32)
    if bool((mass <= 0.0).any()) or bool((mass >= 1.0).any()):
        raise ValueError("evidence_mass must be in (0, 1)")

    scores = similarity.detach().float()
    scores = scores.masked_fill(~valid_nouns[None, :, :, None], -1.0e4)
    # The background sink is neutral and always receives a fixed zero score.
    background_scores = scores.new_zeros((batch_size, text_count, patch_count))
    logits = torch.cat([scores, background_scores[:, :, None, :]], dim=2)
    kernel = torch.exp(
        (logits - logits.amax(dim=(-2, -1), keepdim=True)) / float(temperature)
    ).clamp_min(EPSILON)

    row_mass = scores.new_zeros((batch_size, text_count, noun_count + 1))
    row_mass[:, :, :noun_count] = (
        mass[:, :, None] / noun_counts[None, :, None]
    ) * valid_nouns[None, :, :].to(dtype=scores.dtype)
    row_mass[:, :, -1] = 1.0 - mass
    column_mass = scores.new_full((batch_size, text_count, patch_count), 1.0 / patch_count)
    left = scores.new_ones((batch_size, text_count, noun_count + 1))
    right = scores.new_ones((batch_size, text_count, patch_count))
    for _ in range(int(steps)):
        left = row_mass / torch.einsum("btmp,btp->btm", kernel, right).clamp_min(EPSILON)
        right = column_mass / torch.einsum("btmp,btm->btp", kernel, left).clamp_min(EPSILON)
    transport = left[:, :, :, None] * kernel * right[:, :, None, :]
    return transport.to(dtype=dtype)


def noun_bpe_token_positions(
    text: str,
    selected_nouns: Sequence[str],
    context_length: int,
) -> list[list[int]]:
    """将 caption 名词映射到不含特殊 token 的 OpenCLIP BPE 位置。

    数据集 caption 已经完成清洗。OpenCLIP 的 SOT 位于位置 0，因此实际词汇
    BPE 从位置 1 开始；一个名词可能对应多个 BPE，返回该名词的全部可见位置。
    """
    words = [word for word in text.split() if word]
    if not words:
        raise ValueError("Cannot derive noun features from an empty caption")

    tokenizer = open_clip_tokenizer._tokenizer
    pieces_per_word = [tokenizer.encode(word) for word in words]
    flattened = [piece for pieces in pieces_per_word for piece in pieces]
    if flattened != tokenizer.encode(text):
        raise ValueError(
            "Caption word-to-BPE reconstruction disagrees with the OpenCLIP tokenizer: "
            f"{text!r}"
        )

    # 预留 SOT/EOT 位置，不把 padding 当作词汇特征；超出上下文长度的部分截断。
    lexical_limit = max(0, int(context_length) - 2)
    cursor = 1
    word_to_pieces: list[list[int]] = []
    for pieces in pieces_per_word:
        end = cursor + len(pieces)
        word_to_pieces.append(list(range(cursor, min(end, lexical_limit + 1))))
        cursor = end

    requested = set(selected_nouns)
    noun_positions = [word_to_pieces[index] for index, word in enumerate(words) if word in requested]
    noun_positions = [positions for positions in noun_positions if positions]
    if noun_positions:
        return noun_positions

    # 如果名词没有可见 span，回退到第一个真实词汇 token，避免错误使用 SOT。
    for positions in word_to_pieces:
        if positions:
            return [positions]
    raise ValueError(f"Caption contains no visible lexical BPE token: {text!r}")


class SupportAlignmentModel(pipeline.OpenCLIPRetrievalBase):
    """冻结 OpenCLIP 主干 + 可训练 Adapter + caption noun-patch 对齐。"""

    def __init__(self, enable_adapters: bool = True) -> None:
        super().__init__(enable_adapters=enable_adapters)
        feature_dim = getattr(self.model.visual, "output_dim", 512)
        # Keep the historical initialization sequence without retaining the
        # removed, inactive Support-Gate projection parameters.
        _advance_legacy_disabled_init_rng(feature_dim)

    def noun_features(
        self,
        word_features: torch.Tensor,
        raw_texts: Iterable[str],
    ) -> list[torch.Tensor]:
        # 对 batch 中每条 caption：先由 NLTK 选名词，再将名词映射到对应 BPE
        # span，最后对一个名词的多个 BPE 特征做平均池化。
        noun_features: list[torch.Tensor] = []
        for index, text in enumerate(raw_texts):
            selected_nouns = self.extract_nouns(text)
            noun_positions = noun_bpe_token_positions(
                text,
                selected_nouns,
                context_length=word_features.shape[1],
            )
            pooled = [word_features[index, positions].mean(dim=0) for positions in noun_positions]
            noun_tensor = torch.stack(pooled, dim=0)
            noun_features.append(noun_tensor)
        return noun_features

    def training_features(
        self,
        image: torch.Tensor,
        text_ids: torch.Tensor,
        raw_texts: Iterable[str],
        idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Sequence[torch.Tensor], torch.Tensor]:
        # 训练阶段首先得到图像 Patch/全局特征和文本 token/全局特征。
        patches, image_global = self.get_vis_emb(image)
        use_local = bool(cfg("use_local_alignment", True))
        word_features, text_global = self.get_txt_emb(text_ids, return_word_feats=use_local)
        if not use_local:
            # M0 关闭局部对齐时，仍保留全局特征和一个零局部损失，保证主训练
            # 接口不需要分叉。
            zero_loss = image_global.sum() * 0.0
            return patches, image_global, text_global, [], zero_loss
        nouns = self.noun_features(word_features, raw_texts)
        local_losses = self.sample_fine_losses(patches, nouns)
        return patches, image_global, text_global, nouns, local_losses.mean()

    def forward(self, image, text_ids, raw_texts, idx=None, label=None):
        # SupportAlignmentModel 的基础 forward 返回三个训练损失，CCH 子类会在
        # 此基础上额外加入 candidate loss。
        patches, image_global, text_global, nouns, fine_loss = self.training_features(image, text_ids, raw_texts, idx=idx)
        contrastive = self.get_contrastive_loss(image_global, text_global, idx)
        triplet = self.get_triplet_loss(image_global, text_global, idx)
        return contrastive, triplet, fine_loss

    def sample_fine_losses(
        self,
        patch_features: torch.Tensor,
        noun_features_list: Sequence[torch.Tensor],
        temperature: float = 0.07,
    ) -> torch.Tensor:
        # 返回 batch 中每个样本的 noun-patch + patch-mask 局部损失。
        losses: list[torch.Tensor] = []
        _, patch_count, _ = patch_features.shape
        side = int(patch_count ** 0.5)
        use_information_balance = bool(cfg("use_information_balance", False))
        information_temperature = float(
            cfg("information_balance_temperature", DEFAULT_INFORMATION_BALANCE_TEMPERATURE)
        )
        information_steps = int(cfg("information_balance_steps", DEFAULT_INFORMATION_BALANCE_STEPS))
        information_mode = str(cfg("information_balance_mode", "background_sinkhorn"))
        evidence_mass = float(
            cfg("information_balance_evidence_mass", DEFAULT_INFORMATION_BALANCE_EVIDENCE_MASS)
        )
        background_logit = float(
            cfg("information_balance_background_logit", DEFAULT_INFORMATION_BALANCE_BACKGROUND_LOGIT)
        )
        for batch_index, noun_features in enumerate(noun_features_list):
            patches = patch_features[batch_index]
            # noun-Patch 相似度矩阵：[num_nouns, 49]。
            similarity = torch.matmul(noun_features, patches.transpose(0, 1))
            transport = None
            if use_information_balance:
                if information_mode == "background_sinkhorn":
                    transport = information_balanced_evidence_transport(
                        similarity,
                        temperature=information_temperature,
                        steps=information_steps,
                        evidence_mass=evidence_mass,
                        background_logit=background_logit,
                    )
                else:
                    raise ValueError(
                        "Unsupported information_balance_mode: "
                        f"{information_mode!r}; expected 'background_sinkhorn'"
                    )
            contrast_losses: list[torch.Tensor] = []
            for noun_index in range(len(noun_features)):
                noun_similarity = similarity[noun_index]
                # 训练伪监督固定取 Top-5 Patch；推理阶段不在这里使用 Top-k。
                positive_count = min(5, noun_similarity.shape[0])
                top_indices = torch.topk(noun_similarity, positive_count).indices
                logits = noun_similarity / temperature
                labels = torch.zeros_like(logits)
                if transport is None:
                    labels[top_indices] = 1.0
                elif information_mode == "background_sinkhorn":
                    # No hard Top-5 pseudo positives: normalize the real-noun
                    # transport row over all patches.  The background row has
                    # already absorbed the mass judged unrelated to the noun.
                    positive_mass = transport[noun_index]
                    labels = positive_mass / positive_mass.sum().clamp_min(EPSILON)
                contrast_losses.append(-torch.sum(F.log_softmax(logits, dim=0) * labels))

            # 将 noun 相似度聚合成 7×7 空间图。mean 是主方法默认值；max
            # 仅用于聚合算子敏感性实验，其他 IBET 目标保持不变。
            spatial_aggregation = str(cfg("spatial_aggregation", "mean"))
            noun_maps = similarity.view(len(noun_features), side, side)
            if spatial_aggregation == "mean":
                pooled = noun_maps.mean(dim=0)
            elif spatial_aggregation == "max":
                pooled = noun_maps.max(dim=0).values
            else:
                raise ValueError(
                    "Unsupported spatial_aggregation: "
                    f"{spatial_aggregation!r}; expected 'mean' or 'max'"
                )
            if (
                transport is not None
                and information_mode == "background_sinkhorn"
            ):
                target_mass = evidence_mass
                target_mask = transport[:-1].sum(dim=0) / target_mass
                target_mask = target_mask.view(side, side).clamp(0.0, 1.0)
            else:
                top_mask = torch.topk(pooled.view(-1), min(5, pooled.numel())).indices
                target_mask = torch.zeros_like(pooled)
                target_mask.view(-1).scatter_(-1, top_mask, 1)
            mask_loss = F.binary_cross_entropy_with_logits(pooled / temperature, target_mask.float())
            noun_patch_weight = float(cfg("noun_patch_weight", DEFAULT_NOUN_PATCH_WEIGHT))
            patch_mask_weight = float(cfg("patch_mask_weight", DEFAULT_PATCH_MASK_WEIGHT))
            losses.append(noun_patch_weight * torch.stack(contrast_losses).mean() + patch_mask_weight * mask_loss)
        return torch.stack(losses)

    def local_scores(self, patches: torch.Tensor, nouns: Sequence[torch.Tensor], text_globals: torch.Tensor) -> torch.Tensor:
        # 推理阶段对图像库和文本库的全部组合计算局部得分：
        # patches [N_image,49,D]，text_globals [N_text,D]。
        noun_values, noun_mask = pad_nouns(nouns)
        # similarity 的形状为 [N_image,N_text,49,max_nouns]。
        similarity = torch.einsum("ind,jmd->ijnm", F.normalize(patches, dim=-1), F.normalize(noun_values, dim=-1))
        local_scoring_mode = str(cfg("local_scoring_mode", DEFAULT_LOCAL_SCORING_MODE))
        if local_scoring_mode == "ibet_ot":
            # B1：推理阶段也使用信息平衡的软传输，而不是训练后退回
            # 到 gate 加权平均。这样训练目标与最终检索分数保持一致。
            if not bool(cfg("use_information_balance", False)):
                raise ValueError("local_scoring_mode='ibet_ot' requires use_information_balance=True")
            information_temperature = float(
                cfg("information_balance_temperature", DEFAULT_INFORMATION_BALANCE_TEMPERATURE)
            )
            information_steps = int(
                cfg("information_balance_steps", DEFAULT_INFORMATION_BALANCE_STEPS)
            )
            information_mass = float(
                cfg("information_balance_evidence_mass", DEFAULT_INFORMATION_BALANCE_EVIDENCE_MASS)
            )
            information_mass_value: float | torch.Tensor = information_mass
            # The einsum above returns [image, text, patch, noun]; transport
            # consumes [image, text, noun, patch].
            similarity_m_p = similarity.permute(0, 1, 3, 2)
            transport = batched_information_balanced_evidence_transport(
                similarity_m_p,
                noun_mask,
                temperature=information_temperature,
                steps=information_steps,
                evidence_mass=information_mass_value,
                background_logit=float(cfg(
                    "information_balance_background_logit",
                    DEFAULT_INFORMATION_BALANCE_BACKGROUND_LOGIT,
                )),
            )
            score_mode = str(
                cfg(
                    "information_balance_score_mode",
                    DEFAULT_INFORMATION_BALANCE_SCORE_MODE,
                )
            )
            if score_mode not in {"evidence_only", "background_aware"}:
                raise ValueError(
                    "Unsupported information_balance_score_mode: "
                    f"{score_mode!r}; expected 'evidence_only' or 'background_aware'"
                )
            if isinstance(information_mass_value, torch.Tensor):
                denominator = information_mass_value.to(device=similarity.device).reshape(1, -1)
            else:
                denominator = similarity.new_full(
                    (1, text_globals.shape[0]), float(information_mass_value)
                )
            real_transport = transport[:, :, :-1, :]
            real_evidence = (real_transport * similarity_m_p).sum(dim=(-1, -2))
            local_aggregation = str(cfg("local_inference_aggregation", "transport_mean"))
            if local_aggregation not in {"transport_mean", "max"}:
                raise ValueError(
                    "Unsupported local_inference_aggregation: "
                    f"{local_aggregation!r}; expected 'transport_mean' or 'max'"
                )
            if local_aggregation == "max":
                # 对每个 noun 先按其 IBET 证据行归一化，再取 patch 最大值，
                # 最后对 caption 中的有效 noun 求平均。这样 max 实验只改变
                # 聚合算子，不改变 evidence mass、背景槽或 Sinkhorn 约束。
                noun_mass = real_transport.sum(dim=-1).clamp_min(EPSILON)
                normalized_transport = real_transport / noun_mass.unsqueeze(-1)
                noun_scores = (normalized_transport * similarity_m_p).max(dim=-1).values
                noun_mask_float = noun_mask.to(device=noun_scores.device, dtype=noun_scores.dtype)
                max_evidence = (noun_scores * noun_mask_float[None, :, :]).sum(dim=-1) / noun_mask_float.sum(dim=-1).clamp_min(EPSILON)
                if score_mode == "background_aware":
                    background_score = similarity.new_zeros((1,), dtype=torch.float32)
                    background_evidence = (
                        transport[:, :, -1, :] * background_score
                    ).sum(dim=-1)
                    return max_evidence + background_evidence
                return max_evidence
            if score_mode == "background_aware":
                # Keep the null row in the balanced transport plan, but give
                # it a fixed zero score so it adds no background penalty.
                background_score = similarity.new_zeros((1,), dtype=torch.float32)
                background_evidence = (
                    transport[:, :, -1, :] * background_score
                ).sum(dim=-1)
                return real_evidence + background_evidence
            return real_evidence / denominator.clamp_min(EPSILON)

        if local_scoring_mode != "uniform_mean":
            raise ValueError(
                "Unsupported local_scoring_mode: "
                f"{local_scoring_mode!r}; expected 'uniform_mean' or 'ibet_ot'"
            )
        # E0/E2 ablations: uniformly average all patches and valid nouns.
        weights = noun_mask[None, :, None, :].to(dtype=similarity.dtype)
        # The mask has a singleton patch dimension, so its own sum counts
        # valid nouns only. Include every patch in the averaging denominator.
        valid_pair_count = weights.sum(dim=(-1, -2)) * similarity.shape[-2]
        return (similarity * weights).sum(dim=(-1, -2)) / valid_pair_count.clamp_min(EPSILON)


class SupportAlignmentRunner(pipeline.TrainingRunner):
    @staticmethod
    def set_trainable(model: SupportAlignmentModel) -> None:
        # Freeze the pretrained backbone and train only the adapters.
        for _, module in model.named_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
        for name, module in model.named_modules():
            if "adapter" in name:
                module.train()
                for parameter in module.parameters():
                    parameter.requires_grad = True

    @torch.no_grad()
    def evaluation_artifacts(self, data_loader):
        # 评估流程：提取所有图像特征 -> 提取所有文本/名词特征 -> 计算全局分数
        # -> 分块计算完整局部分数 -> 固定 0.85/0.15 融合。
        self.model.eval()
        image_patch_batches: list[torch.Tensor] = []
        image_global_batches: list[torch.Tensor] = []
        for image, _ in data_loader:
            # 图像端保留全局特征和全部 49 个 Patch。
            patch_features, global_features = self.model.get_vis_emb(image.to(self.device, non_blocking=True))
            image_patch_batches.append(patch_features)
            image_global_batches.append(global_features)
        image_patches = torch.cat(image_patch_batches, dim=0)
        image_globals = torch.cat(image_global_batches, dim=0)

        use_local = bool(cfg("use_local_alignment", True))
        nouns: list[torch.Tensor] = []
        text_global_batches: list[torch.Tensor] = []
        for start in range(0, len(data_loader.dataset.text), pipeline.config.batch_size_test_text):
            # 文本端分批编码，避免一次性编码全部 caption 超出显存。
            captions = data_loader.dataset.text[start : start + pipeline.config.batch_size_test_text]
            word_features, text_globals = self.model.get_txt_emb(
                self.tokenize(captions).to(self.device),
                return_word_feats=use_local,
            )
            if use_local:
                batch_nouns = self.model.noun_features(word_features, captions)
                nouns.extend(batch_nouns)
            text_global_batches.append(text_globals)
        text_globals = torch.cat(text_global_batches, dim=0)
        # 全局相似度矩阵：[N_image, N_text]。
        global_scores = image_globals @ text_globals.t()
        global_weight, local_weight = pairwise_score_weights(self.model)
        if use_local and local_weight > 0.0:
            local_scores = torch.empty_like(global_scores)
            chunk_size = int(cfg("evaluation_image_chunk_size", DEFAULT_EVALUATION_IMAGE_CHUNK_SIZE))
            for start in range(0, image_patches.shape[0], chunk_size):
                # 按图像块计算局部分数，逻辑上仍然覆盖完整 gallery。
                end = min(start + chunk_size, image_patches.shape[0])
                local_scores[start:end] = self.model.local_scores(image_patches[start:end], nouns, text_globals)
            # local_scores 会影响基础矩阵，但 CCH 训练时使用的 local 分支会 detach。
            scores = global_weight * global_scores + local_weight * local_scores
        else:
            scores = global_scores
        return (
            scores.cpu().numpy(),
            scores.t().cpu().numpy(),
            image_patches,
            image_globals,
            text_globals,
            nouns,
        )

    @torch.no_grad()
    def evaluation(self, data_loader):
        raw_i2t, raw_t2i, *_ = self.evaluation_artifacts(data_loader)
        return raw_i2t, raw_t2i

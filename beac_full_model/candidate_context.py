"""BEAC 的可训练候选上下文校准模块。

输入是完整的图像-文本分数矩阵，而不是单独的一对图像和文本。
模块利用行方向排序、列方向排序、I2T/T2I 方向混合以及 mutual agreement，
输出最终的两个检索方向分数矩阵。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


EPSILON = 1e-8


def logit(value: float) -> float:
    value = min(1.0 - 1e-4, max(1e-4, float(value)))
    return math.log(value / (1.0 - value))


def row_zscore(scores: torch.Tensor) -> torch.Tensor:
    """按查询行标准化分数，不改变同一行候选的排序关系。"""
    scores = scores.float()
    centered = scores - scores.mean(dim=1, keepdim=True)
    return centered / centered.std(dim=1, keepdim=True).clamp_min(EPSILON)


def rank_percentile(scores: torch.Tensor) -> torch.Tensor:
    """把每一行候选的降序排名映射到 [0, 1]。"""
    order = scores.detach().argsort(dim=1, descending=True)
    ranks = torch.empty(order.shape, dtype=torch.float32, device=scores.device)
    values = torch.linspace(1.0, 0.0, steps=scores.shape[1], device=scores.device)
    ranks.scatter_(1, order, values.view(1, -1).expand_as(ranks))
    return ranks


def positive_mask(idx: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    # 同一个 image_id 的图像/文本项互为正样本；没有 idx 时退化为对角正样本。
    if idx is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.eye(rows, cols, device=device, dtype=torch.bool)
    return idx.view(rows, 1).eq(idx.view(1, cols))


def soft_positive_targets(idx: torch.Tensor, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    # 多 caption 场景下，一个图像可能对应多个正确文本，因此目标不是 one-hot。
    positives = positive_mask(idx, rows, cols).float()
    image_to_text = positives / positives.sum(dim=1, keepdim=True).clamp_min(1.0)
    text_to_image = positives.t()
    text_to_image = text_to_image / text_to_image.sum(dim=1, keepdim=True).clamp_min(1.0)
    return image_to_text, text_to_image


def soft_cross_entropy(scores: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    return -(F.log_softmax(scores / max(float(temperature), EPSILON), dim=1) * target).sum(dim=1).mean()


def symmetric_soft_cross_entropy(
    scores_i2t: torch.Tensor,
    scores_t2i: torch.Tensor,
    idx: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    # 对 I2T 和 T2I 两个方向各计算一次多正样本交叉熵，再取平均。
    target_i2t, target_t2i = soft_positive_targets(idx, scores_i2t.shape[0], scores_i2t.shape[1])
    return 0.5 * (
        soft_cross_entropy(scores_i2t, target_i2t.to(scores_i2t.device), temperature)
        + soft_cross_entropy(scores_t2i, target_t2i.to(scores_t2i.device), temperature)
    )


class CandidateContextHead(nn.Module):
    """学习行/列排序上下文和双向候选一致性。

    论文主方法使用四个共享的可学习标量；方向特定参数和 base residual
    扩展不属于本发布包。
    """

    def __init__(self) -> None:
        super().__init__()
        # rank、direction、context 使用 sigmoid，mutual 使用 tanh。
        self.rank_logit = nn.Parameter(torch.tensor(logit(0.75), dtype=torch.float32))
        self.direction_logit = nn.Parameter(torch.tensor(logit(0.75), dtype=torch.float32))
        self.context_logit = nn.Parameter(torch.tensor(logit(1.0 - 1e-4), dtype=torch.float32))
        self.mutual_weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.use_rank_context = True
        self.use_direction_context = True
        self.use_context_mixture = True
        self.use_mutual_agreement = True

    def configure_components(
        self,
        *,
        use_rank_context: bool = True,
        use_direction_context: bool = True,
        use_context_mixture: bool = True,
        use_mutual_agreement: bool = True,
    ) -> None:
        """固定当前消融实验的 CCH 组件开关。

        关闭的组件会在 forward 中结构化屏蔽，而不是只把参数初始化为 0，
        这样训练时不会通过梯度更新重新启用。
        """
        self.use_rank_context = bool(use_rank_context)
        self.use_direction_context = bool(use_direction_context)
        self.use_context_mixture = bool(use_context_mixture)
        self.use_mutual_agreement = bool(use_mutual_agreement)

    def reset(
        self,
        rank_weight: float,
        direction_weight: float,
        context_weight: float,
        mutual_weight: float,
    ) -> None:
        # 初始化值来自固定实验配置；reset 只改数值，不改变组件开关。
        with torch.no_grad():
            self.rank_logit.fill_(logit(rank_weight))
            self.direction_logit.fill_(logit(direction_weight))
            self.context_logit.fill_(logit(context_weight))
            self.mutual_weight.fill_(mutual_weight)

    def _weights(self, base_scores: torch.Tensor) -> tuple[torch.Tensor, ...]:
        rank_logit = self.rank_logit
        direction_logit = self.direction_logit
        context_logit = self.context_logit
        mutual_weight = self.mutual_weight
        return (
            torch.sigmoid(rank_logit) if self.use_rank_context else base_scores.new_zeros(()),
            torch.sigmoid(direction_logit) if self.use_direction_context else base_scores.new_zeros(()),
            torch.sigmoid(context_logit) if self.use_context_mixture else base_scores.new_zeros(()),
            torch.tanh(mutual_weight) if self.use_mutual_agreement else base_scores.new_zeros(()),
        )

    def forward(
        self,
        base_scores: torch.Tensor,
        mutual_temperature: float,
        mutual_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # base_scores 的形状为 [N_image, N_text]。
        # 返回 image_to_text=[N_image,N_text] 和 text_to_image=[N_text,N_image]。
        # 行上下文：一张图像在所有候选 caption 中的相对位置。
        row_raw = row_zscore(base_scores)
        row_rank = row_zscore(rank_percentile(base_scores))

        # 列上下文：一条 caption 在所有候选图像中的相对位置。
        column_raw = row_zscore(base_scores.t())
        column_rank = row_zscore(rank_percentile(base_scores.t()))

        i2t_rank, i2t_direction, i2t_context, i2t_mutual = self._weights(base_scores)
        t2i_rank, t2i_direction, t2i_context, t2i_mutual = self._weights(base_scores)
        i2t_row_context = (1.0 - i2t_rank) * row_raw + i2t_rank * row_rank
        i2t_column_context = (1.0 - i2t_rank) * column_raw + i2t_rank * column_rank
        # T2I 的查询维度是文本，因此其行上下文必须转置到 [N_text, N_image]。
        # column_raw/column_rank 本身已经是 [N_text, N_image]。
        t2i_row_context = (1.0 - t2i_rank) * row_raw.t() + t2i_rank * row_rank.t()
        t2i_column_context = (1.0 - t2i_rank) * column_raw + t2i_rank * column_rank

        # 将图像到文本和文本到图像两个方向的上下文进行混合。
        image_to_text_context = (1.0 - i2t_direction) * i2t_row_context + i2t_direction * i2t_column_context.t()
        text_to_image_context = (
            (1.0 - t2i_direction) * t2i_column_context
            + t2i_direction * t2i_row_context
        )

        # mutual agreement 使用 detach 后的行/列 softmax，衡量某个候选是否在
        # 两个检索方向上同时得到较高概率；因此它不会通过该分支更新 base_scores。
        row_probability = torch.softmax(base_scores.detach() / mutual_temperature, dim=1)
        column_probability = torch.softmax(base_scores.detach().t() / mutual_temperature, dim=1).t()
        mutual = row_zscore(torch.sqrt((row_probability * column_probability).clamp_min(EPSILON)))
        i2t_mutual_scale = i2t_mutual * mutual_scale
        t2i_mutual_scale = t2i_mutual * mutual_scale

        # 最终分数不是简单的 base_scores + CCH，而是基础分数上下文、方向上下文
        # 和 mutual agreement 的组合。
        image_to_text = (1.0 - i2t_context) * row_raw + i2t_context * image_to_text_context + i2t_mutual_scale * mutual
        text_to_image = (1.0 - t2i_context) * column_raw + t2i_context * text_to_image_context + t2i_mutual_scale * mutual.t()
        return image_to_text, text_to_image

    def parameters_record(self) -> dict[str, float]:
        # 将当前实际生效的可学习权重写入 manifest，方便复现实验审计。
        def record() -> dict[str, float]:
            probe = torch.empty((), device=next(self.parameters()).device)
            rank, direction_weight, context, mutual = self._weights(probe)
            return {
                "rank_weight": round(float(rank.detach().cpu()), 6),
                "direction_weight": round(float(direction_weight.detach().cpu()), 6),
                "context_weight": round(float(context.detach().cpu()), 6),
                "mutual_weight": round(float(mutual.detach().cpu()), 6),
            }

        result = {
            "direction_specific": False,
            "use_base_residual": False,
            "use_rank_context": self.use_rank_context,
            "use_direction_context": self.use_direction_context,
            "use_context_mixture": self.use_context_mixture,
            "use_mutual_agreement": self.use_mutual_agreement,
        }
        result.update(record())
        return result

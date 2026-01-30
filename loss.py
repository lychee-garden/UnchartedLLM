"""
loss.py - xVal连续数值预测Loss函数（向量MSE + 权重惩罚）

Loss函数：
    Stage 1: Text Loss + λ·Hit Loss (向量MSE)
    Stage 2: Text Loss + λ·Hit Loss + μ·Shift Loss (向量MSE)

说明：
    - Hit Loss: 比较 predicted_embedding（向量）与 true_target_embedding（向量）
    - Shift Loss: 比较 predicted_embedding（向量）与 input_target_embedding（向量）的差距
    - predicted_embedding 直接从模型输出获取
    - target_embedding 通过 batch_value_to_xval_embedding 从数值转换

权重惩罚：loss * (1 / target_value) - 对靠近0点的误差予以更大惩罚
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from core import batch_value_to_xval_embedding, XVAL_ALPHA


# ============================================================================
# Loss函数
# ============================================================================

def compute_text_loss(
    text_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """计算Text Loss（标准语言模型交叉熵）"""
    shift_logits = text_logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    text_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction='mean')
    return text_loss


def compute_hit_loss(
    predicted_embeddings: torch.Tensor,
    true_target_values: torch.Tensor,
    base_embedding: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算Hit Loss（向量MSE + 权重惩罚）

    比较 predicted_embedding（向量，从模型输出）与 true_target_embedding（向量，从数值转换）

    Args:
        predicted_embeddings: [batch_size, seq_len, hidden_size] - 模型预测的输出embedding
        true_target_values: [batch_size, seq_len] - 真实target值
        base_embedding: [hidden_size] - [NUM] token的基础embedding
        mask: [batch_size, seq_len]

    Returns:
        hit_loss: scalar
    """
    # 将true_target_values转换为embedding向量
    true_target_embeddings = batch_value_to_xval_embedding(
        true_target_values, base_embedding
    )  # [batch_size, seq_len, hidden_size]

    # 计算向量间的MSE
    mse_loss = F.mse_loss(predicted_embeddings, true_target_embeddings, reduction='none')
    # mse_loss: [batch_size, seq_len, hidden_size]

    # 对hidden_size维度求平均，得到每个位置的loss
    mse_loss = mse_loss.mean(dim=-1)  # [batch_size, seq_len]

    # 权重惩罚：1000000 / true_target（避免除零）
    weight = 1000000.0 / (true_target_values + 1e-6)
    weighted_loss = mse_loss * weight

    # 应用mask
    if mask is not None:
        weighted_loss = weighted_loss * mask
        loss = weighted_loss.sum() / mask.sum().clamp(min=1)
    else:
        loss = weighted_loss.mean()

    return loss


def compute_shift_loss(
    predicted_embeddings: torch.Tensor,
    input_target_values: torch.Tensor,
    true_target_values: torch.Tensor,
    base_embedding: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算Shift Loss（向量MSE + 权重惩罚）

    比较 predicted_embedding（向量，从模型输出）与 input_target_embedding（向量，从输入数值转换）的差距
    目的：惩罚输入输出之间的差距，鼓励模型保持target稳定

    Args:
        predicted_embeddings: [batch_size, seq_len, hidden_size] - 模型预测的输出embedding
        input_target_values: [batch_size, seq_len] - 输入的target值
        true_target_values: [batch_size, seq_len] - 真实target值（用于权重计算）
        base_embedding: [hidden_size] - [NUM] token的基础embedding
        mask: [batch_size, seq_len]

    Returns:
        shift_loss: scalar
    """
    # 将input_target_values转换为embedding向量（输入）
    input_target_embeddings = batch_value_to_xval_embedding(
        input_target_values, base_embedding
    )  # [batch_size, seq_len, hidden_size]

    # 计算向量间的MSE：比较输出与输入的差距
    mse_loss = F.mse_loss(predicted_embeddings, input_target_embeddings, reduction='none')
    # mse_loss: [batch_size, seq_len, hidden_size]

    # 对hidden_size维度求平均，得到每个位置的loss
    mse_loss = mse_loss.mean(dim=-1)  # [batch_size, seq_len]

    # 权重惩罚：1000000 / true_target（避免除零）
    weight = 1000000.0 / (true_target_values + 1e-6)
    weighted_loss = mse_loss * weight

    # 应用mask
    if mask is not None:
        weighted_loss = weighted_loss * mask
        loss = weighted_loss.sum() / mask.sum().clamp(min=1)
    else:
        loss = weighted_loss.mean()

    return loss


# ============================================================================
# 组合Loss函数
# ============================================================================

def compute_loss_stage1(
    text_logits: torch.Tensor,
    predicted_embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    true_target_values: torch.Tensor,
    base_embedding: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    lambda_reg: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Stage 1 Loss: Text Loss + λ·Hit Loss

    Args:
        text_logits: [batch_size, seq_len, vocab_size]
        predicted_embeddings: [batch_size, seq_len, hidden_size] - 模型预测的输出embedding
        input_ids: [batch_size, seq_len]
        true_target_values: [batch_size, seq_len] - 真实target值
        base_embedding: [hidden_size] - [NUM] token的基础embedding
        attention_mask: [batch_size, seq_len]
        lambda_reg: Hit Loss权重
    """
    text_loss = compute_text_loss(text_logits, input_ids, attention_mask)
    hit_loss = compute_hit_loss(predicted_embeddings, true_target_values, base_embedding, mask=attention_mask)

    total_loss = text_loss + lambda_reg * hit_loss

    loss_dict = {
        'total_loss': total_loss.item(),
        'text_loss': text_loss.item(),
        'hit_loss': hit_loss.item(),
        'weighted_hit_loss': (lambda_reg * hit_loss).item(),
        'shift_loss': 0.0
    }

    return total_loss, loss_dict


def compute_loss_stage2(
    text_logits: torch.Tensor,
    predicted_embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    true_target_values: torch.Tensor,
    input_target_values: torch.Tensor,
    base_embedding: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    mu_shift: float = 0.5,
    lambda_reg: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Stage 2 Loss: Text Loss + λ·Hit Loss + μ·Shift Loss

    Args:
        text_logits: [batch_size, seq_len, vocab_size]
        predicted_embeddings: [batch_size, seq_len, hidden_size] - 模型预测的输出embedding
        input_ids: [batch_size, seq_len]
        true_target_values: [batch_size, seq_len] - 真实target值（用于Hit Loss）
        input_target_values: [batch_size, seq_len] - 输入的target值（用于Shift Loss）
        base_embedding: [hidden_size] - [NUM] token的基础embedding
        attention_mask: [batch_size, seq_len]
        mu_shift: Shift Loss权重
        lambda_reg: Hit Loss权重
    """
    text_loss = compute_text_loss(text_logits, input_ids, attention_mask)
    hit_loss = compute_hit_loss(predicted_embeddings, true_target_values, base_embedding, mask=attention_mask)
    shift_loss = compute_shift_loss(predicted_embeddings, input_target_values, true_target_values, base_embedding, mask=attention_mask)

    total_loss = text_loss + lambda_reg * hit_loss + mu_shift * shift_loss

    loss_dict = {
        'total_loss': total_loss.item(),
        'text_loss': text_loss.item(),
        'hit_loss': hit_loss.item(),
        'weighted_hit_loss': (lambda_reg * hit_loss).item(),
        'shift_loss': shift_loss.item(),
        'weighted_shift_loss': (mu_shift * shift_loss).item()
    }

    return total_loss, loss_dict


# ============================================================================
# 工具函数
# ============================================================================

def print_loss_summary(loss_dict: Dict[str, float], step: int, stage: int):
    """打印loss摘要"""
    hit_loss = loss_dict.get('hit_loss', 0.0)
    shift_loss = loss_dict.get('shift_loss', 0.0)
    print(f"[Step {step:>6}] [Stage {stage}] "
          f"Total: {loss_dict['total_loss']:>8.4f} | "
          f"Text: {loss_dict['text_loss']:>6.4f} | "
          f"Hit: {hit_loss:>8.4f} | "
          f"Shift: {shift_loss:>8.4f}")


def get_loss_function(stage: int):
    """根据阶段获取loss函数"""
    if stage == 1:
        return compute_loss_stage1
    elif stage == 2:
        return compute_loss_stage2
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")

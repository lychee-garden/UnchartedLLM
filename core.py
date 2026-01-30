"""
core.py - xVal (Continuous Value Encoding) 方案核心模块

核心创新：
1. xVal编码：使用单个 [NUM] token，数值体现在 embedding 的缩放上
2. 公式：E_x = α · x · e_base，其中 α = 1/64
3. 连续数值预测（非离散）

版本：3.0.0 - xVal
日期：2026-01-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


# ============================================================================
# 配置常量
# ============================================================================

MAX_TARGET_VALUE = 32768
XVAL_ALPHA = 1.0 / 64.0  # xVal缩放因子 α = 1/64
NUM_SPECIAL_TOKENS = 1  # 只需要一个 [NUM] token


# ============================================================================
# xVal 编码函数
# ============================================================================

def value_to_xval_embedding(
    value: float,
    base_embedding: torch.Tensor,
    alpha: float = XVAL_ALPHA
) -> torch.Tensor:
    """
    xVal编码：将数值转换为缩放的embedding

    公式: E_x = α · x · e_base

    Args:
        value: 数值 (0-32768)
        base_embedding: [NUM] token的基础embedding [hidden_size]
        alpha: 缩放因子 (默认 1/64)

    Returns:
        scaled_embedding: 缩放后的embedding [hidden_size]
    """
    scaled_value = alpha * value
    scaled_embedding = scaled_value * base_embedding
    return scaled_embedding


def batch_value_to_xval_embedding(
    values: torch.Tensor,
    base_embedding: torch.Tensor,
    alpha: float = XVAL_ALPHA
) -> torch.Tensor:
    """
    批量xVal编码：将一批数值转换为缩放的embedding

    公式: E_x = α · x · e_base

    Args:
        values: 数值张量 [batch_size, seq_len] 或 [seq_len]
        base_embedding: [NUM] token的基础embedding [hidden_size]
        alpha: 缩放因子 (默认 1/64)

    Returns:
        scaled_embeddings: 缩放后的embedding [batch_size, seq_len, hidden_size] 或 [seq_len, hidden_size]
    """
    # 缩放数值
    scaled_values = alpha * values  # [batch_size, seq_len] 或 [seq_len]

    # 扩展维度以便广播
    # base_embedding: [hidden_size]
    # scaled_values: [batch_size, seq_len] → [batch_size, seq_len, 1]
    if scaled_values.dim() == 2:
        scaled_values = scaled_values.unsqueeze(-1)  # [batch_size, seq_len, 1]
    elif scaled_values.dim() == 1:
        scaled_values = scaled_values.unsqueeze(-1)  # [seq_len, 1]

    # 广播乘法: [batch_size, seq_len, 1] * [hidden_size] → [batch_size, seq_len, hidden_size]
    scaled_embeddings = scaled_values * base_embedding

    return scaled_embeddings


def batch_to_xval_embedding(
    values: torch.Tensor,
    base_embedding: torch.Tensor,
    alpha: float = XVAL_ALPHA
) -> torch.Tensor:
    """
    批量xVal编码的别名函数（为了兼容性）

    这是 batch_value_to_xval_embedding 的别名

    Args:
        values: 数值张量 [batch_size, seq_len] 或 [seq_len]
        base_embedding: [NUM] token的基础embedding [hidden_size]
        alpha: 缩放因子 (默认 1/64)

    Returns:
        scaled_embeddings: 缩放后的embedding [batch_size, seq_len, hidden_size] 或 [seq_len, hidden_size]
    """
    return batch_value_to_xval_embedding(values, base_embedding, alpha)


# ============================================================================
# Tokenizer扩展
# ============================================================================

def extend_tokenizer_safe(tokenizer):
    """
    安全扩展tokenizer，添加1个 [NUM] token

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        num_token_id: [NUM] token的ID
    """
    # 定义Special Token
    special_token = "[NUM]"

    # 添加到tokenizer
    num_added = tokenizer.add_special_tokens({
        'additional_special_tokens': [special_token]
    })

    print(f"[core] Added {num_added} special token to tokenizer")
    print(f"[core] New vocab size: {len(tokenizer)}")

    # 获取token ID
    num_token_id = tokenizer.convert_tokens_to_ids(special_token)
    print(f"[core] [NUM] token ID: {num_token_id}")

    return num_token_id


# ============================================================================
# Embedding初始化
# ============================================================================

def initialize_num_embedding(
    model,
    num_token_id: int,
    device: str = "cuda",
    hidden_size: int = 4096,
    init_scale: float = 1.0
):
    """
    初始化 [NUM] token 的 embedding

    使用随机初始化，但确保模长适中

    Args:
        model: UnchartedModelWrapper
        num_token_id: [NUM] token的ID
        device: 设备
        hidden_size: 模型隐藏层维度
        init_scale: 初始化缩放因子
    """
    print(f"[core] Initializing [NUM] token embedding...")

    # 获取embedding层
    embedding_layer = model.base_model.get_input_embeddings()

    # 随机初始化
    with torch.no_grad():
        # 使用正态分布初始化
        new_embedding = torch.randn(hidden_size, device=device, dtype=embedding_layer.weight.dtype)

        # 归一化到单位向量，然后缩放
        new_embedding = F.normalize(new_embedding, p=2, dim=0) * init_scale

        # 写入embedding层
        embedding_layer.weight[num_token_id] = new_embedding

    print(f"[core] Initialized [NUM] token embedding")
    print(f"  Embedding norm: {new_embedding.norm().item():.4f}")


# ============================================================================
# 工具函数
# ============================================================================

def compute_alpha_from_mu(mu_shift: float) -> float:
    """
    根据mu_shift计算噪声强度alpha

    公式: α = 0.4 - 0.16 × μ

    Args:
        mu_shift: Shift loss权重 (0-1)

    Returns:
        alpha: 噪声强度
    """
    return 0.4 - 0.16 * mu_shift


def sample_noisy_target(true_target: torch.Tensor, alpha: float = 0.32) -> torch.Tensor:
    """
    为target添加噪声（用于训练鲁棒性）

    公式: noisy_target = true_target + α × U(-true_target, true_target)

    Args:
        true_target: [seq_len] 真实target值
        alpha: 噪声强度 (0-1)

    Returns:
        noisy_target: [seq_len] 带噪声的target值
    """
    # 保存原始数据类型
    original_dtype = true_target.dtype

    noise = torch.rand_like(true_target, dtype=original_dtype) * 2 - 1  # U(-1, 1)
    noise = noise * true_target * alpha
    noisy_target = true_target + noise

    # Clamp到有效范围
    noisy_target = torch.clamp(noisy_target, min=0, max=MAX_TARGET_VALUE)

    return noisy_target


def sample_relative_noisy_target(
    next_target: torch.Tensor,
    alpha: float = 0.32,
    no_noise_threshold: float = 30.0
) -> torch.Tensor:
    """
    为next_target添加相对噪声（用于训练时步骤间的噪声）

    关键特性：
    1. 噪声范围相对于next_target本身：noise ~ U(-α*next_target, +α*next_target)
    2. 当next_target < no_noise_threshold时，不添加噪声
    3. 用于训练时模拟推理过程中的不确定性

    公式:
        if next_target >= threshold:
            noisy_target = next_target + α × U(-next_target, next_target)
        else:
            noisy_target = next_target (不加噪声)

    Args:
        next_target: [batch_size, seq_len] 或 [seq_len] - 下一步的target值
        alpha: 噪声强度 (0-1)，默认0.32表示±32%的相对噪声
        no_noise_threshold: 低于此阈值不添加噪声，默认30

    Returns:
        noisy_target: [batch_size, seq_len] 或 [seq_len] - 带噪声的target值

    Example:
        next_target = 100
        alpha = 0.32
        → noise ~ U(-32, +32)
        → noisy_target ~ [68, 132]

        next_target = 20 (< 30)
        → noisy_target = 20 (不加噪声)
    """
    # 保存原始数据类型
    original_dtype = next_target.dtype

    # 创建mask：只对 >= threshold 的位置添加噪声
    noise_mask = (next_target >= no_noise_threshold).to(dtype=original_dtype)

    # 生成相对噪声: U(-1, 1) * next_target * alpha
    noise = torch.rand_like(next_target, dtype=original_dtype) * 2 - 1  # U(-1, 1)
    noise = noise * next_target * alpha  # 相对于next_target的噪声

    # 只在mask位置应用噪声
    noisy_target = next_target + noise * noise_mask

    # Clamp到有效范围
    noisy_target = torch.clamp(noisy_target, min=0, max=MAX_TARGET_VALUE)

    return noisy_target


def print_xval_info():
    """
    打印xVal方案信息
    """
    print("\n" + "="*60)
    print("xVal (Continuous Value Encoding) Configuration")
    print("="*60)
    print(f"Max target value: {MAX_TARGET_VALUE}")
    print(f"xVal alpha (α): {XVAL_ALPHA} (1/64)")
    print(f"Number of special tokens: {NUM_SPECIAL_TOKENS} ([NUM])")
    print(f"Encoding formula: E_x = α · x · e_base")
    print(f"Value range after scaling: [0, {XVAL_ALPHA * MAX_TARGET_VALUE:.2f}]")
    print("="*60 + "\n")


# ============================================================================
# 兼容性函数（用于旧代码）
# ============================================================================

def value_to_token_id(value: int) -> int:
    """
    兼容性函数：在xVal方案中，所有值都使用同一个 [NUM] token

    注意：这个函数仅用于兼容旧代码。在xVal方案中，
    实际的数值信息体现在embedding的缩放上，而不是token ID上。

    Args:
        value: 数值 (0-32768)

    Returns:
        token_id: 始终返回0（表示使用 [NUM] token）
    """
    return 0  # 在xVal方案中，只有一个token


def token_id_to_value(token_id: int) -> int:
    """
    兼容性函数：在xVal方案中无法从token ID反推数值

    注意：这个函数仅用于兼容旧代码。在xVal方案中，
    数值信息存储在embedding中，无法从token ID获取。

    Args:
        token_id: token ID

    Returns:
        value: 返回0（占位符）
    """
    return 0  # 在xVal方案中，token ID不包含数值信息


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("\n[core] Testing xVal encoding...")

    # 打印配置信息
    print_xval_info()

    # 测试xVal编码
    print("\n[core] Testing value_to_xval_embedding...")
    base_embedding = torch.randn(4096)
    test_values = [0, 100, 1000, 10000, 32768]

    print("\nValue → Scaled Embedding Norm:")
    for val in test_values:
        scaled_emb = value_to_xval_embedding(val, base_embedding)
        print(f"  {val:>6} → norm = {scaled_emb.norm().item():.4f}")

    # 测试批量编码
    print("\n[core] Testing batch_value_to_xval_embedding...")
    values = torch.tensor([[100, 200, 300], [1000, 2000, 3000]], dtype=torch.float32)
    scaled_embeddings = batch_value_to_xval_embedding(values, base_embedding)
    print(f"  Input shape: {values.shape}")
    print(f"  Output shape: {scaled_embeddings.shape}")
    print(f"  Output norms: {scaled_embeddings.norm(dim=-1)}")

    print("\n[core] All tests passed!")

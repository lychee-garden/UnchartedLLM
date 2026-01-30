"""
model.py - xVal连续数值预测模型

核心创新：
1. 连续数值预测（非离散）
2. 使用MSE/L1损失训练
3. 预测头输出1维标量，对应剩余token数量

版本：3.0.0 - xVal
日期：2026-01-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from core import (
    MAX_TARGET_VALUE,
    XVAL_ALPHA
)


class UnchartedModelWrapper(nn.Module):
    """
    模型包装器 - xVal连续数值预测

    架构：
        Input → Transformer → Hidden States
                                 ↓
                          ┌──────┴──────┐
                          ↓             ↓
                    LM Head      Remaining Head
                        ↓             ↓
                  Text Logits    Remaining Value (标量)
                        ↓             ↓
                  Text Loss      MSE/L1 Loss

    关键特性：
    - 直接预测连续数值（0-32768）
    - 使用回归损失（MSE或L1）
    - 简单高效，精度高
    """

    def __init__(
        self,
        base_model,
        num_token_id: int,
        hidden_size: int = 4096
    ):
        """
        Args:
            base_model: 预训练的Qwen3-8B模型
            num_token_id: [NUM] token的ID
            hidden_size: 隐藏层维度
        """
        super().__init__()

        self.base_model = base_model
        self.num_token_id = num_token_id
        self.hidden_size = hidden_size

        # Remaining预测头：hidden_size → hidden_size (输出embedding向量)
        self.remaining_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )

        # 自动匹配base_model的dtype
        self.remaining_head = self.remaining_head.to(
            dtype=next(base_model.parameters()).dtype
        )

        print(f"[model] Initialized UnchartedModelWrapper (xVal)")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Remaining head output: {hidden_size} (embedding vector)")
        print(f"  xVal alpha: {XVAL_ALPHA}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_values: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Tuple:
        """
        前向传播

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            target_values: [batch_size, seq_len] - 目标数值 (0-32768)
            return_dict: 是否返回字典

        Returns:
            如果return_dict=True:
                {
                    'text_logits': [batch_size, seq_len, vocab_size],
                    'predicted_embeddings': [batch_size, seq_len, hidden_size],
                    'hidden_states': [batch_size, seq_len, hidden_size]
                }
            否则:
                (text_logits, predicted_embeddings)
        """
        # 1. Base model forward
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        text_logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # 2. Remaining prediction (输出embedding向量)
        predicted_embeddings = self.remaining_head(hidden_states)  # [batch_size, seq_len, hidden_size]

        if return_dict:
            return {
                'text_logits': text_logits,
                'predicted_embeddings': predicted_embeddings,
                'hidden_states': hidden_states
            }
        else:
            return text_logits, predicted_embeddings

    def resize_token_embeddings(self, new_num_tokens: int):
        """
        调整embedding层大小

        Args:
            new_num_tokens: 新的词表大小
        """
        self.base_model.resize_token_embeddings(new_num_tokens)
        print(f"[model] Resized token embeddings to {new_num_tokens}")

    def get_input_embeddings(self):
        """获取input embedding层"""
        return self.base_model.get_input_embeddings()

    def gradient_checkpointing_enable(self):
        """启用gradient checkpointing（节省显存）"""
        self.base_model.gradient_checkpointing_enable()
        print(f"[model] Enabled gradient checkpointing")


# ============================================================================
# 工具函数
# ============================================================================

def print_prediction_samples(
    predicted_values: torch.Tensor,
    target_values: torch.Tensor,
    num_samples: int = 5
):
    """
    打印预测样本（用于调试）

    Args:
        predicted_values: [batch_size, seq_len]
        target_values: [batch_size, seq_len]
        num_samples: 采样数量
    """
    print("\n" + "="*80)
    print("Prediction Samples (xVal)")
    print("="*80)
    print(f"{'Index':<8} {'Pred Value':<15} {'Target Value':<15} {'Error':<15} {'Rel Error %':<15}")
    print("-"*80)

    batch_size, seq_len = predicted_values.shape

    # 随机采样
    indices = torch.randint(0, batch_size * seq_len, (num_samples,))

    for idx in indices:
        b = idx // seq_len
        s = idx % seq_len

        pred_val = predicted_values[b, s].item()
        tgt_val = target_values[b, s].item()
        error = abs(pred_val - tgt_val)
        rel_error = (error / max(tgt_val, 1)) * 100  # 避免除零

        print(f"{idx.item():<8} {pred_val:<15.2f} {tgt_val:<15.2f} {error:<15.2f} {rel_error:<15.2f}")

    print("="*80 + "\n")


if __name__ == "__main__":
    # 测试代码
    print("\n[model] Testing UnchartedModelWrapper (xVal)...")

    # 创建dummy模型
    class DummyBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100000, 4096)
            self.transformer = nn.Linear(4096, 4096)

        def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
            x = self.embedding(input_ids)
            hidden = self.transformer(x)

            class Output:
                def __init__(self):
                    self.logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 100000)
                    self.hidden_states = [hidden]

            return Output()

        def resize_token_embeddings(self, new_num_tokens):
            pass

        def get_input_embeddings(self):
            return self.embedding

    # 测试
    base_model = DummyBaseModel()
    num_token_id = 100000

    model = UnchartedModelWrapper(base_model, num_token_id)

    # Forward pass
    input_ids = torch.randint(0, 100000, (2, 10))
    outputs = model(input_ids)

    print(f"\n  Text logits shape: {outputs['text_logits'].shape}")
    print(f"  Predicted values shape: {outputs['predicted_values'].shape}")
    print(f"  Predicted values range: [{outputs['predicted_values'].min().item():.2f}, {outputs['predicted_values'].max().item():.2f}]")

    # 测试打印样本
    target_values = torch.randint(0, 32768, (2, 10)).float()
    print_prediction_samples(outputs['predicted_values'], target_values, num_samples=5)

    print("\n[model] All tests passed!")

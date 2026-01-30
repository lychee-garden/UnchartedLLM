"""
inference.py - xVal推理逻辑

核心创新：
1. 模型输出embedding向量
2. 通过与base_embedding的相似度计算数值
3. 系统规则：强制递减（next_target = current_target - 1）
4. 可控长度生成

版本：3.1.0
日期：2026-01-25
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from core import token_id_to_value, value_to_token_id, XVAL_ALPHA


def embedding_to_value(
    predicted_embedding: torch.Tensor,
    base_embedding: torch.Tensor,
    alpha: float = XVAL_ALPHA
) -> float:
    """
    将预测的embedding向量转换回数值

    通过计算与base_embedding的缩放关系来反推数值
    公式: E_x = α · x · e_base
    反推: x = ||E_x|| / (α · ||e_base||)

    Args:
        predicted_embedding: [hidden_size] - 预测的embedding向量
        base_embedding: [hidden_size] - [NUM] token的基础embedding
        alpha: 缩放因子 (默认 1/64)

    Returns:
        value: 预测的数值
    """
    # 计算模长比例
    pred_norm = predicted_embedding.norm(p=2)
    base_norm = base_embedding.norm(p=2)

    # 反推数值
    value = pred_norm / (alpha * base_norm)

    return value.item()


@dataclass
class RemainingState:
    """
    倒计时状态管理

    Attributes:
        current_target: 当前剩余token数量
        current_token_id: 当前对应的<R>token ID
        history: 历史记录
    """
    current_target: int
    current_token_id: int
    history: List[Tuple[int, int]]  # [(target, token_id), ...]

    def update(self, new_target: int):
        """
        更新状态（系统规则：强制递减）

        Args:
            new_target: 新的目标值
        """
        # 系统规则：强制递减
        if self.current_target > 0:
            self.current_target = self.current_target - 1
        else:
            self.current_target = 0

        # 更新token ID
        self.current_token_id = value_to_token_id(self.current_target)

        # 记录历史
        self.history.append((self.current_target, self.current_token_id))

    def should_stop(self) -> bool:
        """
        判断是否应该停止生成

        Returns:
            True if current_target <= 0
        """
        return self.current_target <= 0


def inference_step(
    model,
    input_ids: torch.Tensor,
    num_token_id: int,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9
) -> Tuple[torch.Tensor, int]:
    """
    单步推理

    Args:
        model: UnchartedModelWrapper
        input_ids: [1, seq_len]
        num_token_id: [NUM] token的ID
        attention_mask: [1, seq_len]
        temperature: 采样温度
        top_k: Top-K采样
        top_p: Top-P采样

    Returns:
        next_token_id: 下一个文本token
        predicted_value: 预测的剩余数量
    """
    model.eval()

    with torch.no_grad():
        # Forward
        outputs = model(input_ids, attention_mask, return_dict=True)

        # 1. 文本token采样
        text_logits = outputs['text_logits'][:, -1, :]  # [1, vocab_size]

        # Temperature scaling
        text_logits = text_logits / temperature

        # Top-K filtering
        if top_k > 0:
            indices_to_remove = text_logits < torch.topk(text_logits, top_k)[0][..., -1, None]
            text_logits[indices_to_remove] = float('-inf')

        # Top-P filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(text_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            text_logits[:, indices_to_remove] = float('-inf')

        # 采样
        probs = F.softmax(text_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # 2. Remaining预测（xVal方案：从embedding向量转换为数值）
        predicted_embedding = outputs['predicted_embeddings'][0, -1]  # [hidden_size]

        # 获取base_embedding
        embedding_layer = model.base_model.get_input_embeddings()
        base_embedding = embedding_layer.weight[num_token_id]

        # 转换为数值
        predicted_value = embedding_to_value(predicted_embedding, base_embedding)

    return next_token_id, int(predicted_value)


def generate_with_countdown(
    model,
    tokenizer,
    prompt: str,
    initial_target: int,
    num_token_id: int,
    max_length: int = 512,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda",
    verbose: bool = True
) -> Tuple[str, RemainingState]:
    """
    可控长度生成（倒计时机制）

    Args:
        model: UnchartedModelWrapper
        tokenizer: HuggingFace tokenizer
        prompt: 输入提示
        initial_target: 初始目标长度
        num_token_id: [NUM] token的ID
        max_length: 最大生成长度
        temperature: 采样温度
        top_k: Top-K采样
        top_p: Top-P采样
        device: 设备
        verbose: 是否打印详细信息

    Returns:
        generated_text: 生成的文本
        state: 最终状态
    """
    model.eval()

    # 1. 编码prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    if verbose:
        print(f"\n{'='*80}")
        print(f"xVal Controllable Generation")
        print(f"{'='*80}")
        print(f"Prompt: {prompt}")
        print(f"Initial target: {initial_target}")
        print(f"Max length: {max_length}")
        print(f"{'='*80}\n")

    # 2. 初始化状态
    initial_token_id = value_to_token_id(initial_target)
    state = RemainingState(
        current_target=initial_target,
        current_token_id=initial_token_id,
        history=[(initial_target, initial_token_id)]
    )

    # 3. 生成循环
    generated_tokens = []

    for step in range(max_length):
        # 单步推理
        next_token_id, predicted_value = inference_step(
            model,
            input_ids,
            num_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        # 添加到序列
        generated_tokens.append(next_token_id.item())
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        # 更新状态（系统规则：强制递减）
        state.update(predicted_value)

        # 打印进度
        if verbose and step % 10 == 0:
            decoded = tokenizer.decode(generated_tokens[-10:])
            print(f"[Step {step:>4}] Target: {state.current_target:>5} | "
                  f"Predicted: {predicted_value:>5} | "
                  f"Text: {decoded[:50]}...")

        # 检查终止条件
        # EOS token
        if next_token_id.item() == tokenizer.eos_token_id:
            if verbose:
                print(f"\n[Stop] EOS token generated.")
            break

    # 4. 解码
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Generation Complete")
        print(f"{'='*80}")
        print(f"Generated length: {len(generated_tokens)}")
        print(f"Final target: {state.current_target}")
        print(f"Generated text:\n{generated_text}")
        print(f"{'='*80}\n")

    return generated_text, state


def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    targets: List[int],
    num_token_id: int,
    max_length: int = 512,
    temperature: float = 1.0,
    device: str = "cuda"
) -> List[Tuple[str, RemainingState]]:
    """
    批量生成

    Args:
        model: UnchartedModelWrapper
        tokenizer: HuggingFace tokenizer
        prompts: 输入提示列表
        targets: 目标长度列表
        num_token_id: [NUM] token的ID
        max_length: 最大生成长度
        temperature: 采样温度
        device: 设备

    Returns:
        results: [(generated_text, state), ...]
    """
    results = []

    for prompt, target in zip(prompts, targets):
        text, state = generate_with_countdown(
            model,
            tokenizer,
            prompt,
            target,
            num_token_id,
            max_length=max_length,
            temperature=temperature,
            device=device,
            verbose=False
        )
        results.append((text, state))

    return results


# ============================================================================
# 评估函数
# ============================================================================

def evaluate_length_control(
    model,
    tokenizer,
    test_prompts: List[str],
    test_targets: List[int],
    num_token_id: int,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    评估长度控制能力

    Args:
        model: UnchartedModelWrapper
        tokenizer: HuggingFace tokenizer
        test_prompts: 测试提示列表
        test_targets: 测试目标列表
        num_token_id: [NUM] token的ID
        device: 设备

    Returns:
        metrics: 评估指标字典
    """
    print(f"\n{'='*80}")
    print(f"Evaluating Length Control")
    print(f"{'='*80}\n")

    results = batch_generate(
        model,
        tokenizer,
        test_prompts,
        test_targets,
        num_token_id,
        device=device
    )

    # 计算指标
    errors = []
    relative_errors = []

    for i, ((text, state), target) in enumerate(zip(results, test_targets)):
        actual_length = len(tokenizer.encode(text))
        error = abs(actual_length - target)
        relative_error = error / target if target > 0 else 0

        errors.append(error)
        relative_errors.append(relative_error)

        print(f"[{i+1}/{len(test_prompts)}] Target: {target:>4} | "
              f"Actual: {actual_length:>4} | "
              f"Error: {error:>4} | "
              f"Rel Error: {relative_error:.2%}")

    # 汇总
    metrics = {
        'mean_absolute_error': sum(errors) / len(errors),
        'mean_relative_error': sum(relative_errors) / len(relative_errors),
        'max_error': max(errors),
        'min_error': min(errors)
    }

    print(f"\n{'='*80}")
    print(f"Evaluation Results")
    print(f"{'='*80}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"{'='*80}\n")

    return metrics


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("\n[inference] Testing inference functions...")

    # 测试状态管理
    print("\n[Test 1] RemainingState")
    state = RemainingState(
        current_target=100,
        current_token_id=value_to_token_id(100),
        history=[]
    )

    for i in range(5):
        print(f"  Step {i}: target={state.current_target}, token_id={state.current_token_id}")
        state.update(state.current_target)

    print(f"  Should stop: {state.should_stop()}")

    print("\n[inference] All tests passed!")

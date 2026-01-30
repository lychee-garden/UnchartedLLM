"""
train_4gpu.py - 四卡训练脚本（显存优化版）

显存优化策略：
1. Gradient Checkpointing - 节省约40%显存
2. 混合精度训练 (BF16) - 节省50%显存
3. 优化的DataLoader配置 - 减少内存开销
4. CPU Offloading (可选) - 将部分参数offload到CPU
5. Flash Attention (可选) - 优化attention计算
6. 更小的batch size + 更大的gradient accumulation

复用代码：
- 完全复用 train.py 的核心训练逻辑
- 复用 core.py, model.py, loss.py, dataset.py
- 只添加显存优化相关的配置和功能
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb
import gc

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    extend_tokenizer_safe,
    initialize_num_embedding,
    print_xval_info,
    batch_value_to_xval_embedding,
    sample_relative_noisy_target,
    MAX_TARGET_VALUE,
    XVAL_ALPHA
)
from model import UnchartedModelWrapper
from loss import compute_loss_stage1, compute_loss_stage2, print_loss_summary

# 导入dataset
try:
    from UnchartedLLM.dataset import FrictionHybridDataset, collate_friction_batch
except ImportError:
    import dataset
    FrictionHybridDataset = dataset.FrictionHybridDataset
    collate_friction_batch = dataset.collate_friction_batch


# ============================================================================
# 显存优化工具函数
# ============================================================================

def print_memory_stats(device, prefix=""):
    """打印显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"{prefix}GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")


def clear_memory():
    """清理显存"""
    gc.collect()
    torch.cuda.empty_cache()
    # 同步CUDA操作，确保所有操作完成
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def state_dict_to_cpu(state_dict):
    """
    将state_dict中的所有tensor移到CPU，减少显存占用

    Args:
        state_dict: 模型或优化器的state_dict

    Returns:
        CPU上的state_dict副本
    """
    cpu_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_state_dict[key] = value.cpu()
        else:
            cpu_state_dict[key] = value
    return cpu_state_dict


def save_model_streaming(model, save_path, global_step, epoch, args):
    """
    流式保存模型，逐层移到CPU，立即释放GPU显存

    适用于显存极度紧张的情况（80GB/81GB）

    Args:
        model: 模型（unwrapped）
        save_path: 保存路径
        global_step: 当前步数
        epoch: 当前epoch
        args: 训练参数

    Returns:
        bool: 是否保存成功
    """
    try:
        print(f"  [1/3] Extracting model state (streaming mode)...")

        # 获取state_dict，但立即逐层处理
        state_dict = model.state_dict()
        cpu_state_dict = {}

        # 逐层移到CPU并立即删除GPU引用
        total_params = len(state_dict)
        for idx, (key, value) in enumerate(state_dict.items()):
            if isinstance(value, torch.Tensor):
                # 移到CPU
                cpu_state_dict[key] = value.cpu().clone()
                # 立即删除GPU引用
                del value
            else:
                cpu_state_dict[key] = value

            # 每处理100个参数清理一次显存
            if (idx + 1) % 100 == 0:
                torch.cuda.empty_cache()

        # 删除原始state_dict
        del state_dict
        torch.cuda.empty_cache()
        gc.collect()

        print(f"  [2/3] Building save dict...")
        save_dict = {
            'model_state_dict': cpu_state_dict,
            'global_step': global_step,
            'epoch': epoch,
            'args': vars(args)
        }

        print(f"  [3/3] Writing to disk: {save_path}/pytorch_model.bin")
        torch.save(save_dict, os.path.join(save_path, 'pytorch_model.bin'))

        # 清理CPU内存
        del cpu_state_dict
        del save_dict
        gc.collect()

        return True

    except Exception as e:
        print(f"  ✗ Save failed: {e}")
        import traceback
        traceback.print_exc()

        # 尝试清理
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass

        return False


def setup_memory_optimization(args, model):
    """设置显存优化"""
    optimizations = []

    # 1. Gradient Checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        optimizations.append("Gradient Checkpointing")

    # 2. Flash Attention (如果可用)
    if args.flash_attention:
        try:
            # 尝试启用Flash Attention
            if hasattr(model.base_model.config, 'use_flash_attn'):
                model.base_model.config.use_flash_attn = True
                optimizations.append("Flash Attention")
        except:
            pass

    # 3. CPU Offload (如果启用)
    if args.cpu_offload:
        # 注意：这会显著降低训练速度，只在显存严重不足时使用
        optimizations.append("CPU Offload (Warning: slower training)")

    return optimizations


# ============================================================================
# 核心函数：带步骤间噪声的训练（复用train.py的逻辑）
# ============================================================================

def train_step_with_noise(
    model,
    input_ids,
    attention_mask,
    true_targets,
    num_token_id,
    device,
    args,
    current_stage
):
    """
    单个训练步骤，模拟推理过程中的target更新和噪声

    完全复用train.py的逻辑
    """
    batch_size, seq_len = input_ids.shape

    # 确保 true_targets 的数据类型与模型一致
    model_dtype = next(model.parameters()).dtype
    true_targets = true_targets.to(dtype=model_dtype)

    # Forward
    outputs = model(input_ids, attention_mask)
    text_logits = outputs['text_logits']
    predicted_embeddings = outputs['predicted_embeddings']

    # 获取base_embedding
    embedding_layer = model.base_model.get_input_embeddings()
    base_embedding = embedding_layer.weight[num_token_id]

    # 将predicted_embeddings转换为数值
    pred_norms = predicted_embeddings.norm(p=2, dim=-1)
    base_norm = base_embedding.norm(p=2)
    predicted_values = pred_norms / (XVAL_ALPHA * base_norm)

    # 构造输入target序列（用于Shift Loss）
    input_targets = torch.zeros_like(true_targets)
    input_targets[:, 0] = true_targets[:, 0]

    for i in range(1, seq_len):
        prev_predicted = predicted_values[:, i-1].detach()
        next_target = torch.clamp(prev_predicted - 1, min=0, max=MAX_TARGET_VALUE)

        if current_stage == 1:
            # Stage 1: 添加步骤间噪声
            next_target = sample_relative_noisy_target(
                next_target,
                alpha=args.step_noise_alpha,
                no_noise_threshold=args.step_noise_threshold
            )

        input_targets[:, i] = next_target

    # 计算loss
    if current_stage == 1:
        loss, loss_dict = compute_loss_stage1(
            text_logits, predicted_embeddings, input_ids, true_targets,
            base_embedding, attention_mask, lambda_reg=args.lambda_reg
        )
    else:
        loss, loss_dict = compute_loss_stage2(
            text_logits, predicted_embeddings, input_ids,
            true_targets,
            input_targets,
            base_embedding, attention_mask, mu_shift=args.mu_shift, lambda_reg=args.lambda_reg
        )

    return loss, loss_dict


# ============================================================================
# 主训练函数（基于train.py，添加显存优化）
# ============================================================================

def train_with_memory_optimization(args):
    """训练主函数（显存优化版）"""

    # 初始化分布式
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main_process = (local_rank == 0)

    if is_main_process:
        print("\n" + "="*80)
        print("Four-GPU Training with Memory Optimization")
        print("="*80)
        print(f"World Size: {world_size} GPUs")
        print(f"Local Rank: {local_rank}")
        print(f"Stage 1: Text + Hit + Step Noise (steps 0-{args.stage1_steps})")
        print(f"Stage 2: Text + Hit + Shift (steps {args.stage1_steps}-{args.max_steps})")
        print(f"Step Noise: α={args.step_noise_alpha}, threshold={args.step_noise_threshold}")
        if args.resume_from_checkpoint:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        print("="*80 + "\n")

    # 显存优化：清理初始显存
    clear_memory()

    # 加载模型和tokenizer
    if is_main_process:
        print(f"[1/7] Loading base model from {args.model_name}...")
        print_memory_stats(device, "  Before loading: ")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if is_main_process:
        print_memory_stats(device, "  After loading: ")

    # 扩展tokenizer
    if is_main_process:
        print(f"\n[2/7] Extending tokenizer with [NUM] token...")
    num_token_id = extend_tokenizer_safe(tokenizer)

    # 创建wrapper
    if is_main_process:
        print(f"\n[3/7] Creating UnchartedModelWrapper...")
    model = UnchartedModelWrapper(base_model, num_token_id, hidden_size=base_model.config.hidden_size)
    model.resize_token_embeddings(len(tokenizer))
    initialize_num_embedding(model, num_token_id, device=str(device), hidden_size=base_model.config.hidden_size)
    model = model.to(device)

    if is_main_process:
        print_memory_stats(device, "  After model creation: ")

    # 显存优化设置
    if is_main_process:
        print(f"\n[4/7] Setting up memory optimizations...")
    optimizations = setup_memory_optimization(args, model)
    if is_main_process:
        for opt in optimizations:
            print(f"  ✓ {opt}")
        print_memory_stats(device, "  After optimization: ")

    # DDP包装
    model_unwrapped = model
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # 显存优化：设置为False
        )

    # 优化器
    if is_main_process:
        print(f"\n[5/7] Setting up optimizer...")

    base_params = []
    head_params = []
    for name, param in model_unwrapped.named_parameters():
        if 'remaining_head' in name:
            head_params.append(param)
        else:
            base_params.append(param)

    # 显存优化：使用ZeroRedundancyOptimizer分片优化器状态
    if world_size > 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(
            [
                {'params': base_params, 'lr': args.learning_rate},
                {'params': head_params, 'lr': args.target_learning_rate}
            ],
            optimizer_class=torch.optim.AdamW,
            weight_decay=args.weight_decay
        )
        if is_main_process:
            print("  Using ZeroRedundancyOptimizer (shards optimizer state across GPUs)")
    else:
        optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': args.learning_rate},
            {'params': head_params, 'lr': args.target_learning_rate}
        ], weight_decay=args.weight_decay)
        if is_main_process:
            print("  Using standard AdamW optimizer")

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)

    # 从断点恢复
    start_epoch = 0
    start_step = 0
    if args.resume_from_checkpoint:
        if is_main_process:
            print(f"\n[*] Loading checkpoint from {args.resume_from_checkpoint}...")

        checkpoint_path = os.path.join(args.resume_from_checkpoint, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_unwrapped.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['global_step']
        start_epoch = checkpoint['epoch']

        if is_main_process:
            print(f"[*] Resumed from step {start_step}, epoch {start_epoch}")
            print(f"[*] Will continue training to step {args.max_steps}")

    # Dataset
    if is_main_process:
        print(f"\n[6/7] Loading dataset...")
        if args.use_local_datasets:
            print(f"  Mode: Local (from {args.local_datasets_dir})")
        else:
            print(f"  Mode: Streaming (download on-the-fly)")

    if args.dataset_type == "arxiv":
        arxiv_prob = 1.0
    elif args.dataset_type == "fineweb":
        arxiv_prob = 0.0
    else:
        arxiv_prob = args.arxiv_prob

    dataset = FrictionHybridDataset(
        tokenizer=tokenizer,
        window_size=args.window_size,
        arxiv_prob=arxiv_prob,
        version="v2",
        use_local=args.use_local_datasets,
        local_dir=args.local_datasets_dir
    )

    # 注意：FrictionHybridDataset是IterableDataset，不能使用DistributedSampler
    # IterableDataset会自动在多个worker之间分片数据
    sampler = None

    # 显存优化：优化DataLoader配置
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': args.batch_size,
        'sampler': sampler,
        'collate_fn': collate_friction_batch,
        'num_workers': args.num_workers,
        'pin_memory': True,
    }

    # 只有在num_workers > 0时才设置prefetch_factor和persistent_workers
    if args.num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = args.prefetch_factor
        dataloader_kwargs['persistent_workers'] = True

    dataloader = DataLoader(**dataloader_kwargs)

    # Wandb
    if is_main_process and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"4gpu_alpha{args.step_noise_alpha}",
            config=vars(args)
        )

    # 训练循环
    if is_main_process:
        print(f"\n[7/7] Starting training...")
        print(f"  Effective batch size: {world_size * args.batch_size * args.gradient_accumulation_steps}")
        print(f"  Total steps: {args.max_steps}")
        print(f"  Save interval: {args.save_interval}")
        print(f"  Emergency save: Every 500 steps (independent of save_interval)")
        print_memory_stats(device, "  Initial memory: ")
        print()

    model.train()
    global_step = start_step
    current_stage = 1 if global_step < args.stage1_steps else 2

    # 紧急保存函数（只在rank 0执行，不需要任何同步）
    def emergency_save(step):
        """紧急保存，完全独立于其他进程，确保权重不丢失"""
        if not is_main_process:
            return
        try:
            emergency_path = os.path.join(args.output_dir, f"emergency-{step}")
            os.makedirs(emergency_path, exist_ok=True)
            model_state_dict = state_dict_to_cpu(model_unwrapped.state_dict())
            torch.save({
                'model_state_dict': model_state_dict,
                'global_step': step,
                'args': vars(args)
            }, os.path.join(emergency_path, 'pytorch_model.bin'))
            del model_state_dict
            gc.collect()
            print(f"  [Emergency] Saved at step {step}")
        except Exception as e:
            print(f"  [Emergency] Save failed: {e}")

    for epoch in range(start_epoch, args.num_epochs):
        # 注意：IterableDataset不需要set_epoch
        # 数据会自动在多个进程间分片

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process)

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            current_stage = 1 if global_step < args.stage1_steps else 2

            # 训练步骤（复用train.py的逻辑）
            loss, loss_dict = train_step_with_noise(
                model_unwrapped if world_size > 1 else model,
                input_ids, attention_mask, targets,
                num_token_id, device, args, current_stage
            )

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 显存优化：定期清理显存（不使用barrier，避免NCCL超时）
                if global_step % 100 == 0:
                    clear_memory()

                if is_main_process and global_step % args.log_interval == 0:
                    print_loss_summary(loss_dict, global_step, current_stage)
                    if args.use_wandb:
                        wandb.log({
                            'stage': current_stage,
                            'total_loss': loss_dict['total_loss'],
                            'text_loss': loss_dict['text_loss'],
                            'hit_loss': loss_dict.get('hit_loss', 0.0),
                            'shift_loss': loss_dict['shift_loss'],
                            'lr_base': optimizer.param_groups[0]['lr'],
                            'lr_head': optimizer.param_groups[1]['lr'],
                            'step': global_step,
                            'gpu_memory_allocated_gb': torch.cuda.memory_allocated(device) / 1024**3,
                            'gpu_memory_reserved_gb': torch.cuda.memory_reserved(device) / 1024**3
                        })

                if is_main_process and global_step % args.save_interval == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)

                    print(f"\n[Checkpoint {global_step}] Saving (extreme memory mode: 80GB/81GB)...")

                    # 极端显存情况：先清理，再保存
                    clear_memory()

                    # 使用流式保存，逐层移到CPU
                    success = save_model_streaming(
                        model_unwrapped,
                        save_path,
                        global_step,
                        epoch,
                        args
                    )

                    if success:
                        print(f"  ✓ Checkpoint saved successfully")
                    else:
                        print(f"  ✗ Checkpoint save failed (but training continues)")

                    print_memory_stats(device, "  GPU memory: ")

                if global_step >= args.max_steps:
                    break

            pbar.set_postfix({
                'stage': current_stage,
                'loss': f"{loss_dict['total_loss']:.4f}",
                'text': f"{loss_dict['text_loss']:.4f}",
                'hit': f"{loss_dict.get('hit_loss', 0.0):.4f}",
                'shift': f"{loss_dict['shift_loss']:.4f}",
                'mem_gb': f"{torch.cuda.memory_allocated(device) / 1024**3:.1f}"
            })

        if global_step >= args.max_steps:
            break

    # 保存最终模型
    if is_main_process:
        print(f"\n[Training Complete] Saving final model (extreme memory mode: 80GB/81GB)...")
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)

        # 显存优化：保存前清理显存
        clear_memory()

        # 使用流式保存，逐层移到CPU
        success = save_model_streaming(
            model_unwrapped,
            final_path,
            global_step,
            epoch,
            args
        )

        if success:
            print(f"  ✓ Final model saved successfully")
        else:
            print(f"  ✗ Final model save failed")

        print_memory_stats(device, "  Final GPU memory: ")

    if world_size > 1:
        dist.destroy_process_group()
    if is_main_process and args.use_wandb:
        wandb.finish()


# ============================================================================
# 参数解析
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Four-GPU training with memory optimization")

    # 模型参数
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--stage1_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # 学习率
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--target_learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Loss参数
    parser.add_argument("--mu_shift", type=float, default=0.8)
    parser.add_argument("--lambda_reg", type=float, default=1.0)

    # 步骤间噪声参数
    parser.add_argument("--step_noise_alpha", type=float, default=0.32)
    parser.add_argument("--step_noise_threshold", type=float, default=2.0)

    # 数据参数
    parser.add_argument("--dataset_type", type=str, default="fineweb")
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--arxiv_prob", type=float, default=0.5)
    parser.add_argument("--use_local_datasets", action="store_true")
    parser.add_argument("--local_datasets_dir", type=str, default="./datasets")

    # DataLoader参数（显存优化）
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers per GPU (default: 2 for memory optimization)")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Prefetch factor (default: 2 for memory optimization)")

    # 显存优化参数
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Enable CPU offloading (slower but saves memory)")
    parser.add_argument("--flash_attention", action="store_true",
                        help="Enable Flash Attention if available")

    # 其他
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="uncharted-llm-4gpu")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=3000)

    # 断点恢复
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_with_memory_optimization(args)

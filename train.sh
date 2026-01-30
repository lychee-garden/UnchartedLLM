#!/bin/bash

# train.sh - 多卡训练启动脚本（显存优化版）
# 两阶段训练：Stage 1 (Hit Loss + Step Noise) + Stage 2 (Hit Loss + Shift Loss)
# 显存优化策略：
#   1. Gradient Checkpointing（节省约40%显存）
#   2. 更小的batch size per GPU
#   3. 更大的gradient accumulation
#   4. BF16混合精度训练
#   5. CPU offloading（可选）
#   6. 优化的DataLoader设置

# ============================================================================
# 配置
# ============================================================================

# GPU配置 - 修改这里来改变使用的GPU数量
NUM_GPUS=4             # 使用的GPU数量（修改这里）
GPU_IDS="4,5,6,7"              # GPU ID列表，用逗号分隔（根据NUM_GPUS修改）

# 环境配置
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PYTHONPATH=/home/wanghaoxiao/temp_lizhiyuan/Qwen3:$PYTHONPATH

# 分布式训练配置
export MASTER_ADDR=localhost
export MASTER_PORT=29501
export WORLD_SIZE=$NUM_GPUS

# 显存优化配置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# NCCL调试和超时配置
export NCCL_DEBUG=WARN                    # NCCL调试级别（WARN/INFO）
export NCCL_TIMEOUT=300                   # 增加超时时间到30分钟（单位：秒）
export NCCL_ASYNC_ERROR_HANDLING=1        # 启用异步错误处理
export NCCL_IB_DISABLE=0                  # 启用InfiniBand（如果可用）
export NCCL_SOCKET_IFNAME=^lo,docker      # 排除loopback和docker接口

# 模型路径
MODEL_NAME="/home/wanghaoxiao/temp_lizhiyuan/Qwen3-8B"
OUTPUT_DIR="./checkpoints/$(date +%Y%m%d_%H%M%S)"

# 断点恢复（如果需要从断点恢复，取消注释并设置此路径）
# 使用方法：
#   1. 从断点恢复时，设置 RESUME_FROM_CHECKPOINT 为checkpoint目录路径
#   2. 同时设置 OUTPUT_DIR 为原训练目录（继续保存到同一目录）或新目录
#   3. 训练将从checkpoint的步数继续，直到 MAX_STEPS
# 示例：
#   RESUME_FROM_CHECKPOINT="./checkpoints/20240115_123456/checkpoint-5000"
#   OUTPUT_DIR="./checkpoints/20240115_123456"  # 继续使用原目录
RESUME_FROM_CHECKPOINT=""

# 训练参数（显存优化）
MAX_STEPS=250
STAGE1_STEPS=125
BATCH_SIZE=1                      # 每张卡的batch size（显存优化：保持为1）
GRADIENT_ACCUMULATION_STEPS=8     # 梯度累积步数（4卡 × 1 × 16 = 64 effective batch size）

# 学习率
LEARNING_RATE=1e-5           # Base model
TARGET_LEARNING_RATE=1e-4    # Remaining head

# Loss权重
LAMBDA_REG=1.0               # Hit loss权重
MU_SHIFT=0.8                   # Shift loss权重

# 步骤间噪声参数
STEP_NOISE_ALPHA=0.32        # 相对噪声强度：±32%
STEP_NOISE_THRESHOLD=2.00

# 数据参数
DATASET_TYPE="fineweb"         # 选项: "hybrid", "arxiv", "fineweb"
WINDOW_SIZE=1024               # 序列长度（显存优化：保持1024）

# 数据集加载模式
USE_LOCAL_DATASETS=true        # 是否使用本地数据集（推荐true以减少网络开销）
LOCAL_DATASETS_DIR="./datasets"  # 本地数据集目录

# DataLoader优化配置
NUM_WORKERS=2                  # 每张卡的worker数（显存优化：减少到2）
PREFETCH_FACTOR=2              # 预取因子（显存优化：减少到2）

# 其他参数
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
LOG_INTERVAL=10
SAVE_INTERVAL=1250

# Wandb配置
USE_WANDB=true  # 暂时禁用wandb，避免网络问题
WANDB_PROJECT="uncharted-llm"

# 显存优化选项
ENABLE_GRADIENT_CHECKPOINTING=true    # 启用梯度检查点（强烈推荐）
ENABLE_CPU_OFFLOAD=false              # 启用CPU offload（可选，会降低速度但节省显存）
ENABLE_FLASH_ATTENTION=false          # 启用Flash Attention（如果可用）

# ============================================================================
# 打印配置
# ============================================================================

echo "============================================================================"
echo "Multi-GPU Training with Memory Optimization ($NUM_GPUS GPUs)"
echo "============================================================================"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "Resume: $RESUME_FROM_CHECKPOINT"
fi
echo ""
echo "GPU Configuration:"
echo "  GPUs: $GPU_IDS ($NUM_GPUS GPUs)"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective batch size: $((NUM_GPUS * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo ""
echo "Memory Optimization:"
echo "  Gradient Checkpointing: $ENABLE_GRADIENT_CHECKPOINTING"
echo "  CPU Offload: $ENABLE_CPU_OFFLOAD"
echo "  Flash Attention: $ENABLE_FLASH_ATTENTION"
echo "  BF16 Mixed Precision: Enabled"
echo "  DataLoader Workers: $NUM_WORKERS per GPU"
echo ""
echo "Training Stages:"
echo "  Stage 1 (steps 0-$STAGE1_STEPS): Text + Hit + Step Noise"
echo "  Stage 2 (steps $STAGE1_STEPS-$MAX_STEPS): Text + Hit + Shift"
echo ""
echo "Step Noise: α=$STEP_NOISE_ALPHA, threshold=$STEP_NOISE_THRESHOLD"
echo "Learning Rates: Base=$LEARNING_RATE, Head=$TARGET_LEARNING_RATE"
echo "Loss Weights: λ=$LAMBDA_REG, μ=$MU_SHIFT"
echo "Dataset: $DATASET_TYPE, Window=$WINDOW_SIZE"
if [ "$USE_LOCAL_DATASETS" = true ]; then
    echo "Dataset Mode: Local (from $LOCAL_DATASETS_DIR)"
else
    echo "Dataset Mode: Streaming (download on-the-fly)"
fi
echo "============================================================================"
echo ""

# ============================================================================
# 显存检查
# ============================================================================

echo "Checking GPU memory..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | while IFS=, read -r idx name total free; do
    echo "  GPU $idx: $name - Total: ${total}MB, Free: ${free}MB"
done
echo ""

# ============================================================================
# 启动训练
# ============================================================================

# 构建参数
TRAIN_ARGS="--model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --max_steps $MAX_STEPS \
    --stage1_steps $STAGE1_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --target_learning_rate $TARGET_LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --mu_shift $MU_SHIFT \
    --lambda_reg $LAMBDA_REG \
    --step_noise_alpha $STEP_NOISE_ALPHA \
    --step_noise_threshold $STEP_NOISE_THRESHOLD \
    --dataset_type $DATASET_TYPE \
    --window_size $WINDOW_SIZE \
    --num_workers $NUM_WORKERS \
    --prefetch_factor $PREFETCH_FACTOR \
    --use_bf16 \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL"

# 添加可选参数
if [ "$USE_LOCAL_DATASETS" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --use_local_datasets --local_datasets_dir $LOCAL_DATASETS_DIR"
    # 强制离线模式，避免访问 HuggingFace
    export HF_OFFLINE=1
    export HF_DATASETS_OFFLINE=1  # datasets 库专用的离线模式变量
    echo "  [Info] HF_OFFLINE=1 and HF_DATASETS_OFFLINE=1 enabled (using local datasets only)"
fi

if [ "$USE_WANDB" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --use_wandb --wandb_project $WANDB_PROJECT"
fi

if [ "$ENABLE_GRADIENT_CHECKPOINTING" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --gradient_checkpointing"
fi

if [ "$ENABLE_CPU_OFFLOAD" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --cpu_offload"
fi

if [ "$ENABLE_FLASH_ATTENTION" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --flash_attention"
fi

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --resume_from_checkpoint $RESUME_FROM_CHECKPOINT"
fi

# 使用torchrun启动分布式训练
echo "Starting distributed training with torchrun..."
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py $TRAIN_ARGS

# ============================================================================
# 训练完成
# ============================================================================

echo ""
echo "============================================================================"
echo "Training Complete! Checkpoints: $OUTPUT_DIR"
echo "============================================================================"
echo ""
echo "Final GPU memory status:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits | while IFS=, read -r idx used free; do
    echo "  GPU $idx: Used: ${used}MB, Free: ${free}MB"
done
echo ""

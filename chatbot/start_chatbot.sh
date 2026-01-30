#!/bin/bash

# start_chatbot_gpu.sh - 启动 UnchartedLLM Chatbot 服务器 (支持命令行指定 GPU)

# ============================================================================
# 使用说明
# ============================================================================
#
# 用法:
#   bash start_chatbot_gpu.sh [GPU_ID] [PORT]
#
# 示例:
#   bash start_chatbot_gpu.sh 0 8765        # 使用 GPU 0, 端口 8765
#   bash start_chatbot_gpu.sh 1 8766        # 使用 GPU 1, 端口 8766
#   bash start_chatbot_gpu.sh 0,1 8765      # 使用 GPU 0 和 1, 端口 8765
#   bash start_chatbot_gpu.sh cpu 8765      # 使用 CPU, 端口 8765
#
# ============================================================================

# ============================================================================
# 解析命令行参数
# ============================================================================

# 默认配置
DEFAULT_GPU_ID="6"
DEFAULT_PORT=8765

# 从命令行参数获取配置
GPU_ID=${1:-$DEFAULT_GPU_ID}
PORT=${2:-$DEFAULT_PORT}

# ============================================================================
# 配置
# ============================================================================

# 模型路径 (请修改为你的模型checkpoint路径)
MODEL_PATH="./checkpoints/20260128_162550/final/pytorch_model.bin"

# 基础模型路径
BASE_MODEL="/home/wanghaoxiao/temp_lizhiyuan/Qwen3-8B"

# 服务器配置
HOST="0.0.0.0"

# 设备类型
if [ "$GPU_ID" = "cpu" ]; then
    DEVICE="cpu"
else
    DEVICE="cuda"
    export CUDA_VISIBLE_DEVICES=$GPU_ID
fi

# Python 路径
export PYTHONPATH=/home/wanghaoxiao/temp_lizhiyuan/Qwen3:$PYTHONPATH

# ============================================================================
# 检查依赖
# ============================================================================

echo "============================================================================"
echo "UnchartedLLM Chatbot Server Launcher"
echo "============================================================================"
echo ""

# 显示 GPU 信息
if [ "$DEVICE" = "cuda" ]; then
    echo "GPU 信息:"
    echo "  指定 GPU ID: $GPU_ID"
    echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo ""

    # 检查 nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        echo "可用的 GPU:"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | while IFS=',' read -r idx name total free; do
            echo "  GPU $idx: $name | Total: $total | Free: $free"
        done
        echo ""
    fi
fi

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型文件不存在: $MODEL_PATH"
    echo "请修改脚本中的 MODEL_PATH 变量"
    exit 1
fi

echo "✓ 模型文件: $MODEL_PATH"

# 检查基础模型
if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ 错误: 基础模型目录不存在: $BASE_MODEL"
    echo "请修改脚本中的 BASE_MODEL 变量"
    exit 1
fi

echo "✓ 基础模型: $BASE_MODEL"

# 检查 Python 包
echo ""
echo "检查 Python 依赖..."

python3 -c "import aiohttp" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少依赖: aiohttp"
    echo "请安装: pip install aiohttp aiohttp-cors"
    exit 1
fi

python3 -c "import aiohttp_cors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少依赖: aiohttp-cors"
    echo "请安装: pip install aiohttp-cors"
    exit 1
fi

echo "✓ 所有依赖已安装"

# ============================================================================
# 启动服务器
# ============================================================================

echo ""
echo "============================================================================"
echo "启动 Chatbot 服务器..."
echo "============================================================================"
echo ""
echo "配置信息:"
echo "  模型路径: $MODEL_PATH"
echo "  基础模型: $BASE_MODEL"
echo "  服务器地址: http://$HOST:$PORT"
echo "  设备: $DEVICE"
if [ "$DEVICE" = "cuda" ]; then
    echo "  GPU ID: $GPU_ID"
fi
echo ""
echo "SSH 端口转发命令 (在本地机器上运行):"
echo "  ssh -L $PORT:localhost:$PORT $(whoami)@$(hostname)"
echo ""
echo "然后在本地浏览器打开:"
echo "  http://localhost:$PORT"
echo ""
echo "============================================================================"
echo ""

# 启动服务器
python3 backend.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --device "$DEVICE" \
    --host "$HOST" \
    --port "$PORT"

"""
backend.py - UnchartedLLM Chatbot Backend Server

提供 WebSocket API 用于实时交互式文本生成，支持动态 target 控制。

功能:
1. 加载训练好的 UnchartedLLM 模型
2. 提供 WebSocket 接口用于实时生成
3. 每一步输出: target输入向量、target输出向量、新token、已输出token数
4. 支持动态修改 target 值
5. 计算并保存: 输入target数值、输出target数值、模长、cosine值
6. 生成结束后返回完整历史数据用于可视化
"""

import os
import sys
import json
import asyncio
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# WebSocket server
from aiohttp import web
import aiohttp_cors

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from core import (
    extend_tokenizer_safe,
    initialize_num_embedding,
    batch_value_to_xval_embedding,
    XVAL_ALPHA,
    MAX_TARGET_VALUE
)
from model import UnchartedModelWrapper


@dataclass
class StepData:
    """单步生成数据"""
    step: int
    token_id: int
    token_text: str
    input_target_vector: List[float]  # 输入target向量 (前10维)
    output_target_vector: List[float]  # 输出target向量 (前10维)
    input_target_value: float  # 输入target数值
    output_target_value: float  # 输出target数值
    output_target_norm: float  # 输出target模长
    cosine_similarity: float  # cosine值 (数值/模长)
    total_tokens: int  # 已输出token数
    planned_target_value: float  # 规划的target数值
    control_strategy: str  # 控制策略 ("planned" 或 "predicted")
    ppl: Optional[float] = None  # 困惑度 (Perplexity)


class TargetPlanner:
    """Target规划器 - 根据用户定义的控制点计算每步应该输入的target值"""

    def __init__(self, control_points: List[Tuple[float, float]], initial_target: float):
        """
        初始化规划器

        Args:
            control_points: 控制点列表 [(x1, y1), (x2, y2), ...]
                           x: 已输出token数 / initial_target (归一化)
                           y: 输入target值 / initial_target (归一化)
            initial_target: 初始target值
        """
        self.control_points = sorted(control_points, key=lambda p: p[0])  # 按x排序
        self.initial_target = initial_target

        # 验证控制点
        if not self.control_points:
            raise ValueError("控制点列表不能为空")

        # 确保第一个点是 (0, 1)
        if self.control_points[0] != (0.0, 1.0):
            self.control_points.insert(0, (0.0, 1.0))

        # 确保最后一个点的y值为0
        if self.control_points[-1][1] != 0.0:
            last_x = self.control_points[-1][0]
            self.control_points.append((last_x, 0.0))

        print(f"[TargetPlanner] Initialized with {len(self.control_points)} control points:")
        for i, (x, y) in enumerate(self.control_points):
            print(f"  Point {i}: ({x:.4f}, {y:.4f})")

    def get_planned_target(self, total_tokens: int) -> float:
        """
        根据已输出token数计算规划的target值

        Args:
            total_tokens: 已输出的token数

        Returns:
            planned_target: 规划的target值
        """
        # 归一化x坐标
        x_normalized = total_tokens / self.initial_target

        # 如果超出范围，返回0
        if x_normalized >= self.control_points[-1][0]:
            return 0.0

        # 线性插值
        for i in range(len(self.control_points) - 1):
            x1, y1 = self.control_points[i]
            x2, y2 = self.control_points[i + 1]

            if x1 <= x_normalized <= x2:
                # 线性插值公式
                if x2 == x1:
                    y_normalized = y1
                else:
                    t = (x_normalized - x1) / (x2 - x1)
                    y_normalized = y1 + t * (y2 - y1)

                # 反归一化
                planned_target = y_normalized * self.initial_target
                return max(0.0, planned_target)

        # 默认返回0
        return 0.0


class UnchartedChatbot:
    """UnchartedLLM Chatbot 核心类"""

    def __init__(
        self,
        model_path: str,
        base_model_name: str = "Qwen/Qwen2.5-8B",
        device: str = "cuda"
    ):
        """
        初始化 Chatbot

        Args:
            model_path: 训练好的模型checkpoint路径
            base_model_name: 基础模型名称
            device: 设备
        """
        self.device = device
        self.stop_generation = False  # 停止生成标志
        print(f"[Chatbot] Initializing on {device}...")

        # 加载 tokenizer
        print(f"[Chatbot] Loading tokenizer from {base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )

        # 加载基础模型
        print(f"[Chatbot] Loading base model from {base_model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # 扩展 tokenizer
        print(f"[Chatbot] Extending tokenizer...")
        self.num_token_id = extend_tokenizer_safe(self.tokenizer)

        # 创建 wrapper
        print(f"[Chatbot] Creating UnchartedModelWrapper...")
        self.model = UnchartedModelWrapper(
            base_model,
            self.num_token_id,
            hidden_size=base_model.config.hidden_size
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        initialize_num_embedding(
            self.model,
            self.num_token_id,
            device=device,
            hidden_size=base_model.config.hidden_size
        )

        # 加载训练好的权重
        if model_path and os.path.exists(model_path):
            print(f"[Chatbot] Loading checkpoint from {model_path}...")
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[Chatbot] Checkpoint loaded successfully!")
        else:
            print(f"[Chatbot] Warning: No checkpoint loaded, using initialized model")

        self.model = self.model.to(device)
        self.model.eval()

        # 获取 base_embedding
        embedding_layer = self.model.base_model.get_input_embeddings()
        self.base_embedding = embedding_layer.weight[self.num_token_id]

        print(f"[Chatbot] Initialization complete!")
        print(f"  Device: {device}")
        print(f"  Base embedding norm: {self.base_embedding.norm().item():.4f}")
        print(f"  xVal alpha: {XVAL_ALPHA}")

    def embedding_to_value(self, embedding: torch.Tensor) -> float:
        """
        将 embedding 向量转换为数值

        公式: value = ||embedding|| / (α · ||base_embedding||)

        Args:
            embedding: [hidden_size] 预测的embedding向量

        Returns:
            value: 预测的数值
        """
        pred_norm = embedding.norm(p=2)
        base_norm = self.base_embedding.norm(p=2)
        value = pred_norm / (XVAL_ALPHA * base_norm)
        return value.item()

    def value_to_embedding(self, value: float) -> torch.Tensor:
        """
        将数值转换为 embedding 向量

        公式: embedding = α · value · base_embedding

        Args:
            value: 数值

        Returns:
            embedding: [hidden_size] embedding向量
        """
        scaled_value = XVAL_ALPHA * value
        embedding = scaled_value * self.base_embedding
        return embedding

    def compute_cosine_similarity(self, embedding: torch.Tensor) -> float:
        """
        计算 embedding 与 base_embedding 的 cosine 相似度

        实际上就是: value / norm 的比值

        Args:
            embedding: [hidden_size] 预测的embedding向量

        Returns:
            cosine: cosine相似度
        """
        # 方法1: 直接计算 cosine
        cosine = F.cosine_similarity(
            embedding.unsqueeze(0),
            self.base_embedding.unsqueeze(0),
            dim=1
        )
        return cosine.item()

    def compute_ppl_at_position(
        self,
        input_ids: torch.Tensor,
        window_size: int = 512
    ) -> Optional[float]:
        """
        计算当前位置的困惑度 (Perplexity)

        基于 metrics_1.py 中的 PPL 计算方法

        Args:
            input_ids: [seq_len] 当前的完整输入序列
            window_size: 滑动窗口大小

        Returns:
            ppl: 困惑度值，如果窗口太小则返回 None
        """
        seq_len = input_ids.shape[0]

        # 窗口太小，跳过
        if seq_len < 10:
            return None

        # 提取窗口（取最后 window_size 个 tokens）
        start_pos = max(0, seq_len - window_size)
        window_ids = input_ids[start_pos:].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(window_ids, return_dict=True)
            logits = outputs['text_logits']

            # 计算负对数似然
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = window_ids[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            ppl = torch.exp(loss).item()

        return ppl

    async def generate_step_by_step(
        self,
        prompt: str,
        initial_target: float,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        websocket = None,
        control_points: Optional[List[Tuple[float, float]]] = None,
        control_strategy: str = "none",
        compelling_steps: int = 0,
        ppl_window_size: int = 512
    ) -> List[StepData]:
        """
        逐步生成文本，支持规划控制

        Args:
            prompt: 输入提示
            initial_target: 初始目标长度
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: Top-K采样
            top_p: Top-P采样
            websocket: WebSocket连接 (用于实时发送数据)
            control_points: 控制点列表 [(x1, y1), (x2, y2), ...]
            control_strategy: 控制策略 ("none", "partial", "full")
                - "none": 不使用规划，完全使用predicted_target
                - "partial": 前compelling_steps步使用规划，之后使用predicted_target
                - "full": 全程使用规划
            compelling_steps: 强制控制的步数（仅在strategy="partial"时有效）
            ppl_window_size: PPL计算的滑动窗口大小

        Returns:
            history: 完整的生成历史
        """
        # 编码 prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        # 初始化规划器
        planner = None
        if control_points and control_strategy != "none":
            try:
                planner = TargetPlanner(control_points, initial_target)
                print(f"[Generate] Target planner initialized with strategy: {control_strategy}")
                if control_strategy == "partial":
                    print(f"[Generate] Compelling steps: {compelling_steps}")
            except Exception as e:
                print(f"[Generate] Failed to initialize planner: {e}")
                planner = None

        # 初始化状态
        current_target_value = initial_target
        current_target_embedding = self.value_to_embedding(current_target_value)

        history = []
        generated_tokens = []
        self.stop_generation = False  # 重置停止标志

        print(f"\n[Generate] Starting generation...")
        print(f"  Prompt: {prompt}")
        print(f"  Initial target: {initial_target}")
        print(f"  Max length: {max_length}")
        print(f"  Control strategy: {control_strategy}")

        for step in range(max_length):
            # 检查停止标志
            if self.stop_generation:
                print(f"[Generate] Generation stopped by user at step {step}")
                break

            # 记录输入 target
            input_target_value = current_target_value
            input_target_vector = current_target_embedding.detach().cpu().float().numpy()[:10].tolist()

            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids, return_dict=True)
                text_logits = outputs['text_logits'][:, -1, :]  # [1, vocab_size]
                predicted_embedding = outputs['predicted_embeddings'][0, -1]  # [hidden_size]

            # 采样下一个 token
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

            # 解码 token
            token_id = next_token_id.item()
            token_text = self.tokenizer.decode([token_id])
            generated_tokens.append(token_id)

            # 计算输出 target 的各项指标
            output_target_value = self.embedding_to_value(predicted_embedding)
            output_target_norm = predicted_embedding.norm(p=2).item()
            cosine_similarity = self.compute_cosine_similarity(predicted_embedding)
            output_target_vector = predicted_embedding.detach().cpu().float().numpy()[:10].tolist()

            # 计算规划的target值
            planned_target_value = 0.0
            if planner:
                planned_target_value = planner.get_planned_target(len(generated_tokens))

            # 确定当前使用的控制策略
            current_strategy = "predicted"
            if planner:
                if control_strategy == "full":
                    current_strategy = "planned"
                elif control_strategy == "partial" and step < compelling_steps:
                    current_strategy = "planned"

            # 计算 PPL (困惑度)
            ppl = None
            if len(generated_tokens) > 0:
                # 构建完整的输入序列（prompt + 已生成的tokens）
                full_input_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0]
                full_input_ids = torch.cat([
                    full_input_ids,
                    torch.tensor(generated_tokens, device=self.device)
                ])
                ppl = self.compute_ppl_at_position(full_input_ids, window_size=ppl_window_size)

            # 创建步骤数据
            step_data = StepData(
                step=step,
                token_id=token_id,
                token_text=token_text,
                input_target_vector=input_target_vector,
                output_target_vector=output_target_vector,
                input_target_value=input_target_value,
                output_target_value=output_target_value,
                output_target_norm=output_target_norm,
                cosine_similarity=cosine_similarity,
                total_tokens=len(generated_tokens),
                planned_target_value=planned_target_value,
                control_strategy=current_strategy,
                ppl=ppl
            )

            history.append(step_data)

            # 实时发送数据到前端
            if websocket:
                try:
                    # 检查WebSocket状态，避免向关闭的连接发送数据
                    if not websocket.closed:
                        await websocket.send_json({
                            'type': 'step',
                            'data': asdict(step_data)
                        })
                        # 让出控制权，避免阻塞事件循环
                        await asyncio.sleep(0)
                    else:
                        print(f"[Generate] WebSocket closed, stopping generation")
                        break
                except Exception as e:
                    print(f"[Generate] Error sending step data: {e}")
                    break  # 发送失败时停止生成

            # 打印进度
            if step % 10 == 0:
                ppl_str = f"ppl={ppl:.2f}" if ppl is not None else "ppl=N/A"
                print(f"[Generate] Step {step}: token='{token_text}', "
                      f"input_target={input_target_value:.2f}, "
                      f"output_target={output_target_value:.2f}, "
                      f"planned_target={planned_target_value:.2f}, "
                      f"strategy={current_strategy}, "
                      f"cosine={cosine_similarity:.4f}, "
                      f"{ppl_str}")

            # 更新 input_ids
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            # 根据控制策略更新下一步的 target
            if current_strategy == "planned":
                # 使用规划的target
                next_target_value = planner.get_planned_target(len(generated_tokens))
            else:
                # 使用predicted_target - 1
                next_target_value = max(0, output_target_value - 1)

            current_target_value = next_target_value
            current_target_embedding = self.value_to_embedding(current_target_value)

            # 检查终止条件
            if token_id == self.tokenizer.eos_token_id:
                print(f"[Generate] EOS token generated at step {step}")
                break

        print(f"[Generate] Generation complete! Total tokens: {len(generated_tokens)}")

        return history


# ============================================================================
# WebSocket Server
# ============================================================================

class ChatbotServer:
    """Chatbot WebSocket 服务器"""

    def __init__(self, chatbot: UnchartedChatbot, host: str = '0.0.0.0', port: int = 8765):
        self.chatbot = chatbot
        self.host = host
        self.port = port
        self.app = web.Application()
        self.current_websocket = None
        self.is_generating = False  # 生成状态标志

        # 设置路由
        self.app.router.add_get('/ws', self.websocket_handler)
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/', self.index_handler)

        # 可选：如果有 static 目录，则添加静态文件路由
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        if os.path.exists(static_dir):
            self.app.router.add_static('/static', static_dir)

        # 配置 CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })

        for route in list(self.app.router.routes()):
            cors.add(route)

    async def index_handler(self, request):
        """返回前端 HTML 页面"""
        html_path = os.path.join(os.path.dirname(__file__), 'frontend.html')
        if os.path.exists(html_path):
            return web.FileResponse(html_path)
        else:
            return web.Response(text="Frontend not found. Please create frontend.html", status=404)

    async def health_check(self, request):
        """健康检查"""
        return web.json_response({'status': 'ok', 'message': 'Chatbot server is running'})

    async def websocket_handler(self, request):
        """WebSocket 连接处理"""
        ws = web.WebSocketResponse(
            heartbeat=30.0,  # 30秒心跳
            timeout=300.0    # 5分钟超时
        )
        await ws.prepare(request)

        self.current_websocket = ws

        print(f"[Server] WebSocket connected from {request.remote}")

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    if data['type'] == 'generate':
                        # 开始生成
                        if self.is_generating:
                            print(f"[Server] Warning: Generation already in progress, rejecting new request")
                            await ws.send_json({
                                'type': 'error',
                                'message': 'Generation already in progress'
                            })
                            continue

                        prompt = data.get('prompt', 'Hello, world!')
                        initial_target = data.get('initial_target', 100)
                        max_length = data.get('max_length', 512)
                        temperature = data.get('temperature', 1.0)
                        top_k = data.get('top_k', 50)
                        top_p = data.get('top_p', 0.9)

                        # 新增：控制参数
                        control_points = data.get('control_points', None)
                        control_strategy = data.get('control_strategy', 'none')
                        compelling_steps = data.get('compelling_steps', 0)
                        ppl_window_size = data.get('ppl_window_size', 512)

                        print(f"[Server] Received generate request:")
                        print(f"  Prompt: {prompt}")
                        print(f"  Initial target: {initial_target}")
                        print(f"  Control strategy: {control_strategy}")
                        print(f"  PPL window size: {ppl_window_size}")
                        if control_points:
                            print(f"  Control points: {control_points}")
                        if control_strategy == 'partial':
                            print(f"  Compelling steps: {compelling_steps}")

                        # 发送确认消息
                        if not ws.closed:
                            await ws.send_json({
                                'type': 'generate_started',
                                'message': 'Generation started'
                            })

                        # 异步生成
                        self.is_generating = True
                        try:
                            history = await self.chatbot.generate_step_by_step(
                                prompt=prompt,
                                initial_target=initial_target,
                                max_length=max_length,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                websocket=ws,
                                control_points=control_points,
                                control_strategy=control_strategy,
                                compelling_steps=compelling_steps,
                                ppl_window_size=ppl_window_size
                            )

                            # 发送完成信号和完整历史
                            if not ws.closed:
                                await ws.send_json({
                                    'type': 'complete',
                                    'history': [asdict(step) for step in history]
                                })
                        except Exception as e:
                            print(f"[Server] Error during generation: {e}")
                            import traceback
                            traceback.print_exc()
                            # 只在连接未关闭时发送错误消息
                            if not ws.closed:
                                try:
                                    await ws.send_json({
                                        'type': 'error',
                                        'message': str(e)
                                    })
                                except:
                                    pass  # 如果发送失败，忽略
                        finally:
                            self.is_generating = False
                            print(f"[Server] Generation finished, is_generating reset to False")

                    elif data['type'] == 'stop':
                        # 停止生成
                        print(f"[Server] Received stop request")
                        self.chatbot.stop_generation = True
                        if not ws.closed:
                            await ws.send_json({
                                'type': 'stopped',
                                'message': 'Generation stopped'
                            })

                elif msg.type == web.WSMsgType.ERROR:
                    print(f"[Server] WebSocket error: {ws.exception()}")

        finally:
            print(f"[Server] WebSocket disconnected")
            # 确保在连接断开时重置生成状态
            if self.is_generating:
                print(f"[Server] Warning: Connection closed while generating, resetting is_generating flag")
                self.is_generating = False
                self.chatbot.stop_generation = True
            self.current_websocket = None

        return ws

    def run(self):
        """启动服务器"""
        print(f"\n{'='*80}")
        print(f"UnchartedLLM Chatbot Server")
        print(f"{'='*80}")
        print(f"Server running on http://{self.host}:{self.port}")
        print(f"WebSocket endpoint: ws://{self.host}:{self.port}/ws")
        print(f"Health check: http://{self.host}:{self.port}/health")
        print(f"\nSSH Port Forwarding:")
        print(f"  ssh -L {self.port}:localhost:{self.port} user@remote_host")
        print(f"  Then open: http://localhost:{self.port}")
        print(f"{'='*80}\n")
        print(f"Press Ctrl+C to stop the server\n")

        # 使用 print_exception=False 来避免显示大量错误信息
        # 使用 access_log=None 来减少日志输出
        try:
            web.run_app(
                self.app,
                host=self.host,
                port=self.port,
                print=lambda x: None,  # 禁用启动信息（我们已经打印了）
                access_log=None  # 禁用访问日志
            )
        except KeyboardInterrupt:
            print("\n[Server] Shutting down gracefully...")
            print("[Server] Server stopped.")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="UnchartedLLM Chatbot Server")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-8B',
                        help='Base model name')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host')
    parser.add_argument('--port', type=int, default=8765,
                        help='Server port')

    args = parser.parse_args()

    # 初始化 chatbot
    chatbot = UnchartedChatbot(
        model_path=args.model_path,
        base_model_name=args.base_model,
        device=args.device
    )

    # 启动服务器
    server = ChatbotServer(chatbot, host=args.host, port=args.port)
    server.run()


if __name__ == '__main__':
    main()

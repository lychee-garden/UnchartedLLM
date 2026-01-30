"""
FrictionHybridDataset - Unified Version
Supports both v1 (Soft Attention) and v2 (Numerical Projector) approaches
Handles mixed long (arxiv) and short (fineweb-edu) text sources with proper target calculation.
"""

import re
import random
import os
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from typing import Dict, Optional, Iterator, List
from transformers import PreTrainedTokenizer
import ssl
import urllib3

# 尝试相对导入，失败则使用绝对导入
try:
    from .core import sample_noisy_target, compute_alpha_from_mu
except ImportError:
    from core import sample_noisy_target, compute_alpha_from_mu


class FrictionHybridDataset(IterableDataset):
    """
    Unified hybrid dataset for training with remaining token countdown.
    Supports both v1 (token-based) and v2 (continuous value) approaches.

    Features:
    - Mixes arxiv (long) and fineweb-edu (short) sources
    - Aggressive cleaning for arxiv (removes references/appendices)
    - 32K length guard (no truncation, discard if too long)
    - Random mid-cut strategy with global target anchoring
    - Streaming support for large datasets
    """

    # Regex patterns for fineweb cleaning
    FINEWEB_CLEANING_PATTERNS = [
        # Common web page elements
        r'(?i)cookie\s+policy.*',
        r'(?i)privacy\s+policy.*',
        r'(?i)terms\s+of\s+service.*',
        r'(?i)subscribe\s+to\s+our\s+newsletter.*',
        r'(?i)sign\s+up\s+for.*',
        r'(?i)follow\s+us\s+on.*',

        # Navigation and UI elements
        r'(?i)home\s*\|\s*about\s*\|\s*contact.*',
        r'(?i)copyright\s+©.*',
        r'(?i)all\s+rights\s+reserved.*',

        # Ads and promotional content
        r'(?i)advertisement.*',
        r'(?i)sponsored\s+content.*',
        r'(?i)click\s+here.*',

        # Social media
        r'(?i)share\s+on\s+(facebook|twitter|linkedin).*',
        r'(?i)tweet\s+this.*',

        # Comments section
        r'(?i)leave\s+a\s+comment.*',
        r'(?i)post\s+a\s+comment.*',
        r'(?i)\d+\s+comments?.*',
    ]

    # Regex patterns for arxiv cleaning (保留但不使用)
    ARXIV_CLEANING_PATTERNS = [
        # References section (various formats)
        r'(?i)\n\s*references\s*\n.*',
        r'(?i)\n\s*bibliography\s*\n.*',
        r'(?i)\n\s*\[\s*\d+\s*\].*',  # Numbered references like [1], [2]

        # Appendix sections
        r'(?i)\n\s*appendix\s*[a-z]?\s*\n.*',
        r'(?i)\n\s*supplementary\s+materials?\s*\n.*',

        # Acknowledgments (often at end)
        r'(?i)\n\s*acknowledgments?\s*\n.*',
        r'(?i)\n\s*funding\s*\n.*',

        # Common arxiv metadata
        r'(?i)arxiv:\d+\.\d+.*',
        r'(?i)preprint.*',
    ]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        arxiv_dataset_name: str = "togethercomputer/RedPajama-Data-1T",  # 使用 RedPajama
        fineweb_dataset_name: str = "HuggingFaceFW/fineweb-edu",
        fineweb_split: str = "sample-10BT",
        arxiv_prob: float = 0.5,
        max_tokens: int = 32768,
        max_chars_prefilter: int = 130000,
        window_size: int = 1024,  # 训练窗口大小
        stride: int = 1024,  # 滑动窗口步长（无重叠）
        ending_window_size: int = 1024,  # 结尾专用窗口大小
        ending_repeat: int = 2,  # 结尾样本重复次数
        min_seq_len: int = 512,
        seed: Optional[int] = None,
        version: str = "v1",  # "v1" or "v2"
        use_local: bool = False,  # 是否使用本地数据集
        local_dir: Optional[str] = None,  # 本地数据集目录
        arxiv_num_files: int = 10  # RedPajama arxiv 使用的文件数量
    ):
        """
        Initialize FrictionHybridDataset.

        Args:
            tokenizer: HuggingFace tokenizer
            arxiv_dataset_name: Arxiv dataset identifier (default: RedPajama-Data-1T)
            fineweb_dataset_name: FineWeb-Edu dataset identifier
            fineweb_split: FineWeb split to use
            arxiv_prob: Probability of sampling from arxiv (0.0-1.0)
            max_tokens: Maximum token length (32768)
            max_chars_prefilter: Character-level prefilter threshold
            window_size: Size of sliding window (default: 1024)
            stride: Stride for sliding window (default: 1024, no overlap)
            ending_window_size: Size of ending window for target~0 training (default: 1024)
            ending_repeat: Number of times to repeat ending sample (default: 2)
            min_seq_len: Minimum sequence length
            seed: Random seed for reproducibility
            version: "v1" for soft attention, "v2" for numerical projector
            use_local: Whether to use local datasets (default: False, streaming)
            local_dir: Local datasets directory (default: None)
            arxiv_num_files: Number of RedPajama arxiv files to use (default: 10)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.arxiv_prob = arxiv_prob
        self.max_tokens = max_tokens
        self.max_chars_prefilter = max_chars_prefilter
        self.window_size = window_size
        self.stride = stride  # 滑动窗口步长
        self.ending_window_size = ending_window_size  # 结尾窗口大小
        self.ending_repeat = ending_repeat  # 结尾样本重复次数
        self.min_seq_len = min_seq_len
        self.version = version
        self.use_local = use_local
        self.local_dir = local_dir
        self.arxiv_num_files = arxiv_num_files
        self.arxiv_dataset_name = arxiv_dataset_name

        if seed is not None:
            random.seed(seed)

        # Compile cleaning patterns
        self.arxiv_cleaning_patterns = [re.compile(pattern, re.DOTALL) for pattern in self.ARXIV_CLEANING_PATTERNS]
        self.fineweb_cleaning_patterns = [re.compile(pattern, re.DOTALL) for pattern in self.FINEWEB_CLEANING_PATTERNS]

        print(f"[Dataset] Initializing FrictionHybridDataset ({version})")
        print(f"  Mode: {'Local' if use_local else 'Streaming'}")
        if use_local:
            print(f"  Local dir: {local_dir}")
        print(f"  Arxiv probability: {arxiv_prob}")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Window size: {window_size}")
        print(f"  Stride: {stride} ({'no overlap' if stride == window_size else f'{100*(1-stride/window_size):.0f}% overlap'})")
        print(f"  Ending window size: {ending_window_size}")
        print(f"  Ending repeat: {ending_repeat}x")  # 显示重复次数
        print(f"  Max tokens: {max_tokens}")
        print(f"  Window size: {window_size}")
        print(f"  Stride: {stride}")  # 新增：显示步长

        # Configure SSL and retry settings for HuggingFace downloads
        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        if hf_endpoint != "https://huggingface.co":
            print(f"  Using HuggingFace endpoint: {hf_endpoint}")

        # Disable SSL warnings (optional, for debugging)
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Configure load_dataset parameters
        if use_local and local_dir:
            # 本地模式：从本地缓存加载
            print(f"  Loading from local cache...")
            # 检查是否设置了离线模式
            hf_offline = os.environ.get("HF_OFFLINE", "0")
            if hf_offline == "1":
                print(f"  [Info] HF_OFFLINE=1 detected, will use local cache only")
            
            # 在本地模式下，使用 DownloadConfig 禁用网络访问
            try:
                from datasets import DownloadConfig
                try:
                    from datasets import DownloadMode
                    # 尝试使用枚举值
                    download_mode = DownloadMode.REUSE_DATASET_IF_EXISTS
                except (ImportError, AttributeError):
                    # 如果枚举不存在，使用字符串
                    download_mode = "reuse_dataset_if_exists"
                
                download_config = DownloadConfig(
                    cache_dir=local_dir,
                    download_mode=download_mode,  # 只使用已存在的缓存
                    max_retries=0,  # 禁用重试，避免网络访问
                )
                load_kwargs = {
                    "streaming": True,
                    "cache_dir": local_dir,
                    "download_config": download_config,
                }
            except Exception as e:
                # 如果导入失败，使用基本配置
                print(f"  [Warning] Failed to configure DownloadConfig: {e}")
                load_kwargs = {
                    "streaming": True,
                    "cache_dir": local_dir,
                }
        else:
            # 流式模式：边下边训练（默认）
            print(f"  Using streaming mode (download on-the-fly)...")
            load_kwargs = {
                "streaming": True,
                # Note: num_proc is not supported in streaming mode
            }

            # Add download config for better error handling
            try:
                from datasets import DownloadConfig
                download_config = DownloadConfig(
                    max_retries=10,  # Increase retries for network issues
                )
                load_kwargs["download_config"] = download_config
            except Exception:
                pass

        # Load datasets
        # Load RedPajama arxiv dataset
        try:
            print(f"  Loading RedPajama arxiv dataset...")

            if use_local and local_dir:
                # 本地模式：从本地 JSONL 文件加载
                arxiv_cache = os.path.join(local_dir, "arxiv")
                if os.path.exists(arxiv_cache):
                    # 查找本地 JSONL 文件
                    import glob
                    local_files = glob.glob(os.path.join(arxiv_cache, "*.jsonl"))
                    if local_files:
                        print(f"  Found {len(local_files)} local arxiv files")
                        # 限制文件数量
                        local_files = local_files[:arxiv_num_files]
                        self.arxiv_dataset = load_dataset(
                            "json",
                            data_files=local_files,
                            split="train",
                            streaming=True
                        )
                        print(f"  ✓ Loaded {len(local_files)} local arxiv files")
                    else:
                        print(f"  No local arxiv files found, falling back to streaming...")
                        self.arxiv_dataset = self._load_redpajama_arxiv_streaming(arxiv_num_files)
                else:
                    print(f"  Local arxiv cache not found, using streaming...")
                    self.arxiv_dataset = self._load_redpajama_arxiv_streaming(arxiv_num_files)
            else:
                # 流式模式：从 RedPajama URL 加载
                self.arxiv_dataset = self._load_redpajama_arxiv_streaming(arxiv_num_files)

            print(f"  ✓ Loaded arxiv dataset: {arxiv_dataset_name}")
        except Exception as e:
            print(f"  ✗ Failed to load arxiv: {e}")
            print(f"     Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            self.arxiv_dataset = None

        try:
            if use_local and local_dir:
                fineweb_cache = os.path.join(local_dir, "fineweb")
                # 在本地模式下，使用配置好的 load_kwargs（包含 download_config）
                self.fineweb_dataset = load_dataset(
                    fineweb_dataset_name,
                    name=fineweb_split,
                    split="train",
                    cache_dir=fineweb_cache,
                    streaming=True,
                    download_config=load_kwargs.get("download_config", None)
                )
            else:
                self.fineweb_dataset = load_dataset(
                    fineweb_dataset_name,
                    name=fineweb_split,
                    split="train",
                    **load_kwargs
                )
            print(f"  ✓ Loaded fineweb dataset: {fineweb_dataset_name}/{fineweb_split}")
        except Exception as e:
            print(f"  ✗ Failed to load fineweb: {e}")
            print(f"     Error type: {type(e).__name__}")
            if "SSL" in str(e) or "SSLError" in str(type(e).__name__):
                print(f"     SSL Error detected. Try one of these solutions:")
                print(f"     1. Set environment variable: export HF_ENDPOINT=https://hf-mirror.com")
                print(f"     2. Use VPN or proxy")
                print(f"     3. Retry later (may be temporary network issue)")
                print(f"     4. Pre-download datasets: python download_datasets.py")
            self.fineweb_dataset = None

        if self.arxiv_dataset is None and self.fineweb_dataset is None:
            raise ValueError("Both datasets failed to load!")

    def _load_redpajama_arxiv_streaming(self, num_files: int = 10):
        """
        Load RedPajama arxiv dataset from streaming URLs.

        Args:
            num_files: Number of arxiv JSONL files to load

        Returns:
            Streaming dataset
        """
        from huggingface_hub import hf_hub_download

        try:
            # 下载 URL 列表
            url_file = hf_hub_download(
                repo_id="togethercomputer/RedPajama-Data-1T",
                filename="urls/arxiv.txt",
                repo_type="dataset"
            )

            # 读取 URL
            with open(url_file, 'r') as f:
                urls = [line.strip() for line in f.readlines()]

            # 限制文件数量
            urls = urls[:num_files]
            print(f"  Loading {len(urls)} RedPajama arxiv files...")

            # 使用 json 格式加载多个文件
            dataset = load_dataset(
                "json",
                data_files=urls,
                split="train",
                streaming=True
            )

            return dataset

        except Exception as e:
            print(f"  Failed to load RedPajama arxiv URLs: {e}")
            raise

    def clean_arxiv_text(self, text: str) -> str:
        """
        清洗 arxiv 文本（仅用于对比展示，实际训练不使用）

        Args:
            text: Raw arxiv text

        Returns:
            清洗后的文本
        """
        # 应用所有清洗规则
        for pattern in self.arxiv_cleaning_patterns:
            text = pattern.sub('', text)

        # 移除多余空白
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = text.strip()

        return text

    def clean_fineweb_text(self, text: str) -> str:
        """
        清洗 fineweb 文本（仅用于对比展示，实际训练不使用）

        Args:
            text: Raw fineweb text

        Returns:
            清洗后的文本
        """
        # 应用所有清洗规则
        for pattern in self.fineweb_cleaning_patterns:
            text = pattern.sub('', text)

        # 移除多余空白
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = text.strip()

        return text

    def prefilter_by_length(self, text: str) -> bool:
        """
        First defense: character-level filtering.

        Args:
            text: Input text

        Returns:
            True if text passes filter, False if too long
        """
        return len(text) <= self.max_chars_prefilter

    def tokenize_and_validate(self, text: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Tokenize text and validate length.

        Second defense: token-level filtering.
        CRITICAL: Do NOT truncate. Discard if too long.

        Args:
            text: Input text

        Returns:
            Tokenized dict if valid, None if too long
        """
        # Tokenize without truncation
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True
        )

        # Check token length
        token_length = encoded['input_ids'].size(1)

        if token_length > self.max_tokens:
            return None  # Discard if too long

        if token_length < self.min_seq_len:
            return None  # Discard if too short

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'length': token_length
        }

    def apply_sliding_window_cut(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        total_length: int,
        is_arxiv: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """
        滑动窗口切片策略（无重叠）

        策略：
        - Arxiv: 正常滑动窗口 + 结尾专用样本（重复训练2次）
        - FineWeb: 正常滑动窗口，最后20个token不切片，但保留对target的贡献

        Arxiv 示例（total_length=10000, window_size=1024）：
        - 样本1: [0:1024]       target: [10000, 9999, ..., 8976]
        - 样本2: [1024:2048]    target: [8976, 8975, ..., 7952]
        - ...
        - 样本9: [8192:8976]    target: [1808, 1807, ..., 1024]  # 最后一个正常样本
        - 样本10: [8976:10000]  target: [1024, 1023, ..., 0]     # 结尾样本（第1次）⭐
        - 样本11: [8976:10000]  target: [1024, 1023, ..., 0]     # 结尾样本（第2次）⭐

        FineWeb 示例（total_length=2000, window_size=1024）：
        - 样本1: [0:1024]       target: [2000, 1999, ..., 976]   # target从2000开始
        - 样本2: [1024:1980]    target: [976, 975, ..., 20]      # 只切到1980，最后20个token不切片

        Args:
            input_ids: Full sequence token IDs [total_length]
            attention_mask: Full attention mask [total_length]
            total_length: Total length of sequence
            is_arxiv: Whether this is arxiv data (True) or fineweb (False)

        Returns:
            List of sample dictionaries
        """
        samples = []
        alpha = compute_alpha_from_mu(0.5)

        # FineWeb 特殊处理：最后20个token不切片
        fineweb_exclude_tokens = 20 if not is_arxiv else 0
        effective_length = total_length - fineweb_exclude_tokens

        # 辅助函数：创建单个样本
        def create_sample(start: int, end: int, is_ending: bool) -> dict:
            window_input_ids = input_ids[start:end]
            window_attention_mask = attention_mask[start:end]
            window_length = end - start

            true_target = torch.tensor(
                [total_length - (start + i) for i in range(window_length)],
                dtype=torch.float32
            )
            noisy_target = sample_noisy_target(true_target, alpha=alpha)

            return {
                'input_ids': window_input_ids,
                'attention_mask': window_attention_mask,
                'labels': window_input_ids.clone(),
                'true_target': true_target,
                'noisy_target': noisy_target,
                'start_pos': start,
                'total_length': total_length,
                'is_ending': is_ending
            }

        # 短文本处理
        if effective_length <= self.window_size:
            samples.append(create_sample(0, effective_length, is_ending=True))
            return samples

        # 长文本处理：正常滑动窗口
        if is_arxiv:
            # Arxiv: 切片到 ending_start
            ending_start = max(0, total_length - self.ending_window_size)
            start_pos = 0
            while start_pos + self.window_size <= ending_start:
                samples.append(create_sample(start_pos, start_pos + self.window_size, is_ending=False))
                start_pos += self.stride

            # Arxiv: 添加结尾样本（重复 ending_repeat 次）
            if total_length > self.ending_window_size:
                ending_sample = create_sample(ending_start, total_length, is_ending=True)
                for _ in range(self.ending_repeat):
                    sample_copy = ending_sample.copy()
                    sample_copy['noisy_target'] = sample_noisy_target(ending_sample['true_target'], alpha=alpha)
                    samples.append(sample_copy)
        else:
            # FineWeb: 切片到 effective_length，无结尾重复
            start_pos = 0
            while start_pos < effective_length:
                end_pos = min(start_pos + self.window_size, effective_length)
                is_last = (end_pos >= effective_length)
                samples.append(create_sample(start_pos, end_pos, is_ending=is_last))
                start_pos += self.stride

        return samples

    def process_sample(self, text: str, is_arxiv: bool) -> Optional[List[Dict[str, torch.Tensor]]]:
        """
        Process a single sample through the full pipeline.

        注意：现在返回样本列表（而不是单个样本），因为长文本会被切成多个样本

        Args:
            text: Raw text
            is_arxiv: Whether this is from arxiv (needs cleaning)

        Returns:
            List of processed sample dicts, or None if filtered out
        """
        # Step 1: 清洗（已禁用，直接跳过）
        # if is_arxiv:
        #     text = self.clean_arxiv_text(text)

        # Step 2: First defense - character-level filter
        if not self.prefilter_by_length(text):
            return None

        # Step 3: Tokenize and validate
        tokenized = self.tokenize_and_validate(text)
        if tokenized is None:
            return None

        # Step 4: Apply sliding window cut (返回多个样本)
        samples = self.apply_sliding_window_cut(
            tokenized['input_ids'],
            tokenized['attention_mask'],
            tokenized['length'],
            is_arxiv=is_arxiv  # 传递 is_arxiv 参数
        )

        return samples

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over mixed dataset samples.

        注意：现在每篇论文可能产生多个样本（滑动窗口切片）

        Yields:
            Processed sample dictionaries
        """
        # Create iterators
        arxiv_iter = iter(self.arxiv_dataset) if self.arxiv_dataset else None
        fineweb_iter = iter(self.fineweb_dataset) if self.fineweb_dataset else None

        while True:
            # Decide which source to sample from
            use_arxiv = (
                arxiv_iter is not None and
                (fineweb_iter is None or random.random() < self.arxiv_prob)
            )

            try:
                if use_arxiv:
                    # Sample from arxiv (RedPajama)
                    sample = next(arxiv_iter)
                    # RedPajama 使用 'text' 字段
                    text = sample.get('text', '')
                    if not text:
                        continue

                    processed_samples = self.process_sample(text, is_arxiv=True)

                else:
                    # Sample from fineweb
                    sample = next(fineweb_iter)
                    text = sample.get('text', '')
                    if not text:
                        continue

                    processed_samples = self.process_sample(text, is_arxiv=False)

                # Yield all samples from this text
                if processed_samples is not None:
                    for processed in processed_samples:
                        yield processed

            except StopIteration:
                # Restart iterator if one runs out
                if use_arxiv and arxiv_iter is not None:
                    arxiv_iter = iter(self.arxiv_dataset)
                elif not use_arxiv and fineweb_iter is not None:
                    fineweb_iter = iter(self.fineweb_dataset)
            except Exception as e:
                # Skip problematic samples
                print(f"[Dataset] Skipping sample due to error: {e}")
                continue


def collate_friction_batch(batch):
    """
    Collate function for FrictionHybridDataset.

    Handles variable-length sequences with proper padding.
    Works for both v1 and v2 (both use continuous target values).

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    # Find max length in batch
    max_len = max(sample['input_ids'].size(0) for sample in batch)

    batch_size = len(batch)

    # Initialize batch tensors
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    true_targets = torch.zeros((batch_size, max_len), dtype=torch.float32)
    targets = torch.zeros((batch_size, max_len), dtype=torch.float32)

    # Fill batch
    for i, sample in enumerate(batch):
        seq_len = sample['input_ids'].size(0)
        input_ids[i, :seq_len] = sample['input_ids']
        attention_mask[i, :seq_len] = sample['attention_mask']
        labels[i, :seq_len] = sample['labels']
        true_targets[i, :seq_len] = sample['true_target']

        # Use noisy_target for training (with noise sampling)
        targets[i, :seq_len] = sample['noisy_target']

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'targets': targets,           # Noisy continuous values
        'true_targets': true_targets  # True continuous values
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import argparse

    parser = argparse.ArgumentParser(description="Test FrictionHybridDataset and sample data")
    parser.add_argument('--sample_arxiv', action='store_true',
                        help='Sample arxiv texts and save to txt files')
    parser.add_argument('--sample_fineweb', action='store_true',
                        help='Sample fineweb texts and save to txt files')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for sampled texts (default: current directory)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to extract (default: 5)')
    parser.add_argument('--local_dir', type=str, default='./datasets',
                        help='Local datasets directory (default: ./datasets)')
    parser.add_argument('--use_streaming', action='store_true',
                        help='Force streaming mode (download on-the-fly)')
    parser.add_argument('--arxiv_num_files', type=int, default=10,
                        help='Number of RedPajama arxiv files to use (default: 10)')
    args = parser.parse_args()

    print("=" * 80)
    print("Testing FrictionHybridDataset")
    print("=" * 80)

    # 检查本地数据集是否存在
    use_local = False
    local_dir = args.local_dir

    if not args.use_streaming:
        # 检查本地目录是否存在
        arxiv_cache = os.path.join(local_dir, "arxiv")
        fineweb_cache = os.path.join(local_dir, "fineweb")

        if os.path.exists(arxiv_cache) or os.path.exists(fineweb_cache):
            print(f"\n✓ 检测到本地数据集: {local_dir}")
            print(f"  Arxiv cache: {arxiv_cache} {'(存在)' if os.path.exists(arxiv_cache) else '(不存在)'}")
            print(f"  Fineweb cache: {fineweb_cache} {'(存在)' if os.path.exists(fineweb_cache) else '(不存在)'}")
            use_local = True
        else:
            print(f"\n⚠ 未检测到本地数据集: {local_dir}")
            print(f"  将使用流式模式（边下边用）")
            print(f"  如需使用本地数据集，请先运行: python download_datasets.py")
    else:
        print(f"\n⚠ 强制使用流式模式（--use_streaming）")

    # Initialize tokenizer
    print(f"\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

    # Create dataset
    print(f"\n初始化数据集...")

    # 根据抽样类型设置 arxiv_prob
    if args.sample_arxiv:
        arxiv_prob = 1.0  # 只使用 arxiv
    elif args.sample_fineweb:
        arxiv_prob = 0.0  # 只使用 fineweb
    else:
        arxiv_prob = 0.5  # 混合使用

    dataset = FrictionHybridDataset(
        tokenizer=tokenizer,
        arxiv_prob=arxiv_prob,
        window_size=512,
        min_seq_len=128,
        version="v2",
        use_local=use_local,
        local_dir=local_dir if use_local else None,
        arxiv_num_files=args.arxiv_num_files
    )

    if args.sample_arxiv:
        # 抽查模式：保存 arxiv 数据（清洗前后对比）
        print(f"\n{'='*80}")
        print(f"Sampling {args.num_samples} arxiv texts...")
        print(f"Output directory: {args.output_dir}")
        print(f"{'='*80}\n")

        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)

        # 直接从arxiv数据集抽取原始文本
        arxiv_iter = iter(dataset.arxiv_dataset)
        sampled_count = 0
        attempt_count = 0
        max_attempts = args.num_samples * 10

        while sampled_count < args.num_samples and attempt_count < max_attempts:
            try:
                attempt_count += 1
                sample = next(arxiv_iter)

                # 提取原始文本 (RedPajama 使用 'text' 字段)
                raw_text = sample.get('text', '')
                if not raw_text or len(raw_text) < 100:
                    continue

                # 提取元数据
                meta = sample.get('meta', {})
                arxiv_id = meta.get('arxiv_id', 'unknown') if isinstance(meta, dict) else 'unknown'

                # 清洗文本（仅用于对比）
                cleaned_text = dataset.clean_arxiv_text(raw_text)

                # 保存到文件
                output_file = os.path.join(args.output_dir, f'arxiv_sample_{sampled_count+1}.txt')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write(f"RedPajama Arxiv Sample {sampled_count+1}\n")
                    f.write(f"Arxiv ID: {arxiv_id}\n")
                    f.write("="*80 + "\n\n")

                    f.write("--- RAW TEXT (FULL) ---\n")
                    f.write(raw_text + "\n\n")

                    f.write("--- CLEANED TEXT (FULL) ---\n")
                    f.write(cleaned_text + "\n\n")

                    f.write("--- STATISTICS ---\n")
                    f.write(f"Arxiv ID: {arxiv_id}\n")
                    f.write(f"Raw length (chars): {len(raw_text)}\n")
                    f.write(f"Cleaned length (chars): {len(cleaned_text)}\n")
                    f.write(f"Reduction: {len(raw_text) - len(cleaned_text)} chars ({100*(1-len(cleaned_text)/len(raw_text)):.1f}%)\n")

                    # Tokenize
                    try:
                        tokens = tokenizer.encode(raw_text)
                        f.write(f"Raw token count: {len(tokens)}\n")
                        tokens_cleaned = tokenizer.encode(cleaned_text)
                        f.write(f"Cleaned token count: {len(tokens_cleaned)}\n")
                    except:
                        f.write(f"Token count: (failed to tokenize)\n")

                    # 元数据
                    if isinstance(meta, dict):
                        f.write(f"\n--- METADATA ---\n")
                        for key, value in meta.items():
                            f.write(f"{key}: {value}\n")

                    f.write(f"\n--- NOTE ---\n")
                    f.write(f"实际训练使用: RAW TEXT (完整原文，不清洗)\n")
                    f.write(f"CLEANED TEXT 仅用于对比展示\n")

                print(f"✓ Saved sample {sampled_count+1} to {output_file}")
                print(f"  Arxiv ID: {arxiv_id}")
                print(f"  Raw length: {len(raw_text)} chars")
                print(f"  Cleaned length: {len(cleaned_text)} chars")
                print(f"  Reduction: {100*(1-len(cleaned_text)/len(raw_text)):.1f}%")

                sampled_count += 1

            except StopIteration:
                print(f"✗ Reached end of arxiv dataset after {attempt_count} attempts")
                break
            except Exception as e:
                print(f"✗ Error processing sample: {e}")
                continue

        print(f"\n{'='*80}")
        print(f"Arxiv sampling completed!")
        print(f"  Successfully sampled: {sampled_count}/{args.num_samples}")
        print(f"  Total attempts: {attempt_count}")
        print(f"  Output directory: {args.output_dir}")
        print(f"{'='*80}\n")

    elif args.sample_fineweb:
        # 抽查模式：保存 fineweb 数据（清洗前后对比）
        print(f"\n{'='*80}")
        print(f"Sampling {args.num_samples} fineweb texts...")
        print(f"Output directory: {args.output_dir}")
        print(f"{'='*80}\n")

        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)

        # 直接从fineweb数据集抽取原始文本
        fineweb_iter = iter(dataset.fineweb_dataset)
        sampled_count = 0
        attempt_count = 0
        max_attempts = args.num_samples * 10

        while sampled_count < args.num_samples and attempt_count < max_attempts:
            try:
                attempt_count += 1
                sample = next(fineweb_iter)

                # 提取原始文本
                raw_text = sample.get('text', '')
                if not raw_text or len(raw_text) < 100:
                    continue

                # 提取元数据
                url = sample.get('url', 'unknown')

                # 清洗文本（仅用于对比）
                cleaned_text = dataset.clean_fineweb_text(raw_text)

                # 保存到文件
                output_file = os.path.join(args.output_dir, f'fineweb_sample_{sampled_count+1}.txt')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write(f"FineWeb-Edu Sample {sampled_count+1}\n")
                    f.write(f"URL: {url}\n")
                    f.write("="*80 + "\n\n")

                    f.write("--- RAW TEXT (FULL) ---\n")
                    f.write(raw_text + "\n\n")

                    f.write("--- CLEANED TEXT (FULL) ---\n")
                    f.write(cleaned_text + "\n\n")

                    f.write("--- STATISTICS ---\n")
                    f.write(f"URL: {url}\n")
                    f.write(f"Raw length (chars): {len(raw_text)}\n")
                    f.write(f"Cleaned length (chars): {len(cleaned_text)}\n")
                    f.write(f"Reduction: {len(raw_text) - len(cleaned_text)} chars ({100*(1-len(cleaned_text)/len(raw_text)):.1f}%)\n")

                    # Tokenize
                    try:
                        tokens = tokenizer.encode(raw_text)
                        f.write(f"Raw token count: {len(tokens)}\n")
                        tokens_cleaned = tokenizer.encode(cleaned_text)
                        f.write(f"Cleaned token count: {len(tokens_cleaned)}\n")
                    except:
                        f.write(f"Token count: (failed to tokenize)\n")

                    # 其他元数据
                    f.write(f"\n--- METADATA ---\n")
                    for key, value in sample.items():
                        if key not in ['text']:
                            f.write(f"{key}: {value}\n")

                    f.write(f"\n--- NOTE ---\n")
                    f.write(f"实际训练使用: RAW TEXT (完整原文，不清洗)\n")
                    f.write(f"CLEANED TEXT 仅用于对比展示\n")

                print(f"✓ Saved sample {sampled_count+1} to {output_file}")
                print(f"  URL: {url[:80]}...")
                print(f"  Raw length: {len(raw_text)} chars")
                print(f"  Cleaned length: {len(cleaned_text)} chars")
                print(f"  Reduction: {100*(1-len(cleaned_text)/len(raw_text)):.1f}%")

                sampled_count += 1

            except StopIteration:
                print(f"✗ Reached end of fineweb dataset after {attempt_count} attempts")
                break
            except Exception as e:
                print(f"✗ Error processing sample: {e}")
                continue

        print(f"\n{'='*80}")
        print(f"Fineweb sampling completed!")
        print(f"  Successfully sampled: {sampled_count}/{args.num_samples}")
        print(f"  Total attempts: {attempt_count}")
        print(f"  Output directory: {args.output_dir}")
        print(f"{'='*80}\n")

    else:
        # 测试模式：正常测试迭代
        print("\nTesting iteration...")
        for i, sample in enumerate(dataset):
            if i >= 5:  # Test first 5 samples
                break

            print(f"\nSample {i+1}:")
            print(f"  Input IDs shape: {sample['input_ids'].shape}")
            print(f"  True target shape: {sample['true_target'].shape}")
            print(f"  Start pos: {sample['start_pos']}")
            print(f"  Total length: {sample['total_length']}")
            print(f"  Is ending: {sample.get('is_ending', False)}")  # 显示是否为结尾样本
            print(f"  First 5 targets: {sample['true_target'][:5].tolist()}")
            print(f"  Last 5 targets: {sample['true_target'][-5:].tolist()}")

        print("\n" + "=" * 80)
        print("Test completed!")
        print("=" * 80)

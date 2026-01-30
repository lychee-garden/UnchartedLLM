#!/usr/bin/env python3
"""
download_datasets.py - 预下载数据集到本地

功能：
1. 下载 RedPajama arxiv 数据集
2. 下载 fineweb-edu 数据集
3. 保存到 ./datasets/ 目录
4. 支持断点续传
"""

import os
import sys
import argparse
from datasets import load_dataset, DownloadConfig
from huggingface_hub import hf_hub_download
import requests
from tqdm import tqdm

# 数据集配置
DATASETS_DIR = "./datasets"
REDPAJAMA_DATASET = "togethercomputer/RedPajama-Data-1T"
FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_SPLIT = "sample-10BT"


def download_redpajama_arxiv(save_dir: str, num_files: int = 10):
    """下载 RedPajama arxiv 数据集"""
    print("\n" + "="*80)
    print("Downloading RedPajama Arxiv Dataset")
    print("="*80)

    arxiv_dir = os.path.join(save_dir, "arxiv")
    os.makedirs(arxiv_dir, exist_ok=True)

    try:
        print(f"Dataset: {REDPAJAMA_DATASET} (arxiv subset)")
        print(f"Number of files: {num_files}")
        print(f"Save to: {arxiv_dir}")
        print("\nStep 1: Downloading URL list...")

        # 下载 URL 列表
        url_file = hf_hub_download(
            repo_id=REDPAJAMA_DATASET,
            filename="urls/arxiv.txt",
            repo_type="dataset"
        )

        # 读取 URL
        with open(url_file, 'r') as f:
            urls = [line.strip() for line in f.readlines()]

        # 限制文件数量
        urls = urls[:num_files]
        print(f"✓ Found {len(urls)} arxiv file URLs")

        print(f"\nStep 2: Downloading {len(urls)} JSONL files...")

        # 下载每个文件
        success_count = 0
        failed_urls = []

        for i, url in enumerate(tqdm(urls, desc="Downloading")):
            filename = os.path.basename(url)
            output_path = os.path.join(arxiv_dir, filename)

            # 检查文件是否已存在
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 1000:  # 文件大于 1KB，认为已下载
                    print(f"  [{i+1}/{len(urls)}] ✓ Already exists: {filename}")
                    success_count += 1
                    continue

            # 下载文件
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))

                with open(output_path, 'wb') as f:
                    if total_size > 0:
                        with tqdm(total=total_size, unit='B', unit_scale=True,
                                 desc=f"  [{i+1}/{len(urls)}] {filename}", leave=False) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                success_count += 1
                print(f"  [{i+1}/{len(urls)}] ✓ Downloaded: {filename}")

            except Exception as e:
                print(f"  [{i+1}/{len(urls)}] ✗ Failed: {filename} - {e}")
                failed_urls.append(url)
                # 删除不完整的文件
                if os.path.exists(output_path):
                    os.remove(output_path)

        print(f"\n✓ RedPajama arxiv dataset downloaded!")
        print(f"  Success: {success_count}/{len(urls)}")
        print(f"  Failed: {len(failed_urls)}/{len(urls)}")
        print(f"  Saved to: {arxiv_dir}")

        if failed_urls:
            print(f"\n  Failed URLs:")
            for url in failed_urls:
                print(f"    - {url}")

        return success_count > 0

    except Exception as e:
        print(f"✗ Failed to download RedPajama arxiv dataset: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def download_fineweb(save_dir: str):
    """下载 fineweb-edu 数据集"""
    print("\n" + "="*80)
    print("Downloading FineWeb-Edu Dataset")
    print("="*80)

    fineweb_dir = os.path.join(save_dir, "fineweb")
    os.makedirs(fineweb_dir, exist_ok=True)

    try:
        # 配置下载参数
        download_config = DownloadConfig(
            cache_dir=fineweb_dir,
            max_retries=10,  # Increase retries for network issues
        )

        print(f"Dataset: {FINEWEB_DATASET}")
        print(f"Split: {FINEWEB_SPLIT}")
        print(f"Save to: {fineweb_dir}")
        print("Downloading...")
        print("Note: This may take a while (large dataset)...")

        # 下载数据集（非streaming模式，完整下载）
        dataset = load_dataset(
            FINEWEB_DATASET,
            name=FINEWEB_SPLIT,
            split="train",
            cache_dir=fineweb_dir,
            download_config=download_config
        )

        print(f"✓ FineWeb-Edu dataset downloaded successfully!")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Saved to: {fineweb_dir}")

        return True

    except Exception as e:
        print(f"✗ Failed to download fineweb dataset: {e}")
        print(f"  Error type: {type(e).__name__}")

        # 提供SSL错误的解决方案
        if "SSL" in str(e) or "SSLError" in str(type(e).__name__):
            print("\n  SSL Error detected. Try one of these solutions:")
            print("  1. Set environment variable: export HF_ENDPOINT=https://hf-mirror.com")
            print("  2. Use VPN or proxy")
            print("  3. Retry later (may be temporary network issue)")

        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Download datasets for UnchartedLLM training")
    parser.add_argument('--arxiv_files', type=int, default=10,
                        help='Number of RedPajama arxiv files to download (default: 10)')
    parser.add_argument('--skip_arxiv', action='store_true',
                        help='Skip downloading arxiv dataset')
    parser.add_argument('--skip_fineweb', action='store_true',
                        help='Skip downloading fineweb dataset')
    parser.add_argument('--output_dir', type=str, default=DATASETS_DIR,
                        help=f'Output directory (default: {DATASETS_DIR})')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("Dataset Downloader for UnchartedLLM")
    print("="*80)
    print(f"Datasets will be saved to: {args.output_dir}")
    print(f"RedPajama arxiv files: {args.arxiv_files}")
    print("="*80)

    # 创建数据集目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 检查环境变量
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    if hf_endpoint != "https://huggingface.co":
        print(f"\nUsing HuggingFace endpoint: {hf_endpoint}")

    # 下载数据集
    results = {}

    # 1. 下载 RedPajama Arxiv
    if not args.skip_arxiv:
        results['arxiv'] = download_redpajama_arxiv(args.output_dir, args.arxiv_files)
    else:
        print("\n⊘ Skipping arxiv dataset download")
        results['arxiv'] = None

    # 2. 下载 FineWeb
    if not args.skip_fineweb:
        results['fineweb'] = download_fineweb(args.output_dir)
    else:
        print("\n⊘ Skipping fineweb dataset download")
        results['fineweb'] = None

    # 打印总结
    print("\n" + "="*80)
    print("Download Summary")
    print("="*80)

    if results['arxiv'] is not None:
        print(f"RedPajama Arxiv: {'✓ Success' if results['arxiv'] else '✗ Failed'}")
    else:
        print(f"RedPajama Arxiv: ⊘ Skipped")

    if results['fineweb'] is not None:
        print(f"FineWeb:         {'✓ Success' if results['fineweb'] else '✗ Failed'}")
    else:
        print(f"FineWeb:         ⊘ Skipped")

    print("="*80)

    # 检查是否有成功下载的数据集
    downloaded = [k for k, v in results.items() if v is True]

    if downloaded:
        print(f"\n✓ Downloaded datasets: {', '.join(downloaded)}")
        print(f"\nDatasets saved to: {args.output_dir}")
        print(f"\nTo use local datasets in training:")
        print(f"  python train.py  # Will auto-detect local datasets")
        print(f"\nTo test the dataset:")
        print(f"  python dataset.py --sample_arxiv --num_samples 5")
        return 0
    elif all(v is None for v in results.values()):
        print("\n⊘ All datasets skipped.")
        return 0
    else:
        print("\n✗ All datasets failed to download.")
        print("\nYou can still use streaming mode (default) for training.")
        print("Or try downloading with different settings:")
        print(f"  python download_datasets.py --arxiv_files 5  # Download fewer files")
        return 1


if __name__ == "__main__":
    sys.exit(main())

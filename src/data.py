"""
数据模块 - 小说下载、缓存和批处理
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .config import Config, DataConfig
from .tokenizer import Tokenizer


class NovelDataset(Dataset):
    """小说数据集 - 支持流式读取和缓存"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Tokenizer,
        block_size: int,
        cache_tokens: bool = True,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cache_tokens = cache_tokens
        
        # 加载并编码文本
        cache_path = Path(data_path).with_suffix(".tokens")
        
        if cache_tokens and cache_path.exists():
            self.tokens = torch.load(cache_path)
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
            self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
            if cache_tokens:
                torch.save(self.tokens, cache_path)
    
    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + self.block_size + 1]
        return x, y


def download_novel(url: str, save_path: str, timeout: int = 30) -> bool:
    """下载单本小说"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        # Project Gutenberg 文本可能包含头部/尾部信息
        text = response.text
        
        # 简单清理：移除 Project Gutenberg 头尾
        markers = [
            ("*** START OF", "*** END OF"),
            ("***START OF", "***END OF"),
        ]
        for start_marker, end_marker in markers:
            start_idx = text.find(start_marker)
            if start_idx != -1:
                # 找到该行的结尾
                newline_idx = text.find("\n", start_idx)
                if newline_idx != -1:
                    text = text[newline_idx + 1:]
            
            end_idx = text.find(end_marker)
            if end_idx != -1:
                text = text[:end_idx]
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def prepare_data(config: Config, tokenizer: Tokenizer) -> Tuple[Dataset, Dataset]:
    """准备训练和验证数据集"""
    data_config = config.data
    data_dir = Path(data_config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载小说
    combined_path = data_dir / "novels_combined.txt"
    
    if not combined_path.exists():
        print("Downloading novels...")
        all_texts = []
        
        for url in tqdm(data_config.novels, desc="Downloading"):
            filename = url.split("/")[-1].replace(".txt", "") + ".txt"
            save_path = data_dir / filename
            
            if not save_path.exists():
                success = download_novel(url, str(save_path))
                if not success:
                    continue
            
            with open(save_path, "r", encoding="utf-8") as f:
                all_texts.append(f.read())
        
        # 合并所有文本
        combined_text = "\n\n".join(all_texts)
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write(combined_text)
        
        print(f"Combined text: {len(combined_text):,} characters")
    
    # 编码整个数据集
    cache_path = data_dir / "novels_combined.tokens"
    if cache_path.exists():
        tokens = torch.load(cache_path)
    else:
        with open(combined_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        torch.save(tokens, cache_path)
    
    print(f"Total tokens: {len(tokens):,}")
    
    # 划分训练/验证集 (90/10)
    split_idx = int(len(tokens) * 0.9)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    
    # 创建数据集（使用内存切片，避免重复加载）
    class TokenDataset(Dataset):
        def __init__(self, tokens: torch.Tensor, block_size: int):
            self.tokens = tokens
            self.block_size = block_size
        
        def __len__(self) -> int:
            return max(0, len(self.tokens) - self.block_size)
        
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.tokens[idx:idx + self.block_size]
            y = self.tokens[idx + 1:idx + self.block_size + 1]
            return x, y
    
    train_dataset = TokenDataset(train_tokens, config.model.block_size)
    val_dataset = TokenDataset(val_tokens, config.model.block_size)
    
    return train_dataset, val_dataset


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """创建数据加载器"""
    # M2 Mac 上 num_workers=0 更稳定
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # MPS 不支持 pin_memory
    )


if __name__ == "__main__":
    from .config import get_mini_config
    from .tokenizer import TiktokenTokenizer
    
    config = get_mini_config()
    tokenizer = TiktokenTokenizer()
    
    train_ds, val_ds = prepare_data(config, tokenizer)
    print(f"Train samples: {len(train_ds):,}")
    print(f"Val samples: {len(val_ds):,}")
    
    train_loader = get_dataloader(train_ds, batch_size=4)
    for x, y in train_loader:
        print(f"Batch shape: x={x.shape}, y={y.shape}")
        print(f"Sample: {tokenizer.decode(x[0].tolist()[:50])}")
        break

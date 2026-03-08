"""
训练脚本
"""

import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .config import Config
from .data import prepare_data, get_dataloader
from .model import GPT, create_model
from .tokenizer import TiktokenTokenizer


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.torch_device
        
        print(f"Using device: {self.device}")
        
        # 分词器
        self.tokenizer = TiktokenTokenizer()
        
        # 模型
        self.model = create_model(config.model, self.device)
        
        # 数据
        print("Preparing data...")
        self.train_dataset, self.val_dataset = prepare_data(config, self.tokenizer)
        
        # 数据加载器
        self.train_loader = get_dataloader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
        )
        self.val_loader = get_dataloader(
            self.val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay,
        )
        
        # 学习率调度器
        def lr_lambda(step):
            if step < config.training.warmup_steps:
                return step / config.training.warmup_steps
            return 1.0 - step / config.training.max_steps
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
        
        # 检查点目录
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.step = 0
        self.best_val_loss = float("inf")
    
    def train_step(self, batch: tuple) -> float:
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        self.optimizer.zero_grad()
        _, loss = self.model(x, targets=y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.grad_clip,
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, max_batches: int = None) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            _, loss = self.model(x, targets=y)
            total_loss += loss.item()
            num_batches += 1
            
            if max_batches and num_batches >= max_batches:
                break
        
        self.model.train()
        return total_loss / num_batches
    
    def train(self):
        print(f"\nStarting training for {self.config.training.max_steps} steps...")
        print(f"Train samples: {len(self.train_dataset):,}")
        print(f"Val samples: {len(self.val_dataset):,}")
        
        self.model.train()
        start_time = time.time()
        running_loss = 0.0
        
        data_iter = iter(self.train_loader)
        
        pbar = tqdm(range(self.config.training.max_steps), desc="Training")
        for step in pbar:
            # 获取批次
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            
            # 训练步
            loss = self.train_step(batch)
            running_loss += loss
            self.step += 1
            
            # 更新进度条
            pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
            
            # 评估
            if self.step % self.config.training.eval_interval == 0:
                val_loss = self.evaluate(self.config.training.eval_steps)
                avg_train_loss = running_loss / self.config.training.eval_interval
                
                print(f"\nStep {self.step}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt")
                
                running_loss = 0.0
            
            # 保存检查点
            if self.step % self.config.training.save_interval == 0:
                self.save_checkpoint(f"step_{self.step}.pt")
        
        # 训练完成
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/60:.1f} minutes")
        self.save_checkpoint("final.pt")
    
    def save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        self.model.save_checkpoint(str(path), self.optimizer, self.step)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = checkpoint.get("step", 0)
        print(f"Loaded checkpoint from step {self.step}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default", choices=["mini", "default", "small"])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    # 选择配置
    if args.config == "mini":
        from .config import get_mini_config
        config = get_mini_config()
    elif args.config == "small":
        from .config import get_small_config
        config = get_small_config()
    else:
        config = Config()
    
    # 覆盖参数
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    
    # 训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

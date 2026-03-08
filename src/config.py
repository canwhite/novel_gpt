"""
配置中心 - 集中管理所有超参数和路径配置
适配 MacBook M2 (MPS 加速)
"""

from dataclasses import dataclass, field
from typing import Literal
import torch


@dataclass
class ModelConfig:
    """模型架构配置"""
    n_layer: int = 4           # Transformer 层数
    n_embd: int = 128          # 嵌入维度
    n_head: int = 4            # 注意力头数
    block_size: int = 256      # 最大上下文长度
    vocab_size: int = 100277   # tiktoken cl100k_base 词表大小 (支持中英文)
    dropout: float = 0.1       # Dropout 比率
    
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head
    
    @property
    def param_count(self) -> int:
        """估算参数量"""
        # Token embedding + Position embedding
        emb = self.vocab_size * self.n_embd + self.block_size * self.n_embd
        # Per layer: attn (4 matrices) + mlp (2 matrices) + 2 layernorms
        per_layer = (
            4 * self.n_embd * self.n_embd +  # Q, K, V, O projections
            2 * self.n_embd * self.n_embd * 4 +  # MLP (expand 4x then project back)
            2 * self.n_embd  # LayerNorm parameters (2 per layer)
        )
        # Output head
        lm_head = self.vocab_size * self.n_embd
        return emb + self.n_layer * per_layer + lm_head


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 16       # 批大小 (M2 内存友好)
    learning_rate: float = 3e-4
    max_steps: int = 10000     # 训练步数
    warmup_steps: int = 100    # 预热步数
    eval_interval: int = 500   # 评估间隔
    eval_steps: int = 100      # 评估步数
    save_interval: int = 1000  # 保存间隔
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0     # 梯度裁剪


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    tokenizer_type: Literal["tiktoken", "char"] = "tiktoken"
    
    # 中文小说训练数据（手动放置在 data/ 目录）
    novels: list = field(default_factory=list)


@dataclass
class GenerationConfig:
    """生成配置"""
    temperature: float = 0.8
    top_k: int = 40            # Top-k 采样
    top_p: float = 0.9         # Top-p (nucleus) 采样
    max_new_tokens: int = 500  # 最大生成长度


@dataclass
class Config:
    """主配置类"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # 设备配置 - M2 MPS 加速
    device: str = field(default="auto")
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        if self.device == "auto":
            # M2 Mac 优先使用 MPS
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
    
    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


# 默认配置实例
default_config = Config()


def get_mini_config() -> Config:
    """迷你配置 - 快速测试用"""
    config = Config()
    config.model.n_layer = 2
    config.model.n_embd = 64
    config.model.n_head = 2
    config.model.block_size = 128
    config.training.batch_size = 8
    config.training.max_steps = 100
    return config


def get_small_config() -> Config:
    """小型配置 - ~10M 参数"""
    config = Config()
    config.model.n_layer = 6
    config.model.n_embd = 256
    config.model.n_head = 8
    config.model.block_size = 512
    return config


if __name__ == "__main__":
    # 打印默认配置
    config = Config()
    print(f"Device: {config.device}")
    print(f"Model params: {config.model.param_count:,}")
    print(f"Config: {config.model}")

# Novel GPT - 项目文档

## 项目定位
基于 PyTorch 的 GPT 模型训练系统，用于小说文本生成。适配 MacBook M2 (MPS 加速)，支持中英文训练。

## 核心架构

### 模块结构
```
src/
├── __init__.py      # 模块导出
├── __main__.py      # CLI 入口 (train/generate/info 命令)
├── config.py        # 配置中心 (ModelConfig, TrainingConfig, DataConfig)
├── tokenizer.py     # 分词器 (TiktokenTokenizer 使用 cl100k_base)
├── data.py          # 数据加载 (NovelDataset, prepare_data)
├── model.py         # GPT 模型 (Transformer 架构)
├── train.py         # 训练脚本
└── generate.py      # 生成脚本
```

### 技术栈
- **语言**: Python 3.12
- **框架**: PyTorch 2.0+ (MPS 加速)
- **分词**: tiktoken (cl100k_base, 词表 100277)
- **管理**: uv (包管理)

### 关键配置
- **模型**: n_layer=4, n_embd=128, n_head=4, block_size=256
- **训练**: batch_size=16, lr=3e-4, max_steps=10000
- **设备**: M2 MPS 自动检测
- **数据**: data/ 目录存放训练文本

## 目录结构说明
```
novel_gpt/
├── data/              # 训练数据目录
│   └── *.txt         # 中文/英文小说文本
├── checkpoints/       # 模型检查点
├── docs/             # 文档
├── schema/           # 任务文档
└── src/              # 源代码
```

## 部署流程
1. **准备数据**: 将中文小说文本放入 data/ 目录
2. **训练**: `uv run python -m novel_gpt train --config mini --max_steps 1000`
3. **生成**: `uv run python -m novel_gpt generate checkpoints/best.pt --prompt "提示词"`

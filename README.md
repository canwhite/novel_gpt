# Novel GPT - 小说训练系统

基于 PyTorch 的 GPT 模型，用于训练和生成小说文本。适配 MacBook M2 (MPS 加速)。

## 快速开始

```bash
cd /Users/zack/Desktop/microGPT

# 训练（mini 配置，快速测试）
uv run python -m novel_gpt train --config mini --max_steps 1000

# 生成文本
uv run python -m novel_gpt generate checkpoints/best.pt --prompt "天下大势"

# 交互模式
uv run python -m novel_gpt generate checkpoints/best.pt --interactive
```

---

## 目录结构

```
novel_gpt/
├── __init__.py      # 模块导出
├── __main__.py      # CLI 入口
├── config.py        # 配置中心
├── tokenizer.py     # 分词器
├── data.py          # 数据加载
├── model.py         # GPT 模型
├── train.py         # 训练脚本
└── generate.py      # 生成脚本
```

---

## 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                        数据准备                              │
├─────────────────────────────────────────────────────────────┤
│  1. 从 Project Gutenberg 自动下载小说                        │
│  2. 合并所有文本 → novels_combined.txt                       │
│  3. tiktoken 编码 → novels_combined.tokens (缓存)            │
│  4. 划分训练集 (90%) / 验证集 (10%)                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        模型训练                              │
├─────────────────────────────────────────────────────────────┤
│  1. 初始化 GPT 模型 → 移动到 MPS 设备                        │
│  2. DataLoader 批量加载数据                                  │
│  3. 循环: 前向传播 → 计算损失 → 反向传播 → 更新参数           │
│  4. 定期评估验证集损失，保存最佳模型                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        文本生成                              │
├─────────────────────────────────────────────────────────────┤
│  1. 加载训练好的 checkpoint                                 │
│  2. 编码 prompt → token 序列                                │
│  3. 自回归生成: 预测下一个 token → 追加 → 重复               │
│  4. 温度采样 / Top-k 采样                                   │
│  5. 解码 tokens → 文本                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 技术实现

### 1. 分词器 (tokenizer.py)

```python
# tiktoken GPT-2 编码，词表大小 50257
from novel_gpt import TiktokenTokenizer

tok = TiktokenTokenizer()
tokens = tok.encode("Hello, world!")  # [15496, 11, 995, 0]
text = tok.decode(tokens)             # "Hello, world!"
```

**为什么用 tiktoken**：

- 子词分词，比字符级更高效
- 词表 50257，平衡了效率和泛化
- 与 GPT-2/GPT-3 兼容，可加载预训练权重

### 2. 数据加载 (data.py)

```python
# 自动下载 Project Gutenberg 小说
novels = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
    "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
    # ...
]

# 批量加载数据
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

**缓存机制**：

- 首次运行：下载文本 → 编码 → 保存 `.tokens` 文件
- 后续运行：直接加载 `.tokens`，跳过编码

### 3. 模型架构 (model.py)

```python
GPT(
    transformer = {
        'wte': Embedding(vocab_size, n_embd),      # Token 嵌入
        'wpe': Embedding(block_size, n_embd),      # 位置嵌入
        'h': [Block(...) for _ in range(n_layer)], # Transformer 块
        'ln_f': LayerNorm(n_embd),                 # 最终归一化
    },
    lm_head = Linear(n_embd, vocab_size)           # 输出投影
)
```

**Block 结构**：

```
x → LayerNorm → MultiHead Attention → 残差连接
                                         ↓
x → LayerNorm → MLP (4x expansion)    → 残差连接 → output
```

**M2 适配**：

```python
# 自动检测设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
```

### 4. 训练 (train.py)

```python
# AdamW 优化器
optimizer = AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)

# 学习率调度：warmup + 线性衰减
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0 - step / max_steps

# 训练循环
for batch in train_loader:
    loss = model(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)  # 梯度裁剪
    optimizer.step()
```

**关键技巧**：

- 梯度裁剪防止梯度爆炸
- 权重衰减正则化
- 学习率预热稳定训练初期

### 5. 生成 (generate.py)

```python
# 自回归生成
for _ in range(max_new_tokens):
    logits = model(tokens)              # 前向传播
    logits = logits[-1] / temperature   # 温度缩放

    # Top-k 采样
    top_k = 40
    v, _ = torch.topk(logits, top_k)
    logits[logits < v[-1]] = -inf

    probs = softmax(logits)
    next_token = multinomial(probs)     # 按概率采样
    tokens.append(next_token)
```

**温度参数**：

- `temperature = 0.5`：保守，更确定性
- `temperature = 0.8`：平衡
- `temperature = 1.0+`：创造性，更多随机性

---

## 配置说明

### 预设配置

| 配置    | 参数量 | n_layer | n_embd | n_head | block_size | 用途     |
| ------- | ------ | ------- | ------ | ------ | ---------- | -------- |
| mini    | ~3.3M  | 2       | 64     | 2      | 128        | 快速测试 |
| default | ~7.2M  | 4       | 128    | 4      | 256        | 标准训练 |
| small   | ~38M   | 6       | 256    | 8      | 512        | 更强模型 |

### 查看配置

```bash
uv run python -m novel_gpt info --config default
```

### 自定义配置

```python
from novel_gpt import Config

config = Config()
config.model.n_layer = 6
config.model.n_embd = 256
config.training.batch_size = 32
config.training.max_steps = 20000
```

---

## CLI 命令

### 训练

```bash
# 基础训练
uv run python -m novel_gpt train --config mini

# 自定义参数
uv run python -m novel_gpt train \
    --config default \
    --max_steps 10000 \
    --batch_size 8 \
    --device mps

# 后台训练
nohup uv run python -m novel_gpt train --config default --max_steps 10000 > train.log 2>&1 &
tail -f train.log
```

### 生成

```bash
# 从 prompt 生成
uv run python -m novel_gpt generate checkpoints/best.pt \
    --prompt "It was a dark and stormy night" \
    --max_tokens 500 \
    --temperature 0.8

# 交互模式
uv run python -m novel_gpt generate checkpoints/best.pt --interactive

# 低温度（更确定）
uv run python -m novel_gpt generate checkpoints/best.pt \
    --prompt "She opened the door" \
    --temperature 0.5
```

---

## 训练建议

### 数据量

- 当前：6 本小说，~750K tokens
- 建议：10-50 本小说，~1M-5M tokens

### 训练步数

| 模型大小       | 建议步数 | 预期 loss | 预计时间 (M2) |
| -------------- | -------- | --------- | ------------- |
| mini (3.3M)    | 5,000    | ~5.0      | ~30 分钟      |
| default (7.2M) | 10,000   | ~4.5      | ~2 小时       |
| small (38M)    | 20,000   | ~4.0      | ~8 小时       |

### Loss 参考

| Loss    | 生成质量           |
| ------- | ------------------ |
| > 8.0   | 随机字符           |
| 6.0-8.0 | 学习词频，无意义   |
| 4.5-6.0 | 简单短语，开始连贯 |
| 3.5-4.5 | 短句连贯，有语法   |
| < 3.5   | 长文本连贯         |

---

## 添加自己的数据

### 方法一：修改配置

```python
# config.py
novels = [
    "https://your-novel-url.com/novel.txt",
    # 或本地路径
]
```

### 方法二：直接放入

```bash
# 将文本文件放入 data/ 目录
cp your_novel.txt /Users/zack/Desktop/microGPT/data/

# 删除缓存重新编码
rm data/novels_combined.*
```

---

## 依赖

```toml
torch >= 2.0.0      # MPS 支持
tiktoken >= 0.5.0   # GPT 分词器
requests >= 2.32.0  # 数据下载
tqdm >= 4.66.0      # 进度条
numpy >= 1.24.0     # 数值计算
```

安装：

```bash
uv sync
```

---

## 与原 novel.py 对比

| 特性     | 原 novel.py       | 新 novel_gpt           |
| -------- | ----------------- | ---------------------- |
| 分词     | 字符级 (vocab≈27) | tiktoken (vocab=50257) |
| 自动微分 | 手写 Value 类     | PyTorch                |
| 设备     | CPU               | MPS (M2 GPU)           |
| 批处理   | 单样本            | 批量训练               |
| 参数量   | ~15K              | 3.3M - 38M             |
| 数据     | 人名列表          | 小说文本               |
| 上下文   | 16 tokens         | 128-512 tokens         |

---

## 常见问题

### Q: 生成的文本不连贯？

A: 训练步数不够。继续训练到 loss < 5.0。

### Q: 训练很慢？

A:

1. 确认使用了 MPS：日志应显示 `Using device: mps`
2. 减小 batch_size
3. 使用 mini 配置快速验证

### Q: 内存不足？

A:

1. 减小 `block_size`
2. 减小 `batch_size`
3. 使用更小的配置

### Q: 如何恢复训练？

A: 目前不支持，需重新开始。建议训练前确定好步数。

---

## 许可

教育项目，代码参考 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)。

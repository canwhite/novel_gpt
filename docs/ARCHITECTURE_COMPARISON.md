# novel.py vs novel_gpt 架构对比

本文档详细对比原始 `novel.py` 与重构后的 `novel_gpt/` 模块的核心差异。

---

## 总览

| 维度         | novel.py          | novel_gpt              |
| ------------ | ----------------- | ---------------------- |
| **目的**     | 教学演示          | 生产训练               |
| **自动微分** | 手写 Value 类     | PyTorch autograd       |
| **分词器**   | 字符级 (vocab≈27) | tiktoken (vocab=50257) |
| **参数量**   | ~15K              | 3.3M - 38M             |
| **设备**     | CPU only          | MPS/CUDA/CPU           |
| **代码行数** | 238 行单文件      | 模块化 7 文件          |

---

## 一、自动微分系统

### novel.py: 手写 Value 类

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 前向传播值
        self.grad = 0                   # 梯度
        self._children = children       # 计算图子节点
        self._local_grads = local_grads # 局部导数

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def backward(self):
        # 拓扑排序
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 反向传播
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad  # 链式法则
```

**特点**：

- 完全透明：每个操作的计算图构建可见
- 教学价值高：理解反向传播原理
- 性能差：纯 Python 循环，慢 100-1000x

### novel_gpt: PyTorch autograd

```python
import torch

# 前向传播
logits = model(x)
# 这个是什么损失算法
loss = F.cross_entropy(logits, targets)

# 反向传播（一行搞定）
loss.backward()

# 参数更新
optimizer.step()
```

**特点**：

- 高度优化：C++/CUDA 后端
- 自动求导：无需手动构建计算图
- 支持复杂操作：卷积、注意力等

### 对比

| 特性           | novel.py Value       | PyTorch autograd           |
| -------------- | -------------------- | -------------------------- |
| **计算图构建** | 显式，手动           | 隐式，自动                 |
| **梯度累积**   | 手动 `child.grad +=` | 自动处理                   |
| **内存优化**   | 无                   | 自动释放中间结果           |
| **并行计算**   | 无                   | GPU 加速                   |
| **调试难度**   | 低（完全透明）       | 中（需理解 autograd 机制） |

---

## 二、分词器

### novel.py: 字符级分词

```python
# 构建词表
uchars = sorted(set(''.join(docs)))  # ['a', 'b', 'c', ..., 'z']
vocab_size = len(uchars) + 1  # ~27

# 编码
tokens = [uchars.index(ch) for ch in "hello"]  # [7, 4, 11, 11, 14]

# 解码
text = ''.join(uchars[t] for t in tokens)  # "hello"
```

**问题**：

- 词表太小：只认识小写字母
- 效率低：每个字符一个 token
- 泛化差：无法处理数字、标点、大写

### novel_gpt: tiktoken 子词分词

```python
import tiktoken

enc = tiktoken.get_encoding('gpt2')

# 编码
tokens = enc.encode("Hello, world!")  # [15496, 11, 995, 0]

# 解码
text = enc.decode(tokens)  # "Hello, world!"

# 词表大小
vocab_size = enc.n_vocab  # 50257
```

**优势**：

- 词表大：50257 个 token
- 效率高：常见单词一个 token
- 泛化好：支持所有 Unicode 字符

### 对比

| 特性         | 字符级         | tiktoken       |
| ------------ | -------------- | -------------- |
| **词表大小** | ~27            | 50257          |
| **编码密度** | 每字符 1 token | 每词 ≈ 1 token |
| **未见字符** | 无法处理       | 子词拆分       |
| **训练效率** | 低（序列过长） | 高             |

**编码示例**：

| 文本                  | 字符级 tokens | tiktoken tokens |
| --------------------- | ------------- | --------------- |
| "hello"               | 5             | 1 (31373)       |
| "Hello, world!"       | 13            | 4               |
| "The quick brown fox" | 19            | 5               |

---

## 三、模型架构

### novel.py: 极简实现

```python
# 参数
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4

# 参数存储
state_dict = {
    'wte': [[Value(...) for _ in range(n_embd)] for _ in range(vocab_size)],
    'wpe': [[Value(...) for _ in range(n_embd)] for _ in range(block_size)],
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    # ...

# 前向传播：逐 token 递归
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]

    for li in range(n_layer):
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)   # 手动缓存
        values[li].append(v)
        # ...
```

**特点**：

- 逐 token 处理：每次只计算一个 token 的表示
- 手动 KV 缓存：keys/values 列表追加
- 单层、小维度：参数量 ~15K

### novel_gpt: 批量并行

```python
class GPT(nn.Module):
    def __init__(self, config):
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab_size, n_embd),
            'wpe': nn.Embedding(block_size, n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(n_layer)]),
        })
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # 权重共享
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()

        # 批量嵌入
        tok_emb = self.transformer.wte(idx)  # [B, T, n_embd]
        pos = torch.arange(T)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        # 并行处理所有层
        for block in self.transformer.h:
            x = block(x)

        return self.lm_head(x)
```

**特点**：

- 批量处理：一次计算 [B, T] 个 token
- 矩阵运算：高度并行化
- 可配置：2-6 层，64-512 维度

### 关键差异：注意力机制

**novel.py 递归注意力**：

```python
# 每次处理一个 token
for pos_id in range(n):
    logits = gpt(tokens[pos_id], pos_id, keys, values)

# gpt 内部
for h in range(n_head):
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]  # 所有历史 K
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]  # 所有历史 V

    # Python 列表循环计算注意力
    attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim))
                   for t in range(len(k_h))]
```

**novel_gpt 批量注意力**：

```python
# 一次处理整个序列
def forward(self, x):  # x: [B, T, C]
    qkv = self.c_attn(x)  # [B, T, 3*C]
    q, k, v = qkv.split(self.n_embd, dim=2)

    # 重塑为多头
    q = q.view(B, T, n_head, head_dim).transpose(1, 2)  # [B, nh, T, hd]
    k = k.view(B, T, n_head, head_dim).transpose(1, 2)
    v = v.view(B, T, n_head, head_dim).transpose(1, 2)

    # 矩阵运算计算注意力
    att = (q @ k.transpose(-2, -1)) * scale  # [B, nh, T, T]
    att = att.masked_fill(self.bias == 0, float('-inf'))  # 因果掩码
    att = F.softmax(att, dim=-1)
    y = att @ v  # [B, nh, T, hd]
```

### 对比

| 特性           | novel.py      | novel_gpt            |
| -------------- | ------------- | -------------------- |
| **层数**       | 1             | 2-6                  |
| **嵌入维度**   | 16            | 64-256               |
| **参数量**     | ~15K          | 3.3M-38M             |
| **上下文长度** | 16            | 128-512              |
| **注意力计算** | 逐 token 递归 | 批量矩阵运算         |
| **权重共享**   | 无            | wte = lm_head.weight |

---

## 四、数据处理

### novel.py: 单样本处理

```python
# 数据：人名列表
docs = [line.strip() for line in open('input.txt')]  # ~32K 名字

# 训练：每次取一个名字
for step in range(num_steps):
    doc = docs[step % len(docs)]  # 一个名字
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]

    # 逐 token 前向传播
    for pos_id in range(len(tokens) - 1):
        logits = gpt(tokens[pos_id], pos_id, keys, values)
```

### novel_gpt: 批量处理

```python
# 数据：小说文本
text = open('novels_combined.txt').read()  # ~2.8M 字符
tokens = tokenizer.encode(text)  # ~750K tokens

# 数据集
class NovelDataset(Dataset):
    def __getitem__(self, idx):
        #y是x的向右一位
        x = self.tokens[idx:idx + block_size]
        y = self.tokens[idx + 1:idx + block_size + 1]
        return x, y

# 批量训练
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
for batch in train_loader:
    x, y = batch  # x: [16, 256], y: [16, 256]
    logits, loss = model(x, targets=y)
```

### 对比

| 特性         | novel.py        | novel_gpt         |
| ------------ | --------------- | ----------------- |
| **数据类型** | 人名列表        | 小说文本          |
| **数据量**   | ~32K 样本       | ~750K tokens      |
| **批处理**   | 单样本          | batch_size=16     |
| **序列长度** | 可变（最长 15） | 固定 128-512      |
| **数据加载** | 内存全量        | DataLoader + 缓存 |

---

## 五、训练优化

### novel.py: 手写 Adam

```python
# 初始化优化器状态
m = [0.0] * len(params)  # 一阶矩
v = [0.0] * len(params)  # 二阶矩

# 训练循环
for step in range(num_steps):
    # 前向传播
    loss = forward(...)

    # 反向传播
    loss.backward()

    # Adam 更新
    lr_t = learning_rate * (1 - step / num_steps)  # 线性衰减
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps)
        p.grad = 0
```

### novel_gpt: PyTorch 优化器

```python
# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.01,
)

# 学习率调度
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0 - step / max_steps

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 训练循环
for batch in train_loader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()
```

### 对比

| 特性           | novel.py  | novel_gpt         |
| -------------- | --------- | ----------------- |
| **优化器**     | 手写 Adam | torch.optim.AdamW |
| **学习率调度** | 线性衰减  | Warmup + 线性衰减 |
| **梯度裁剪**   | 无        | grad_clip=1.0     |
| **权重衰减**   | 无        | weight_decay=0.01 |
| **Dropout**    | 无        | dropout=0.1       |

---

## 六、硬件加速

### novel.py: CPU Only

```python
# 所有计算在 CPU 上
# Python 列表 + 浮点数
params = [Value(random.gauss(0, std)) for _ in range(n_params)]
```

**限制**：

- 无 GPU 支持
- Python GIL 限制并行
- 内存带宽瓶颈

### novel_gpt: GPU 加速

```python
# 自动检测设备
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# 模型和数据移动到设备
model = model.to(device)
x = x.to(device)

# GPU 并行计算
logits = model(x)  # 在 GPU 上执行
```

**M2 MPS 特性**：

- 统一内存架构
- 支持 Metal Performance Shaders
- 部分操作可能回退 CPU

### 对比

| 特性         | novel.py    | novel_gpt         |
| ------------ | ----------- | ----------------- |
| **设备**     | CPU only    | MPS/CUDA/CPU      |
| **并行计算** | 无          | GPU 大规模并行    |
| **内存管理** | Python GC   | Tensor + GPU 显存 |
| **性能**     | ~1 step/sec | ~2-10 step/sec    |

---

## 七、代码组织

### novel.py: 单文件

```
novel.py (238 行)
├── 导入 (3 行)
├── 数据加载 (8 行)
├── 分词器 (5 行)
├── Value 类 (70 行)
├── 模型参数 (20 行)
├── 模型函数 (50 行)
├── 训练循环 (40 行)
└── 推理 (15 行)
```

**特点**：

- 紧凑：所有逻辑在一个文件
- 易读：线性结构
- 难扩展：硬编码配置

### novel_gpt: 模块化

```
novel_gpt/
├── __init__.py      # 模块导出
├── __main__.py      # CLI 入口
├── config.py        # 配置中心 (151 行)
├── tokenizer.py     # 分词器 (94 行)
├── data.py          # 数据加载 (175 行)
├── model.py         # GPT 模型 (237 行)
├── train.py         # 训练逻辑 (210 行)
└── generate.py      # 生成逻辑 (141 行)
```

**特点**：

- 解耦：每个模块职责单一
- 可配置：dataclass 配置类
- 易扩展：新增配置/数据源

### 配置管理对比

**novel.py 硬编码**：

```python
n_layer = 1
n_embd = 16
block_size = 16
learning_rate = 0.01
num_steps = 1000
```

**novel_gpt 配置类**：

```python
@dataclass
class ModelConfig:
    n_layer: int = 4
    n_embd: int = 128
    n_head: int = 4
    block_size: int = 256
    vocab_size: int = 50257
    dropout: float = 0.1

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "auto"

# 使用
config = Config()
config.model.n_layer = 6
```

---

## 八、性能对比

### 训练速度

| 配置           | novel.py (CPU) | novel_gpt (M2) |
| -------------- | -------------- | -------------- |
| mini (3.3M)    | -              | ~2 step/sec    |
| default (7.2M) | -              | ~1 step/sec    |
| 15K params     | ~1 step/sec    | -              |

### 生成质量

| 模型              | 训练步数 | Loss | 输出示例                 |
| ----------------- | -------- | ---- | ------------------------ |
| novel.py          | 1000     | ~2.5 | "emily", "jhon", "maria" |
| novel_gpt mini    | 1000     | ~9.5 | 简单词组                 |
| novel_gpt mini    | 5000     | ~5.0 | 短句连贯                 |
| novel_gpt default | 10000    | ~4.0 | 长文本连贯               |

---

## 九、适用场景

### novel.py 适合

- ✅ 学习 Transformer 原理
- ✅ 理解反向传播机制
- ✅ 快速原型验证
- ✅ 无 GPU 环境

### novel_gpt 适合

- ✅ 实际训练任务
- ✅ 大规模数据处理
- ✅ 模型迭代实验
- ✅ 生产环境部署

---

## 十、迁移指南

### 从 novel.py 到 novel_gpt

| 概念         | novel.py                     | novel_gpt                      |
| ------------ | ---------------------------- | ------------------------------ |
| **标量**     | `Value(data)`                | `torch.tensor(data)`           |
| **加法**     | `a + b`                      | `a + b`                        |
| **乘法**     | `a * b`                      | `a * b`                        |
| **反向传播** | `loss.backward()`            | `loss.backward()`              |
| **参数访问** | `state_dict['wte']`          | `model.transformer.wte.weight` |
| **前向传播** | `gpt(token_id, pos_id, ...)` | `model(idx)`                   |

### 代码对照

**novel.py**：

```python
# 前向传播
losses = []
for pos_id in range(n):
    logits = gpt(tokens[pos_id], pos_id, keys, values)
    loss_t = -probs[target_id].log()
    losses.append(loss_t)
loss = sum(losses) / n

# 反向传播
loss.backward()

# 更新
for p in params:
    p.data -= lr * p.grad
    p.grad = 0
```

**novel_gpt**：

```python
# 前向传播（批量）
x = torch.tensor([tokens[:-1]])
y = torch.tensor([tokens[1:]])
logits, loss = model(x, targets=y)

# 反向传播
loss.backward()

# 更新
optimizer.step()
optimizer.zero_grad()
```

---

## 总结

| 维度         | novel.py           | novel_gpt      |
| ------------ | ------------------ | -------------- |
| **定位**     | 教学工具           | 生产框架       |
| **优势**     | 完全透明、易理解   | 高性能、可扩展 |
| **劣势**     | 性能瓶颈、不可扩展 | 门槛高、依赖多 |
| **适合人群** | 初学者、研究者     | 工程师、从业者 |

**novel.py** 展示了 GPT 的核心原理，是理解 Transformer 的最佳起点。

**novel_gpt** 提供了实际训练所需的所有工具，是生产级实现。

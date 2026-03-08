# 开发问题记录

## 问题 1: CLI 参数传递错误

### 现象

```bash
uv run python -m novel_gpt generate checkpoints/final.pt --prompt "Once upon a time"
# 报错: generate: error: the following arguments are required: checkpoint
```

### 原因

`__main__.py` 中构建子命令参数时，`checkpoint` 是位置参数而非可选参数，需要单独处理：

```python
# 错误写法
gen_args = ["generate"]
for k, v in vars(args).items():
    if k == "command" or v is None:  # checkpoint 被跳过了！
        continue
```

### 解决方案

```python
# 正确写法
gen_args = ["generate", args.checkpoint]  # 位置参数单独添加
for k, v in vars(args).items():
    if k in ("command", "checkpoint") or v is None:
        continue
```

### 教训

- 位置参数 (positional argument) 和可选参数 (optional argument) 需要区别处理
- argparse 中 `parser.add_argument("checkpoint")` 定义的是位置参数，必须出现在命令行中

---

## 问题 2: MPS 设备加载 Checkpoint 失败

### 现象

```python
RuntimeError: Placeholder storage has not been allocated on MPS device!
```

### 原因

加载 checkpoint 时，模型参数在 CPU 上初始化，然后直接返回，没有移动到 MPS 设备：

```python
# 错误写法
@classmethod
def load_checkpoint(cls, path: str, device: str = "cpu") -> "GPT":
    checkpoint = torch.load(path, map_location=device)
    model = cls(config)
    model.load_state_dict(checkpoint["model"])
    return model  # 返回的模型在 CPU 上！
```

`map_location=device` 只影响加载时的临时存储，`load_state_dict` 会将参数复制到模型当前设备（CPU）。

### 解决方案

```python
# 正确写法
@classmethod
def load_checkpoint(cls, path: str, device: str = "cpu") -> "GPT":
    checkpoint = torch.load(path, map_location=device)
    model = cls(config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)  # 显式移动到目标设备
    return model
```

### 教训

- `torch.load(map_location=device)` ≠ 模型在 device 上
- `load_state_dict` 后必须显式 `.to(device)`
- MPS 设备与 CUDA 行为不完全一致，需要显式管理设备转移

---

## 问题 3: 生成文本包含大量乱码

### 现象

```
Once upon a time he not the
,... in
 it
, a,,;,,
� out
 a be of,
...
```

大量 `�` 字符出现。

### 原因分析

#### 第一步：检查 token 分布

```python
from collections import Counter
token_counts = Counter(output_tokens)
print(token_counts.most_common(10))
# [(198, 8), (262, 5), (13, 4), (11, 4), (447, 4), ...]
```

Token 447 频繁出现。

#### 第二步：检查 token 解码

```python
import tiktoken
enc = tiktoken.get_encoding('gpt2')

print(enc.decode([447]))  # 输出: '�'
print(enc.decode_tokens_bytes([447]))  # 输出: [b'\xe2\x80']
```

Token 447 解码为 `�`，原始字节是 `\xe2\x80`。

#### 第三步：检查训练数据

```python
# 统计训练数据中 token 447 出现次数
token_447_count = token_counts.get(447, 0)
print(f"Token 447 count: {token_447_count}")  # 输出: 19770
```

Token 447 是训练数据中第 5 常见的 token！

#### 第四步：找到根源

```python
# 找到 token 447 对应的原始文本
context = enc.decode([220, 220, 4808, 36120, 12091, 447, 247, 82, 24501, 13557])
print(context)  # 输出: '   _Reading Jane's Letters._'
```

**发现**：Token 447 是 UTF-8 多字节字符的前两个字节：
- `"` (左智能引号) = `\xe2\x80\x9c` → tokens `[447, 250]`
- `"` (右智能引号) = `\xe2\x80\x9d` → tokens `[447, 251]`

#### 第五步：为什么生成乱码？

模型自回归生成时随机采样，可能只生成了 token 447 而没有生成后续的 250/251，导致不完整的 UTF-8 字节序列，解码时变成乱码。

```
正确: [447, 250] → \xe2\x80\x9c → "
错误: [447]      → \xe2\x80    → �
```

### 解决方案

#### 方案一：后处理清理（已采用）

```python
# generate.py
text = tokenizer.decode(output_tokens)

# 清理无效 UTF-8 字符
text = text.encode('utf-8', errors='ignore').decode('utf-8')
text = text.replace('\ufffd', '')  # 移除 Unicode 替换字符 U+FFFD
```

#### 方案二：预处理训练数据（未采用）

在数据清洗阶段，将智能引号转换为普通引号：

```python
# data.py
text = text.replace('"', '"').replace('"', '"')
text = text.replace(''', "'").replace(''', "'")
```

#### 方案三：采样时过滤无效 token（未采用）

在生成时，检测到 token 447 后强制要求下一个 token 必须是 250 或 251。

### 教训

- tiktoken 的 GPT-2 编码器会将多字节 UTF-8 字符拆分成多个 token
- Project Gutenberg 文本包含大量智能引号等特殊字符
- 自回归采样可能生成不完整的 token 序列
- 数据预处理比后处理更彻底，但后处理更简单快捷

---

## 问题 4: 训练 Loss 不下降

### 现象

训练 1000 步后，loss 仍然很高（~9.5），生成文本不连贯。

### 原因

- **训练步数不够**：~750K tokens 数据，1000 步只看了约 16K tokens（batch_size × block_size × steps）
- **模型太小**：mini config 只有 3.3M 参数
- **数据量有限**：只有 6 本小说

### 解决方案

1. 增加训练步数到 5000-10000
2. 使用 default config（7.2M 参数）
3. 增加训练数据（更多小说）

### Loss 参考

| Loss | 文本质量 |
|------|----------|
| > 8.0 | 随机字符 |
| 6.0-8.0 | 学习词频，无意义 |
| 4.5-6.0 | 简单短语 |
| 3.5-4.5 | 短句连贯 |
| < 3.5 | 长文本连贯 |

---

## 问题 5: 训练中断后无法恢复

### 现象

训练被中断（Ctrl+C 或超时），没有保存 checkpoint，需要从头开始。

### 原因

当前实现只保存 `best.pt` 和 `final.pt`，没有定期保存中间 checkpoint。

### 解决方案（未实现）

```python
# train.py
# 每隔 save_interval 步保存一次
if self.step % self.config.training.save_interval == 0:
    self.save_checkpoint(f"step_{self.step}.pt")

# 支持从 checkpoint 恢复
parser.add_argument("--resume", type=str, help="Resume from checkpoint")
```

---

## 问题 6: M2 MPS 兼容性

### 现象

部分 PyTorch 操作在 MPS 上不支持或行为不一致。

### 注意事项

- `pin_memory=True` 在 MPS 上不支持，DataLoader 应设置 `pin_memory=False`
- MPS 内存是统一内存架构，但仍有容量限制
- 某些操作可能回退到 CPU，影响性能

### 最佳实践

```python
# DataLoader
DataLoader(dataset, batch_size=16, pin_memory=False)

# 设备检测
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 显式设备转移
model = model.to(device)
tensor = tensor.to(device)
```

---

## 总结

| 问题 | 根本原因 | 解决方案 |
|------|----------|----------|
| CLI 参数错误 | 位置参数处理不当 | 单独添加位置参数 |
| MPS 加载失败 | 未显式 .to(device) | load_state_dict 后 .to(device) |
| 生成乱码 | 多字节字符拆分 | 后处理清理无效字符 |
| Loss 不下降 | 训练步数不够 | 增加步数或模型大小 |
| 无法恢复训练 | 缺少中间 checkpoint | 定期保存 checkpoint |
| MPS 兼容性 | 部分操作不支持 | pin_memory=False |

---

## 参考资料

- [tiktoken 文档](https://github.com/openai/tiktoken)
- [PyTorch MPS 文档](https://pytorch.org/docs/stable/notes/mps.html)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

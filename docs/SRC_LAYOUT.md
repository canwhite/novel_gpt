# Src Layout 配置说明

## 问题背景

将代码从项目根目录移动到 `src/` 目录后，`python -m novel_gpt` 命令无法找到模块。

## 问题本质

### 原始目录结构

```
项目根目录/
├── __init__.py
├── __main__.py
├── train.py
├── model.py
└── ...
```

Python 默认将项目根目录加入 `sys.path`，因此 `python -m novel_gpt` 可以直接找到根目录下的模块。

### 移动后的目录结构

```
项目根目录/
└── src/
    ├── __init__.py
    ├── __main__.py
    ├── train.py
    ├── model.py
    └── ...
```

移动后，`src/` 目录不在 Python 的模块搜索路径中，导致导入失败：

```bash
$ python -m novel_gpt
ModuleNotFoundError: No module named 'novel_gpt'
```

## 解决方案

在 `pyproject.toml` 中配置包映射：

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["novel_gpt"]

[tool.setuptools.package-dir]
novel_gpt = "src"
```

### 配置说明

| 配置项 | 作用 |
|--------|------|
| `packages = ["novel_gpt"]` | 声明要打包的模块名 |
| `package-dir.novel_gpt = "src"` | 将 `novel_gpt` 模块映射到 `src/` 目录 |

当运行 `uv sync` 安装包时，setuptools 会创建映射：`novel_gpt` 模块 → `src/` 目录。

## 验证

```bash
# 重新安装
uv sync

# 测试命令
uv run python -m novel_gpt --help
uv run python -m novel_gpt train --config mini --max_steps 1000
```

## 为什么使用 Src Layout

### 优点

1. **清晰的代码隔离**：源代码与项目其他文件（配置、文档、测试）分离
2. **避免导入混淆**：防止意外导入项目根目录而非已安装的包
3. **更好的包结构**：强制使用正确的包导入方式
4. **测试更可靠**：测试时使用的是已安装的包，而非源码目录

### 缺点

需要额外的构建配置（如上所述）

## 参考链接

- [Python Packaging - Src Layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- [Setuptools Package Discovery](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html)

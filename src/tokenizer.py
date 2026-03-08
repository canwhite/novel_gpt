"""
分词器模块 - 支持 tiktoken (GPT-2) 和字符级分词
"""

from abc import ABC, abstractmethod
from typing import List
import tiktoken


class Tokenizer(ABC):
    """分词器抽象基类"""
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass


class TiktokenTokenizer(Tokenizer):
    """tiktoken 分词器 - 使用 cl100k_base (支持中英文)"""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self._vocab_size = self.encoding.n_vocab
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text, allowed_special=set())
    
    def decode(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)


class CharTokenizer(Tokenizer):
    """字符级分词器 - 兼容原 novel.py"""
    
    def __init__(self, chars: str = None):
        if chars is None:
            # 默认可打印 ASCII 字符
            chars = "".join(chr(i) for i in range(32, 127)) + "\n\t"
        self.chars = chars
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self._vocab_size = len(chars)
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def decode(self, tokens: List[int]) -> str:
        return "".join(self.idx_to_char[i] for i in tokens if i in self.idx_to_char)


def get_tokenizer(tokenizer_type: str = "tiktoken") -> Tokenizer:
    """工厂函数 - 获取分词器实例"""
    if tokenizer_type == "tiktoken":
        return TiktokenTokenizer()
    elif tokenizer_type == "char":
        return CharTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


if __name__ == "__main__":
    # 测试
    text = "Hello, world! This is a test."
    
    tok = TiktokenTokenizer()
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)
    print(f"Tiktoken vocab: {tok.vocab_size}")
    print(f"Encoded: {tokens[:20]}...")
    print(f"Decoded: {decoded}")
    
    char_tok = CharTokenizer()
    char_tokens = char_tok.encode(text)
    char_decoded = char_tok.decode(char_tokens)
    print(f"\nChar vocab: {char_tok.vocab_size}")
    print(f"Encoded: {char_tokens}")
    print(f"Decoded: {char_decoded}")

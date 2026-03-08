"""
生成脚本 - 加载训练好的模型生成小说文本
"""

import argparse
from pathlib import Path

import torch

from .config import Config
from .model import GPT
from .tokenizer import TiktokenTokenizer


def generate(
    checkpoint_path: str,
    prompt: str = "",
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "auto",
) -> str:
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"Using device: {device}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model, step, _ = GPT.load_checkpoint(checkpoint_path, device)
    model.eval()
    print(f"Loaded model from step {step}")
    
    tokenizer = TiktokenTokenizer()
    
    if prompt:
        tokens = tokenizer.encode(prompt)
    else:
        tokens = [tokenizer.encoding.eot_token]
    
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    print(f"\nGenerating {max_new_tokens} tokens...")
    print("-" * 50)
    
    with torch.no_grad():
        output_idx = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    
    output_tokens = output_idx[0].tolist()
    text = tokenizer.decode(output_tokens)
    
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = text.replace('\ufffd', '')
    
    print(text)
    print("-" * 50)
    
    return text


def interactive_mode(checkpoint_path: str, device: str = "auto"):
    """交互式生成模式"""
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    model, step, _ = GPT.load_checkpoint(checkpoint_path, device)
    model.eval()
    tokenizer = TiktokenTokenizer()
    
    print(f"Loaded model from step {step}")
    print("Enter prompt (or 'quit' to exit):")
    
    while True:
        prompt = input("\n> ")
        if prompt.lower() in ["quit", "exit", "q"]:
            break
        
        if not prompt.strip():
            continue
        
        tokens = tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        
        with torch.no_grad():
            output_idx = model.generate(
                idx,
                max_new_tokens=200,
                temperature=0.8,
                top_k=40,
            )
        
        text = tokenizer.decode(output_idx[0].tolist())
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        text = text.replace('\ufffd', '')
        print("\n" + text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default="", help="Starting prompt")
    parser.add_argument("--max_tokens", type=int, default=500, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/mps/cuda/cpu)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.checkpoint, args.device)
    else:
        generate(
            args.checkpoint,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )


if __name__ == "__main__":
    main()

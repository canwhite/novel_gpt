#!/usr/bin/env python3
"""
novel_gpt 入口脚本
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="GPT Novel Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 快速测试（迷你配置）
  python -m novel_gpt train --config mini --max_steps 100

  # 标准训练
  python -m novel_gpt train --config default

  # 从检查点生成
  python -m novel_gpt generate checkpoints/best.pt --prompt "Once upon a time"

  # 交互式生成
  python -m novel_gpt generate checkpoints/best.pt --interactive
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train 子命令
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["mini", "default", "small"],
        help="Configuration preset",
    )
    train_parser.add_argument("--max_steps", type=int, help="Override max training steps")
    train_parser.add_argument("--batch_size", type=int, help="Override batch size")
    train_parser.add_argument("--device", type=str, help="Override device (auto/mps/cuda/cpu)")
    
    # Generate 子命令
    gen_parser = subparsers.add_parser("generate", help="Generate text from checkpoint")
    gen_parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    gen_parser.add_argument("--prompt", type=str, default="", help="Starting prompt")
    gen_parser.add_argument("--max_tokens", type=int, default=500, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    gen_parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    gen_parser.add_argument("--device", type=str, default="auto", help="Device")
    gen_parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    # Info 子命令
    info_parser = subparsers.add_parser("info", help="Show configuration info")
    info_parser.add_argument("--config", type=str, default="default", choices=["mini", "default", "small"])
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "train":
        from .train import main as train_main
        train_args = ["train"]
        for k, v in vars(args).items():
            if k == "command" or v is None:
                continue
            if isinstance(v, bool):
                if v:
                    train_args.append(f"--{k}")
            else:
                train_args.append(f"--{k}={v}")
        sys.argv = train_args
        train_main()
    
    elif args.command == "generate":
        from .generate import main as gen_main
        gen_args = ["generate", args.checkpoint]
        for k, v in vars(args).items():
            if k in ("command", "checkpoint") or v is None:
                continue
            if isinstance(v, bool):
                if v:
                    gen_args.append(f"--{k}")
            else:
                gen_args.append(f"--{k}={v}")
        sys.argv = gen_args
        gen_main()
    
    elif args.command == "info":
        from .config import Config, get_mini_config, get_small_config
        
        if args.config == "mini":
            config = get_mini_config()
        elif args.config == "small":
            config = get_small_config()
        else:
            config = Config()
        
        print(f"Configuration: {args.config}")
        print(f"Device: {config.device}")
        print(f"Model parameters: {config.model.param_count:,}")
        print(f"Model config:")
        print(f"  n_layer: {config.model.n_layer}")
        print(f"  n_embd: {config.model.n_embd}")
        print(f"  n_head: {config.model.n_head}")
        print(f"  block_size: {config.model.block_size}")
        print(f"  vocab_size: {config.model.vocab_size}")
        print(f"Training config:")
        print(f"  batch_size: {config.training.batch_size}")
        print(f"  max_steps: {config.training.max_steps}")
        print(f"  learning_rate: {config.training.learning_rate}")


if __name__ == "__main__":
    main()

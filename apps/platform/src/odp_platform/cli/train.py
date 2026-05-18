"""Training CLI."""

import argparse
from pathlib import Path

from odp_platform.config.train_config import TrainConfig
from odp_platform.config.loaders import load_train_config
from odp_platform.training.service import TrainingService


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train YOLO models")
    parser.add_argument("--config", type=Path, help="Path to training config YAML")
    parser.add_argument("--model", type=str, help="Model path or name")
    parser.add_argument("--data", type=Path, help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="Device (0, cpu, etc.)")
    parser.add_argument("--name", type=str, help="Run name")

    args = parser.parse_args()

    if args.config:
        config = load_train_config(args.config)
    else:
        config = TrainConfig()

    if args.model:
        config.model = args.model
    if args.data:
        config.data = args.data
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.imgsz:
        config.imgsz = args.imgsz
    if args.device:
        config.device = args.device
    if args.name:
        config.project_name = args.name

    service = TrainingService(config)
    save_dir = service.train()
    print(f"Training completed. Results: {save_dir}")


if __name__ == "__main__":
    main()

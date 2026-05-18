"""Validation CLI."""

import argparse
from pathlib import Path

from odp_platform.config.val_config import ValConfig
from odp_platform.config.loaders import load_val_config
from odp_platform.evaluation.service import EvaluationService


def main():
    """Main entry point for validation."""
    parser = argparse.ArgumentParser(description="Evaluate YOLO models")
    parser.add_argument("--config", type=Path, help="Path to validation config YAML")
    parser.add_argument("--model", type=str, help="Model weights path")
    parser.add_argument("--data", type=Path, help="Dataset YAML path")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="Device")
    parser.add_argument("--name", type=str, help="Run name")

    args = parser.parse_args()

    if args.config:
        config = load_val_config(args.config)
    else:
        config = ValConfig()

    if args.model:
        config.model = args.model
    if args.data:
        config.data = args.data
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.imgsz:
        config.imgsz = args.imgsz
    if args.device:
        config.device = args.device
    if args.name:
        config.project_name = args.name

    service = EvaluationService(config)
    metrics = service.evaluate()

    print("\nEvaluation Results:")
    print(f"  mAP50:    {metrics['map50']:.4f}")
    print(f"  mAP50-95: {metrics['map50_95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")


if __name__ == "__main__":
    main()

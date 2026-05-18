"""Inference CLI."""

import argparse
from pathlib import Path

from odp_platform.config.infer_config import InferConfig
from odp_platform.config.loaders import load_infer_config
from odp_platform.inference.service import InferenceService


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(description="Run YOLO inference")
    parser.add_argument("--config", type=Path, help="Path to inference config YAML")
    parser.add_argument("--model", type=str, help="Model weights path")
    parser.add_argument("--source", type=Path, help="Input images/videos directory")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    parser.add_argument("--iou", type=float, help="IoU threshold")
    parser.add_argument("--device", type=str, help="Device")
    parser.add_argument("--save-dir", type=Path, help="Output directory")

    args = parser.parse_args()

    if args.config:
        config = load_infer_config(args.config)
    else:
        config = InferConfig()

    if args.model:
        config.model = args.model
    if args.source:
        config.source = args.source
    if args.imgsz:
        config.imgsz = args.imgsz
    if args.conf:
        config.conf_threshold = args.conf
    if args.iou:
        config.iou_threshold = args.iou
    if args.device:
        config.device = args.device
    if args.save_dir:
        config.save_dir = args.save_dir

    service = InferenceService(config)
    results = service.predict()
    print(f"Inference completed. Processed {len(results)} images.")


if __name__ == "__main__":
    main()

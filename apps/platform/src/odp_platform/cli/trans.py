"""Data transformation CLI - VOC to YOLO conversion."""

import argparse
from pathlib import Path

from odp_platform.data_pipline.service import convert_voc_to_yolo, convert_coco_to_yolo


def main():
    """Main entry point for data transformation."""
    parser = argparse.ArgumentParser(description="Convert datasets to YOLO format")
    parser.add_argument("--format", choices=["voc", "coco"], required=True, help="Source format")
    parser.add_argument("--input", type=Path, required=True, help="Input directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--classes", nargs="+", required=True, help="Class names")
    parser.add_argument("--images", type=Path, help="Images directory (for COCO)")

    args = parser.parse_args()

    if args.format == "voc":
        convert_voc_to_yolo(args.input, args.output, args.classes)
    elif args.format == "coco":
        if not args.images:
            parser.error("--images is required for COCO format")
        convert_coco_to_yolo(args.input, args.output, args.images)

    print("Conversion completed successfully.")


if __name__ == "__main__":
    main()

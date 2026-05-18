"""Data pipeline service for VOC to YOLO conversion."""

from pathlib import Path

from odp_platform.data_pipline.core.pascal_voc import PascalVOCConverter
from odp_platform.data_pipline.core.coco import COCOConverter


def convert_voc_to_yolo(
    voc_dir: Path,
    output_dir: Path,
    class_names: list[str],
) -> None:
    """Convert Pascal VOC dataset to YOLO format."""
    converter = PascalVOCConverter(voc_dir, output_dir, class_names)
    converter.convert()


def convert_coco_to_yolo(
    coco_json_path: Path,
    output_dir: Path,
    images_dir: Path,
) -> None:
    """Convert COCO dataset to YOLO format."""
    converter = COCOConverter(coco_json_path, output_dir, images_dir)
    converter.convert()

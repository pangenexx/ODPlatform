"""COCO to YOLO format converter."""

import json
import shutil
from pathlib import Path


class COCOConverter:
    """Convert COCO dataset to YOLO format."""

    def __init__(self, coco_json_path: Path, output_dir: Path, images_dir: Path):
        self.coco_json_path = coco_json_path
        self.output_dir = output_dir
        self.images_dir = images_dir

    def convert(self) -> None:
        """Run the conversion process."""
        with open(self.coco_json_path, "r") as f:
            coco_data = json.load(f)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        class_map = {cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])}
        class_names = [cat["name"] for cat in coco_data["categories"]]

        img_map = {img["id"]: img for img in coco_data["images"]}

        annotations_by_image: dict[int, list] = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        for img_id, img_info in img_map.items():
            self._convert_image(img_info, annotations_by_image.get(img_id, []), class_map)

        self._generate_dataset_yaml(class_names)

    def _convert_image(self, img_info: dict, annotations: list, class_map: dict) -> None:
        """Convert a single COCO image to YOLO format."""
        filename = img_info["file_name"]
        img_width = img_info["width"]
        img_height = img_info["height"]

        yolo_lines = []
        for ann in annotations:
            class_id = class_map[ann["category_id"]]
            x, y, w, h = ann["bbox"]

            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if yolo_lines:
            src_img = self.images_dir / filename
            dst_img = self.output_dir / "images" / filename
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if src_img.exists():
                shutil.copy(src_img, dst_img)

            label_path = self.output_dir / "labels" / f"{Path(filename).stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")

    def _generate_dataset_yaml(self, class_names: list[str]) -> None:
        """Generate dataset.yaml for YOLO training."""
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write(f"path: {self.output_dir}\n")
            f.write("train: images\n")
            f.write("val: images\n\n")
            f.write(f"nc: {len(class_names)}\n")
            f.write(f"names: {class_names}\n")

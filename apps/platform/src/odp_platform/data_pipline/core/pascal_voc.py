"""Pascal VOC to YOLO format converter."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


class PascalVOCConverter:
    """Convert Pascal VOC dataset to YOLO format."""

    def __init__(self, voc_dir: Path, output_dir: Path, class_names: list[str]):
        self.voc_dir = voc_dir
        self.output_dir = output_dir
        self.class_names = class_names
        self.class_map = {name: idx for idx, name in enumerate(class_names)}

    def convert(self) -> None:
        """Run the conversion process."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        splits = ["train", "val", "test"]
        for split in splits:
            self._convert_split(split)

        self._generate_dataset_yaml()

    def _convert_split(self, split: str) -> None:
        """Convert a single split (train/val/test)."""
        annotations_dir = self.voc_dir / "Annotations"
        images_dir = self.voc_dir / "JPEGImages"
        split_file = self.voc_dir / f"{split}.txt"

        if not split_file.exists():
            return

        output_images_dir = self.output_dir / "images" / split
        output_labels_dir = self.output_dir / "labels" / split
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        with open(split_file, "r") as f:
            image_names = [line.strip() for line in f if line.strip()]

        for name in image_names:
            xml_path = annotations_dir / f"{name}.xml"
            if xml_path.exists():
                self._convert_annotation(xml_path, images_dir, output_images_dir, output_labels_dir)

    def _convert_annotation(
        self,
        xml_path: Path,
        images_dir: Path,
        output_images_dir: Path,
        output_labels_dir: Path,
    ) -> None:
        """Convert a single VOC annotation to YOLO format."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        yolo_lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in self.class_map:
                continue

            class_id = self.class_map[class_name]
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if yolo_lines:
            src_img = images_dir / filename
            dst_img = output_images_dir / filename
            if src_img.exists():
                shutil.copy(src_img, dst_img)

            label_path = output_labels_dir / f"{Path(filename).stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")

    def _generate_dataset_yaml(self) -> None:
        """Generate dataset.yaml for YOLO training."""
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write(f"path: {self.output_dir}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("test: images/test\n\n")
            f.write(f"nc: {len(self.class_names)}\n")
            f.write(f"names: {self.class_names}\n")

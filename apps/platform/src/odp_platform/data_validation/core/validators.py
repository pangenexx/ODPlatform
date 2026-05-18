"""Dataset validators."""

from pathlib import Path


class YOLODatasetValidator:
    """Validate YOLO format dataset."""

    def __init__(self, dataset_dir: Path, class_names: list[str]):
        self.dataset_dir = dataset_dir
        self.class_names = class_names
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate(self) -> dict:
        """Run all validation checks."""
        self._check_directory_structure()
        self._check_images()
        self._check_labels()
        self._check_label_format()

        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def _check_directory_structure(self) -> None:
        """Check required directory structure."""
        for split in ["train", "val"]:
            images_dir = self.dataset_dir / "images" / split
            labels_dir = self.dataset_dir / "labels" / split

            if not images_dir.exists():
                self.errors.append(f"Missing directory: {images_dir}")
            if not labels_dir.exists():
                self.errors.append(f"Missing directory: {labels_dir}")

    def _check_images(self) -> None:
        """Check image files exist and have valid extensions."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images_dir = self.dataset_dir / "images"

        for split in ["train", "val"]:
            split_dir = images_dir / split
            if not split_dir.exists():
                continue

            for img_path in split_dir.iterdir():
                if img_path.suffix.lower() not in valid_extensions:
                    self.warnings.append(f"Invalid image extension: {img_path}")

    def _check_labels(self) -> None:
        """Check label files match image files."""
        for split in ["train", "val"]:
            images_dir = self.dataset_dir / "images" / split
            labels_dir = self.dataset_dir / "labels" / split

            if not images_dir.exists() or not labels_dir.exists():
                continue

            for img_path in images_dir.iterdir():
                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    self.warnings.append(f"Missing label for image: {img_path.name}")

    def _check_label_format(self) -> None:
        """Check label file format."""
        labels_dir = self.dataset_dir / "labels"

        for split in ["train", "val"]:
            split_dir = labels_dir / split
            if not split_dir.exists():
                continue

            for label_path in split_dir.iterdir():
                self._validate_label_file(label_path)

    def _validate_label_file(self, label_path: Path) -> None:
        """Validate a single label file."""
        with open(label_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    self.errors.append(f"{label_path}:{line_num}: Expected 5 values, got {len(parts)}")
                    continue

                try:
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= len(self.class_names):
                        self.errors.append(f"{label_path}:{line_num}: Invalid class ID {class_id}")

                    for val in parts[1:]:
                        v = float(val)
                        if v < 0 or v > 1:
                            self.errors.append(f"{label_path}:{line_num}: Value {v} out of range [0, 1]")
                except ValueError:
                    self.errors.append(f"{label_path}:{line_num}: Invalid numeric value")

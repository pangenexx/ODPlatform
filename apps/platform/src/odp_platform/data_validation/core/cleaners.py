"""Dataset cleaners."""

import shutil
from pathlib import Path


class DatasetCleaner:
    """Clean YOLO dataset by removing invalid samples."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    def clean(self, output_dir: Path) -> Path:
        """Clean dataset and save to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for split in ["train", "val", "test"]:
            self._clean_split(split, output_dir)

        return output_dir

    def _clean_split(self, split: str, output_dir: Path) -> None:
        """Clean a single split."""
        images_dir = self.dataset_dir / "images" / split
        labels_dir = self.dataset_dir / "labels" / split

        if not images_dir.exists() or not labels_dir.exists():
            return

        out_images_dir = output_dir / "images" / split
        out_labels_dir = output_dir / "labels" / split
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        for label_path in labels_dir.glob("*.txt"):
            img_path = images_dir / f"{label_path.stem}.jpg"
            if not img_path.exists():
                img_path = images_dir / f"{label_path.stem}.png"

            if img_path.exists() and self._is_valid_label(label_path):
                shutil.copy(img_path, out_images_dir)
                shutil.copy(label_path, out_labels_dir)

    def _is_valid_label(self, label_path: Path) -> bool:
        """Check if a label file is valid."""
        try:
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        return False
                    int(parts[0])
                    for val in parts[1:]:
                        v = float(val)
                        if v < 0 or v > 1:
                            return False
            return True
        except (ValueError, IndexError):
            return False

"""Dataset analyzers."""

from pathlib import Path
from collections import Counter


class DatasetAnalyzer:
    """Analyze YOLO dataset statistics."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    def analyze(self) -> dict:
        """Run dataset analysis."""
        stats = {
            "splits": {},
            "total_images": 0,
            "total_annotations": 0,
            "class_distribution": Counter(),
        }

        for split in ["train", "val", "test"]:
            split_stats = self._analyze_split(split)
            if split_stats:
                stats["splits"][split] = split_stats
                stats["total_images"] += split_stats["image_count"]
                stats["total_annotations"] += split_stats["annotation_count"]
                stats["class_distribution"] += split_stats["class_distribution"]

        return stats

    def _analyze_split(self, split: str) -> dict | None:
        """Analyze a single split."""
        labels_dir = self.dataset_dir / "labels" / split
        if not labels_dir.exists():
            return None

        image_count = 0
        annotation_count = 0
        class_distribution = Counter()

        for label_path in labels_dir.glob("*.txt"):
            image_count += 1
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        annotation_count += 1
                        class_id = int(line.split()[0])
                        class_distribution[class_id] += 1

        return {
            "image_count": image_count,
            "annotation_count": annotation_count,
            "class_distribution": class_distribution,
        }

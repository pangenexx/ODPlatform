"""Chart generator for dataset visualization."""

from pathlib import Path


class ChartGenerator:
    """Generate charts for dataset analysis."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    def generate_class_distribution_plot(self, output_path: Path) -> None:
        """Generate class distribution bar chart."""
        try:
            import matplotlib.pyplot as plt
            from odp_platform.data_validation.core.analyzers import DatasetAnalyzer

            analyzer = DatasetAnalyzer(self.dataset_dir)
            stats = analyzer.analyze()

            if stats["class_distribution"]:
                classes = list(stats["class_distribution"].keys())
                counts = list(stats["class_distribution"].values())

                plt.figure(figsize=(10, 6))
                plt.bar(classes, counts)
                plt.xlabel("Class ID")
                plt.ylabel("Count")
                plt.title("Class Distribution")
                plt.savefig(output_path)
                plt.close()
        except ImportError:
            pass

    def generate_image_size_distribution(self, output_path: Path) -> None:
        """Generate image size distribution scatter plot."""
        try:
            import matplotlib.pyplot as plt
            import cv2

            widths = []
            heights = []

            for split in ["train", "val"]:
                images_dir = self.dataset_dir / "images" / split
                if images_dir.exists():
                    for img_path in images_dir.glob("*.jpg"):
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            h, w = img.shape[:2]
                            widths.append(w)
                            heights.append(h)

            if widths:
                plt.figure(figsize=(8, 6))
                plt.scatter(widths, heights, alpha=0.5)
                plt.xlabel("Width")
                plt.ylabel("Height")
                plt.title("Image Size Distribution")
                plt.savefig(output_path)
                plt.close()
        except ImportError:
            pass

    def generate_bbox_distribution(self, output_path: Path) -> None:
        """Generate bounding box size distribution histogram."""
        try:
            import matplotlib.pyplot as plt

            widths = []
            heights = []

            for split in ["train", "val"]:
                labels_dir = self.dataset_dir / "labels" / split
                if labels_dir.exists():
                    for label_path in labels_dir.glob("*.txt"):
                        with open(label_path, "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    widths.append(float(parts[3]))
                                    heights.append(float(parts[4]))

            if widths:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                ax1.hist(widths, bins=50, alpha=0.7)
                ax1.set_xlabel("Relative Width")
                ax1.set_ylabel("Count")
                ax1.set_title("Bounding Box Width Distribution")

                ax2.hist(heights, bins=50, alpha=0.7, color="orange")
                ax2.set_xlabel("Relative Height")
                ax2.set_ylabel("Count")
                ax2.set_title("Bounding Box Height Distribution")

                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
        except ImportError:
            pass

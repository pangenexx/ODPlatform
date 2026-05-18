"""Dataset visualizers."""

from pathlib import Path

from odp_platform.data_validation.core.chart_generator import ChartGenerator


class DatasetVisualizer:
    """Generate visualizations for dataset analysis."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    def generate_plots(self, output_dir: Path) -> None:
        """Generate all visualization plots."""
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = ChartGenerator(self.dataset_dir)
        generator.generate_class_distribution_plot(output_dir / "class_distribution.png")
        generator.generate_image_size_distribution(output_dir / "image_sizes.png")
        generator.generate_bbox_distribution(output_dir / "bbox_distribution.png")

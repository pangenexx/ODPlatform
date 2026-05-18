"""Data validation service."""

from pathlib import Path

from odp_platform.data_validation.core.validators import YOLODatasetValidator
from odp_platform.data_validation.core.analyzers import DatasetAnalyzer
from odp_platform.data_validation.core.cleaners import DatasetCleaner
from odp_platform.data_validation.core.visualizers import DatasetVisualizer
from odp_platform.data_validation.core.reporters import ReportGenerator


def validate_dataset(dataset_dir: Path, class_names: list[str]) -> dict:
    """Validate a YOLO dataset and return validation results."""
    validator = YOLODatasetValidator(dataset_dir, class_names)
    return validator.validate()


def analyze_dataset(dataset_dir: Path) -> dict:
    """Analyze dataset statistics."""
    analyzer = DatasetAnalyzer(dataset_dir)
    return analyzer.analyze()


def clean_dataset(dataset_dir: Path, output_dir: Path) -> Path:
    """Clean dataset by removing invalid samples."""
    cleaner = DatasetCleaner(dataset_dir)
    return cleaner.clean(output_dir)


def visualize_dataset(dataset_dir: Path, output_dir: Path) -> None:
    """Generate dataset visualizations."""
    visualizer = DatasetVisualizer(dataset_dir)
    visualizer.generate_plots(output_dir)


def generate_report(analysis_results: dict, output_path: Path) -> None:
    """Generate a validation report."""
    reporter = ReportGenerator(analysis_results)
    reporter.save_report(output_path)

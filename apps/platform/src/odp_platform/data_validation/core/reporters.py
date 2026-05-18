"""Report generators."""

import json
from pathlib import Path


class ReportGenerator:
    """Generate validation/analysis reports."""

    def __init__(self, results: dict):
        self.results = results

    def save_report(self, output_path: Path) -> None:
        """Save report to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".json":
            self._save_json(output_path)
        elif output_path.suffix == ".md":
            self._save_markdown(output_path)
        else:
            self._save_json(output_path)

    def _save_json(self, path: Path) -> None:
        """Save report as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)

    def _save_markdown(self, path: Path) -> None:
        """Save report as Markdown."""
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Dataset Validation Report\n\n")

            if "valid" in self.results:
                status = "Valid" if self.results["valid"] else "Invalid"
                f.write(f"**Status**: {status}\n\n")

            if "errors" in self.results and self.results["errors"]:
                f.write("## Errors\n\n")
                for error in self.results["errors"]:
                    f.write(f"- {error}\n")
                f.write("\n")

            if "warnings" in self.results and self.results["warnings"]:
                f.write("## Warnings\n\n")
                for warning in self.results["warnings"]:
                    f.write(f"- {warning}\n")
                f.write("\n")

            if "splits" in self.results:
                f.write("## Split Statistics\n\n")
                for split_name, split_stats in self.results["splits"].items():
                    f.write(f"### {split_name}\n\n")
                    f.write(f"- Images: {split_stats.get('image_count', 0)}\n")
                    f.write(f"- Annotations: {split_stats.get('annotation_count', 0)}\n\n")

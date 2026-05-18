"""Inference components for result processing."""

from pathlib import Path


class DetectionResult:
    """Wrapper for detection results."""

    def __init__(self, boxes: list, class_names: list[str]):
        self.boxes = boxes
        self.class_names = class_names

    def to_dict(self) -> list[dict]:
        """Convert results to dictionary format."""
        results = []
        for box in self.boxes:
            results.append({
                "class_id": int(box.cls),
                "class_name": self.class_names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0].tolist()],
            })
        return results


class ResultExporter:
    """Export inference results to various formats."""

    def __init__(self, results: list[DetectionResult]):
        self.results = results

    def export_json(self, output_path: Path) -> None:
        """Export results to JSON."""
        import json

        all_results = []
        for result in self.results:
            all_results.extend(result.to_dict())

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    def export_csv(self, output_path: Path) -> None:
        """Export results to CSV."""
        import csv

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "confidence", "xmin", "ymin", "xmax", "ymax"])

            for result in self.results:
                for det in result.to_dict():
                    writer.writerow([
                        det["class_id"],
                        det["class_name"],
                        f"{det['confidence']:.4f}",
                        *det["bbox"],
                    ])

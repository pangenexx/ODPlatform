"""Inference pipeline for batch processing."""

from pathlib import Path

from odp_platform.inference.service import InferenceService
from odp_platform.config.infer_config import InferConfig


class InferencePipeline:
    """Pipeline for batch inference with post-processing."""

    def __init__(self, config: InferConfig):
        self.service = InferenceService(config)

    def run(self, source: Path) -> list:
        """Run the full inference pipeline."""
        results = self.service.predict(source)
        return results

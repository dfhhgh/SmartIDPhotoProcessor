"""
JSON exporter for machine-readable evaluation results.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from evaluation.models import ImageEvaluationResult


class JsonExporter:
    """Exports individual image evaluation results to structured JSON files."""

    def __init__(self, output_dir: str) -> None:
        self._json_dir = os.path.join(output_dir, "json")
        os.makedirs(self._json_dir, exist_ok=True)

    def export(self, result: ImageEvaluationResult) -> str:
        """Export an evaluation result to a JSON file and return its path."""
        base_name = os.path.splitext(result.image_name)[0]
        json_path = os.path.join(self._json_dir, f"{base_name}_evaluation.json")
        
        data = asdict(result)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return json_path

"""
Utility functions for saving experiment outputs.

This module provides small helper functions for consistently writing research
metrics, text reports, and other experiment artifacts to disk.
"""

import json
from pathlib import Path


def ensure_parent_directory(path):
    """
    Create the parent directory for a file path if it does not already exist.
    """
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def save_metrics(metrics, output_path):
    """
    Save a dictionary of experiment metrics as formatted JSON.
    """
    output_path = ensure_parent_directory(output_path)

    with open(output_path, "w") as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(f"Saved metrics to {output_path}")


def save_text_report(report, output_path):
    """
    Save a plain-text experiment report.
    """
    output_path = ensure_parent_directory(output_path)

    with open(output_path, "w") as file:
        file.write(report)

    print(f"Saved report to {output_path}")
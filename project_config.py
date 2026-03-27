"""Project-wide configuration utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "master": {
        "host": "localhost",
        "port": 5000,
        "n_workers": 2,
        "n_epochs": 10,
        "lr": 0.01,
        "benchmark_matrix_size": 100,
    },
    "data": {
        "n_samples": 10000,
        "test_size": 0.1,
        "random_state": 42,
    },
    "workers": {
        "artificial_delays": [0.0, 2.0],
    },
    "logging": {
        "training_log_path": "logs/training_log.csv",
        "run_summary_path": "logs/last_run_summary.json",
    },
    "benchmark": {
        "sample_sizes": [10000, 15000, 20000],
        "repeats": 1,
        "master_startup_wait_sec": 1.5,
        "results_csv_path": "logs/benchmark_results.csv",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dictionaries recursively without mutating inputs."""
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config and merge with defaults.

    Missing file is allowed so the project can run with built-in defaults.
    """
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return deepcopy(DEFAULT_CONFIG)

    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("Top-level config content must be a dictionary.")

    return _deep_merge(DEFAULT_CONFIG, data)

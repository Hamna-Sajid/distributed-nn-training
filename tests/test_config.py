"""Tests for project-wide YAML configuration loading."""

from pathlib import Path

from project_config import DEFAULT_CONFIG, load_config


def test_load_config_uses_defaults_when_file_missing(tmp_path: Path):
    cfg = load_config(str(tmp_path / "missing.yaml"))

    assert cfg["data"]["n_samples"] == DEFAULT_CONFIG["data"]["n_samples"]
    assert cfg["master"]["n_workers"] == DEFAULT_CONFIG["master"]["n_workers"]


def test_load_config_merges_nested_overrides(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "\n".join(
            [
                "master:",
                "  n_workers: 4",
                "data:",
                "  n_samples: 25000",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_file))

    assert cfg["master"]["n_workers"] == 4
    assert cfg["data"]["n_samples"] == 25000
    # Unspecified keys should still be present from defaults.
    assert "lr" in cfg["master"]

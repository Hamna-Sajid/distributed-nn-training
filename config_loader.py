"""
config_loader.py — Central configuration reader.

Import CFG in any module to access settings from config.yaml.

Usage
-----
    from config_loader import CFG

    lr        = CFG["model"]["lr"]
    n_samples = CFG["dataset"]["n_samples"]
    host      = CFG["communication"]["host"]
"""

import os
import yaml


def load_config(path: str = None) -> dict:
    """Locate and parse config.yaml, searching up from this file's directory.

    Parameters
    ----------
    path : str or None
        Explicit path to config.yaml. If None, auto-discovers it by
        walking up the directory tree from this file's location.

    Returns
    -------
    dict
        Parsed YAML configuration as a nested dictionary.

    Raises
    ------
    FileNotFoundError
        If config.yaml cannot be found in any parent directory.
    """
    if path:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    search = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        candidate = os.path.join(search, "config.yaml")
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                return yaml.safe_load(f)
        parent = os.path.dirname(search)
        if parent == search:
            break
        search = parent

    raise FileNotFoundError(
        "config.yaml not found. Place it in the project root."
    )


# Module-level singleton
CFG = load_config()
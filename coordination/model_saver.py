"""
coordination/model_saver.py — Save and load trained model weights to disk.

Called at the end of Master.run_and_return_metrics() so the final model
is not lost when the process exits. Saves to the saved_models/ directory
which is mounted as a volume in Docker so files persist on the host.
"""

import os
import json
import datetime
import numpy as np
from config_loader import CFG


class ModelSaver:
    """Saves trained model weights and metadata to disk.

    Parameters
    ----------
    save_dir : str
        Directory to write model files into.
    """

    def __init__(self, save_dir: str = "saved_models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, model, metadata: dict = None) -> str:
        """Save model weights and run metadata to a timestamped JSON file.

        Parameters
        ----------
        model : MLP
            The trained model whose weights will be saved.
        metadata : dict or None
            Run metadata embedded alongside the weights.

        Returns
        -------
        str
            Full path to the saved JSON file.
        """
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        n_samp   = (metadata or {}).get("n_samples", "unknown")
        n_work   = (metadata or {}).get("n_workers", "unknown")
        filename = f"model_{ts}_{n_samp}samp_{n_work}workers.json"
        filepath = os.path.join(self.save_dir, filename)

        payload = {
            "saved_at":     datetime.datetime.now().isoformat(),
            "metadata":     metadata or {},
            "model_config": {
                "input_dim":  CFG["model"]["input_dim"],
                "hidden_dim": CFG["model"]["hidden_dim"],
                "output_dim": CFG["model"]["output_dim"],
                "lr":         CFG["model"]["lr"],
            },
            "weights": model.get_weights(),
        }

        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

        size_mb = os.path.getsize(filepath) / 1e6
        print(f"[ModelSaver] Saved -> {filepath}  ({size_mb:.1f} MB)")
        return filepath

    def save_npz(self, model, metadata: dict = None) -> str:
        """Save model weights in compact NumPy .npz format (~4x smaller than JSON).

        Parameters
        ----------
        model : MLP
            The trained model.
        metadata : dict or None
            Saved as JSON sidecar file.

        Returns
        -------
        str
            Full path to the .npz file.
        """
        ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        n_samp    = (metadata or {}).get("n_samples", "unknown")
        stem      = f"model_{ts}_{n_samp}samp"
        npz_path  = os.path.join(self.save_dir, stem + ".npz")
        meta_path = os.path.join(self.save_dir, stem + "_meta.json")

        flat = {}
        for layer, params in model.get_weights().items():
            for pname, val in params.items():
                flat[f"{layer}__{pname}"] = np.array(val)

        np.savez_compressed(npz_path, **flat)

        with open(meta_path, "w") as f:
            json.dump(metadata or {}, f, indent=2)

        size_mb = os.path.getsize(npz_path) / 1e6
        print(f"[ModelSaver] NPZ saved -> {npz_path}  ({size_mb:.1f} MB)")
        return npz_path

    def load(self, filepath: str) -> dict:
        """Load model weights from a saved JSON file.

        Parameters
        ----------
        filepath : str
            Path to a .json file saved by save().

        Returns
        -------
        dict
            Weights dict — pass directly to model.set_weights().
        """
        with open(filepath) as f:
            payload = json.load(f)
        print(f"[ModelSaver] Loaded from {filepath}")
        print(f"             Saved at:  {payload.get('saved_at')}")
        print(f"             Metadata:  {payload.get('metadata')}")
        return payload["weights"]

    def list_saved(self) -> list:
        """Return all saved model filenames sorted most-recent first."""
        if not os.path.exists(self.save_dir):
            return []
        files = [f for f in os.listdir(self.save_dir)
                 if f.startswith("model_") and f.endswith(".json")]
        return sorted(files, reverse=True)
"""
coordination/performance_report.py — Consolidated performance report.

Combines model evaluation, training history, worker stats, and system
metrics into one JSON file and a printed terminal summary.

Called at the end of Master.run_and_return_metrics(). The output file
is written to benchmark_results/ which is a Docker volume so it
persists on the host machine after the container stops.
"""

import os
import json
import datetime
import numpy as np
from config_loader import CFG


class PerformanceReporter:
    """Builds and saves a consolidated performance report for one training run.

    Parameters
    ----------
    n_samples : int
        Number of training samples used.
    n_epochs : int
        Number of epochs trained.
    n_workers : int
        Number of worker processes used.
    output_dir : str
        Directory to write the report JSON into.
    """

    def __init__(self, n_samples: int, n_epochs: int, n_workers: int,
                 output_dir: str = "benchmark_results"):
        self.n_samples  = n_samples
        self.n_epochs   = n_epochs
        self.n_workers  = n_workers
        self.output_dir = output_dir
        self.report     = {
            "generated_at": datetime.datetime.now().isoformat(),
            "run_config": {
                "n_samples":  n_samples,
                "n_epochs":   n_epochs,
                "n_workers":  n_workers,
                "model":      dict(CFG["model"]),
            },
            "training":   {},
            "evaluation": {},
            "workers":    {},
        }
        os.makedirs(output_dir, exist_ok=True)

    def add_training_history(self, epoch_times: list, avg_losses: list):
        """Record per-epoch timing and loss history.

        Parameters
        ----------
        epoch_times : list of float
            Wall-clock seconds per epoch.
        avg_losses : list of float
            Master average loss per epoch.
        """
        total = sum(epoch_times) if epoch_times else 0
        self.report["training"] = {
            "epoch_times_sec":    [round(t, 3) for t in epoch_times],
            "avg_losses":         [round(l, 6) for l in avg_losses],
            "total_time_sec":     round(total, 2),
            "avg_epoch_sec":      round(total / len(epoch_times), 2) if epoch_times else 0,
            "initial_loss":       round(avg_losses[0], 6)  if avg_losses else None,
            "final_loss":         round(avg_losses[-1], 6) if avg_losses else None,
            "loss_reduction_pct": round(
                (avg_losses[0] - avg_losses[-1]) / avg_losses[0] * 100, 2
            ) if avg_losses and avg_losses[0] > 0 else 0,
            "throughput_sps": round(
                (self.n_samples * self.n_epochs) / total, 2
            ) if total > 0 else 0,
        }

    def add_evaluation(self, y_pred: np.ndarray, y_true: np.ndarray,
                       test_loss: float):
        """Record test-set evaluation results.

        Parameters
        ----------
        y_pred : np.ndarray of shape (n_test, n_classes)
            Model predictions on the held-out test set.
        y_true : np.ndarray of shape (n_test, n_classes)
            Ground truth labels.
        test_loss : float
            BCE loss on the test set.
        """
        from tests.evaluate import compute_metrics
        metrics = compute_metrics(y_pred, y_true)

        # Per-class accuracy (what fraction of samples got class i right)
        preds     = (y_pred >= 0.5).astype(int)
        per_class = {}
        for i in range(y_true.shape[1]):
            correct = int((preds[:, i] == y_true[:, i]).sum())
            per_class[f"class_{i}"] = round(correct / y_true.shape[0], 4)

        self.report["evaluation"] = {
            "test_loss":          round(float(test_loss), 6),
            "accuracy":           metrics["accuracy"],
            "precision":          metrics["precision"],
            "recall":             metrics["recall"],
            "f1":                 metrics["f1"],
            "n_test_samples":     int(y_true.shape[0]),
            "per_class_accuracy": per_class,
        }

    def add_worker_stats(self, workers: dict):
        """Record per-worker RTT, batch size, and final loss.

        Parameters
        ----------
        workers : dict
            The master's self.workers dict.
        """
        stats = {}
        for wid, w in workers.items():
            stats[str(wid)] = {
                "rtt_sec":    w.get("rtt", 0.0),
                "batch_size": w.get("batch_size", 0),
                "final_loss": round(w.get("last_loss", 0.0), 6),
                "speed":      round(w.get("speed", 1.0), 4),
            }
        self.report["workers"] = stats

    def finalize(self) -> str:
        """Print terminal summary and save report JSON.

        Returns
        -------
        str
            Full path to the saved JSON report.
        """
        self._print_summary()
        return self._save_json()

    def _print_summary(self):
        t = self.report["training"]
        e = self.report["evaluation"]
        print("\n" + "=" * 55)
        print("  PERFORMANCE REPORT")
        print("=" * 55)
        print(f"  Dataset:    MNIST — {self.n_samples:,} samples")
        print(f"  Workers:    {self.n_workers}")
        print(f"  Epochs:     {self.n_epochs}")
        print(f"  Total time: {t.get('total_time_sec', '?')}s")
        print(f"  Avg epoch:  {t.get('avg_epoch_sec', '?')}s")
        print(f"  Throughput: {t.get('throughput_sps', '?')} samples/sec")
        print(f"  Loss:       {t.get('initial_loss')} → {t.get('final_loss')} "
              f"({t.get('loss_reduction_pct')}% reduction)")
        print(f"  Test loss:  {e.get('test_loss', '?')}")
        print(f"  Accuracy:   {e.get('accuracy', '?')}")
        print(f"  F1 Score:   {e.get('f1', '?')}")
        print("=" * 55 + "\n")

    def _save_json(self) -> str:
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = (f"report_{ts}_{self.n_samples}samp"
                 f"_{self.n_workers}workers.json")
        fpath = os.path.join(self.output_dir, fname)
        with open(fpath, "w") as f:
            json.dump(self.report, f, indent=2)
        print(f"[Report] Saved -> {fpath}")
        return fpath
"""
coordination/logger.py — Performance logger for distributed training.

Writes per-epoch metrics and wall-clock timing to a CSV file.
The log() method returns the epoch duration in seconds so the master
can collect timing data for the benchmark suite.

All paths are read from config.yaml automatically.
"""

import csv
import os
import time

from config_loader import CFG


class PerformanceLogger:
    """Logs per-epoch training metrics and timing to a structured CSV file.

    Parameters
    ----------
    filepath : str or None
        Path to the output CSV file. If None, uses
        config paths.logs_dir/training_log.csv.

    Attributes
    ----------
    filepath : str
        Resolved path where the CSV is written.
    start_time : float
        Unix timestamp at logger creation — for total elapsed time.
    epoch_start : float
        Timestamp set at the start of each epoch — for per-epoch duration.
    """

    def __init__(self, filepath: str = None):
        if filepath is None:
            logs_dir = CFG["paths"]["logs_dir"]
            filepath = os.path.join(logs_dir, "training_log.csv")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.filepath    = filepath
        self.start_time  = time.time()
        self.epoch_start = self.start_time

        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "avg_loss",
                "epoch_duration_sec",
                "total_elapsed_sec",
                "worker0_loss",
                "worker1_loss",
            ])
        print(f"[Logger] Writing to {self.filepath}")

    def start_epoch(self):
        """Mark the start of an epoch for duration measurement.

        Call this at the top of the training loop, before sending
        SYNCHRONIZE to workers, so the full round-trip time is captured.
        """
        self.epoch_start = time.time()

    def log(self, epoch: int, avg_loss: float,
            worker_losses: dict) -> float:
        """Append one row of metrics and return the epoch duration.

        Parameters
        ----------
        epoch : int
            Current epoch number (1-indexed).
        avg_loss : float
            Master-side weighted average loss for this epoch.
        worker_losses : dict
            Maps worker_id (int) to that worker's local loss (float).
            Missing worker IDs are logged as 0.0.

        Returns
        -------
        float
            Wall-clock duration of this epoch in seconds.
        """
        now            = time.time()
        epoch_duration = round(now - self.epoch_start, 3)
        total_elapsed  = round(now - self.start_time, 3)

        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(avg_loss, 6),
                epoch_duration,
                total_elapsed,
                round(worker_losses.get(0, 0.0), 6),
                round(worker_losses.get(1, 0.0), 6),
            ])

        print(f"[Master] Epoch {epoch:>3} | "
              f"Loss: {avg_loss:.4f} | "
              f"Epoch: {epoch_duration:.1f}s | "
              f"Total: {total_elapsed:.1f}s")

        return epoch_duration
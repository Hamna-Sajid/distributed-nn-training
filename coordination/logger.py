"""
Performance logger for distributed training.

Writes per-epoch metrics including wall-clock timing to a CSV file.
With large models like MNIST + 2048 neurons, timing data is essential
to demonstrate that serialized code is slow — and that parallelism helps.
"""

import csv
import os
import time


class PerformanceLogger:
    """Logs per-epoch training metrics and timing to a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the output CSV file.

    Attributes
    ----------
    filepath : str
        Path where CSV is written.
    start_time : float
        Timestamp at logger creation — used for total elapsed time.
    epoch_start : float
        Timestamp set at the beginning of each epoch — used for per-epoch duration.
    """

    def __init__(
        self,
        filepath: str = "logs/training_log.csv",
        worker_ids: list = None,
        append: bool = False,
    ):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath
        self.start_time = time.time()
        self.epoch_start = self.start_time
        self.worker_ids = sorted(worker_ids or [0, 1])

        mode = "a" if append else "w"
        should_write_header = (not append) or (not os.path.exists(filepath))

        with open(filepath, mode, newline="") as f:
            writer = csv.writer(f)
            if should_write_header:
                header = [
                    "epoch",
                    "avg_loss",
                    "epoch_duration_sec",   # how long THIS epoch took
                    "total_elapsed_sec",    # total time since training started
                ] + [f"worker{wid}_loss" for wid in self.worker_ids]
                writer.writerow(header)
        print(f"[Logger] Writing metrics to {filepath}")

    def start_epoch(self):
        """Call this at the START of each epoch to begin timing it.

        Should be called by the master right before sending SYNCHRONIZE
        to workers, so the full epoch duration is captured.
        """
        self.epoch_start = time.time()

    def log(self, epoch: int, avg_loss: float, worker_losses: dict):
        """Append one row of metrics to the CSV file.

        Parameters
        ----------
        epoch : int
            Current epoch number (1-indexed).
        avg_loss : float
            Master-side weighted average loss for this epoch.
        worker_losses : dict
            Maps worker_id (int) to that worker's local loss (float).
        """
        now = time.time()
        epoch_duration = round(now - self.epoch_start, 3)
        total_elapsed  = round(now - self.start_time, 3)

        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            row = [
                epoch,
                round(avg_loss, 6),
                epoch_duration,
                total_elapsed,
            ] + [round(worker_losses.get(wid, 0.0), 6) for wid in self.worker_ids]
            writer.writerow(row)
        print(f"[Master] Epoch {epoch} | Loss: {avg_loss:.4f} | "
              f"Epoch time: {epoch_duration:.1f}s | Total: {total_elapsed:.1f}s")
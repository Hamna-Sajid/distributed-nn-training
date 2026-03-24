import csv, os, time

class PerformanceLogger:
    """Logs per-epoch training metrics to a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the output CSV file.
    """
    def __init__(self, filepath: str = "logs/training_log.csv"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath
        self.start_time = time.time()
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_loss", "elapsed_sec",
                             "worker0_loss", "worker1_loss"])

    def log(self, epoch: int, avg_loss: float,
            worker_losses: dict):
        """Append one row to the log file.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        avg_loss : float
            Master-side averaged loss for this epoch.
        worker_losses : dict
            Maps worker_id (int) to that worker's local loss (float).
        """
        elapsed = round(time.time() - self.start_time, 3)
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, round(avg_loss, 6), elapsed,
                round(worker_losses.get(0, 0), 6),
                round(worker_losses.get(1, 0), 6),
            ])
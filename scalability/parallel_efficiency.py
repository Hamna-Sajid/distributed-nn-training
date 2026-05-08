"""
scalability/parallel_efficiency.py — Member A: Parallel Efficiency Enhancement.

Optimizes computation distribution across nodes to improve scalability.

This module implements:
1. Mini-batch gradient descent — workers process data in small batches
   per epoch instead of the full shard, reducing memory pressure and
   enabling more frequent gradient updates.
2. Gradient accumulation — accumulate gradients over N mini-batches
   before sending, reducing communication frequency.
3. Parallel efficiency calculator — computes speedup and efficiency
   metrics for reporting.

Usage
-----
    from scalability.parallel_efficiency import MiniBatchScheduler, EfficiencyCalculator
"""

import numpy as np
import time
from config_loader import CFG


class MiniBatchScheduler:
    """Schedules mini-batch training for a worker's local data shard.

    Splits the worker's data shard into mini-batches and yields them
    one at a time. Gradients are accumulated across mini-batches before
    being sent to the master, reducing communication overhead.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Local data shard for this worker.
    y : np.ndarray of shape (n_samples, n_classes)
        Corresponding labels.
    batch_size : int or None
        Mini-batch size. None = use full shard (original M1/M2 behaviour).
    shuffle : bool
        Whether to shuffle data before each epoch.

    Attributes
    ----------
    n_batches : int
        Number of mini-batches per epoch.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 batch_size: int = None, shuffle: bool = True):
        self.X          = X
        self.y          = y
        self.batch_size = batch_size or X.shape[0]
        self.shuffle    = shuffle
        self.n_batches  = max(1, int(np.ceil(X.shape[0] / self.batch_size)))

    def epoch_batches(self):
        """Yield (X_batch, y_batch) pairs for one epoch.

        Yields
        ------
        tuple of (np.ndarray, np.ndarray)
            Mini-batch feature matrix and label matrix.
        """
        n = self.X.shape[0]
        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_idx = indices[start:end]
            yield self.X[batch_idx], self.y[batch_idx]

    def accumulate_gradients(self, model) -> tuple:
        """Run all mini-batches and accumulate gradients for this epoch.

        Computes gradients on each mini-batch, accumulates them with
        proper normalization by batch size, and returns the final
        averaged gradient dict plus the mean loss across batches.

        Parameters
        ----------
        model : MLP
            The local model to run forward/backward on.

        Returns
        -------
        accumulated_grads : dict
            Averaged gradient dict in the same structure as MLP.backward().
        mean_loss : float
            Average loss across all mini-batches this epoch.
        total_samples : int
            Total samples processed (= shard size).
        """
        accumulated = None
        total_loss  = 0.0
        n_batches   = 0

        for X_batch, y_batch in self.epoch_batches():
            y_pred = model.forward(X_batch)
            loss   = model.compute_loss(y_pred, y_batch)
            grads  = model.backward(y_pred, y_batch)
            total_loss += loss
            n_batches  += 1

            batch_weight = X_batch.shape[0] / self.X.shape[0]

            if accumulated is None:
                accumulated = {
                    layer: {p: np.array(v) * batch_weight
                            for p, v in params.items()}
                    for layer, params in grads.items()
                }
            else:
                for layer in accumulated:
                    for p in accumulated[layer]:
                        accumulated[layer][p] += np.array(grads[layer][p]) * batch_weight

        mean_loss = total_loss / max(n_batches, 1)
        return accumulated, mean_loss, self.X.shape[0]


class EfficiencyCalculator:
    """Computes parallel speedup and efficiency metrics.

    Used by Member C's benchmarking to measure scalability trends
    as the number of workers changes.

    Parameters
    ----------
    serial_time_sec : float
        Wall-clock time for the same workload on a single worker (baseline).

    Attributes
    ----------
    serial_time : float
        Stored serial baseline time.
    records : list of dict
        History of all efficiency measurements.
    """

    def __init__(self, serial_time_sec: float):
        self.serial_time = serial_time_sec
        self.records     = []

    def record(self, n_workers: int, parallel_time_sec: float,
               n_samples: int, n_epochs: int) -> dict:
        """Compute and store efficiency metrics for one parallel run.

        Speedup    S(p) = T_serial / T_parallel
        Efficiency E(p) = S(p) / p   (ideal = 1.0)
        Throughput       = (n_samples * n_epochs) / T_parallel

        Parameters
        ----------
        n_workers : int
            Number of parallel workers used.
        parallel_time_sec : float
            Wall-clock time for the parallel run.
        n_samples : int
            Total training samples used.
        n_epochs : int
            Number of epochs trained.

        Returns
        -------
        dict
            Keys: n_workers, speedup, efficiency, throughput_sps,
            parallel_time_sec, serial_time_sec.
        """
        speedup    = round(self.serial_time / parallel_time_sec, 4) if parallel_time_sec > 0 else 0
        efficiency = round(speedup / n_workers, 4) if n_workers > 0 else 0
        throughput = round((n_samples * n_epochs) / parallel_time_sec, 2) if parallel_time_sec > 0 else 0

        record = {
            "n_workers":         n_workers,
            "speedup":           speedup,
            "efficiency":        efficiency,
            "throughput_sps":    throughput,
            "parallel_time_sec": parallel_time_sec,
            "serial_time_sec":   self.serial_time,
        }
        self.records.append(record)

        print(f"[Efficiency] Workers={n_workers} | "
              f"Speedup={speedup:.3f}x | "
              f"Efficiency={efficiency:.3f} | "
              f"Throughput={throughput:.1f} sps")
        return record

    def get_summary(self) -> list:
        """Return all recorded efficiency measurements.

        Returns
        -------
        list of dict
            All records in insertion order.
        """
        return self.records

    def print_table(self):
        """Print a formatted efficiency table to stdout."""
        print("\n" + "=" * 60)
        print(f"  {'Workers':>8} {'Speedup':>10} {'Efficiency':>12} {'Throughput':>12}")
        print("=" * 60)
        for r in self.records:
            print(f"  {r['n_workers']:>8} {r['speedup']:>10.3f} "
                  f"{r['efficiency']:>12.3f} {r['throughput_sps']:>12.1f}")
        print("=" * 60 + "\n")
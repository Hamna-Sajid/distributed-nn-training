"""
coordination/master.py — Master node for synchronous distributed training.

All default settings are read from config.yaml via config_loader.
Individual parameters can be overridden in the constructor so that
benchmark_suite.py can sweep different sample sizes without editing config.

After training, run_and_return_metrics() returns a dict of all key results
so benchmark_suite.py can save them to JSON, CSV, and Excel.

M3 notes
--------
- _training_loop now returns (epoch_times, avg_losses) as a tuple
- apply_gradients receives {"layer1": {"dW":..., "db":...}} structure
  which is what AdaptiveAggregator produces
- AdaptiveAggregator.aggregate() returns (aggregated, worker_weights)
  where aggregated has the same layer/dW/db structure as MLP.backward()
"""

import socket
import time
import numpy as np

from communication.protocol import (
    encode_message, decode_message,
    recv_gradient_message,
    MSG_REGISTER, MSG_BENCHMARK, MSG_BENCH_RESULT,
    MSG_DATA_SHARD, MSG_GRADIENT, MSG_MODEL_UPDATE,
    MSG_SYNCHRONIZE, MSG_DONE
)
from neural_network.mlp import MLP
from data.loader import generate_dataset, partition_data
from coordination.logger import PerformanceLogger
from tests.evaluate import compute_metrics
from config_loader import CFG

from coordination.sync_barrier import SyncBarrier
from coordination.adaptive_aggregator import AdaptiveAggregator


class Master:
    """Coordinates distributed training across multiple worker nodes.

    Default values come from config.yaml. Pass explicit arguments to
    override specific settings (used by benchmark_suite.py).

    Parameters
    ----------
    n_workers : int or None
        Number of workers to wait for. None = read from config.
    host : str or None
        Bind address. None = read from config.
    port : int or None
        TCP port. None = read from config.
    n_epochs : int or None
        Training epochs. None = read from config.
    lr : float or None
        Learning rate. None = read from config.
    n_samples : int or None
        MNIST samples to load. None = read from config.

    Attributes
    ----------
    workers : dict
        Maps worker_id (int) to a dict with keys:
        conn, speed, rtt, batch_size, last_loss.
    model : MLP
        The global model maintained by the master.
    """

    def __init__(self, n_workers=None, host=None, port=None,
                 n_epochs=None, lr=None, n_samples=None):
        self.n_workers = n_workers if n_workers is not None else CFG["training"]["n_workers"]
        self.host      = host      or CFG["communication"]["host"]
        self.port      = port      or CFG["communication"]["port"]
        self.n_epochs  = n_epochs  if n_epochs  is not None else CFG["training"]["n_epochs"]
        self.lr        = lr        if lr        is not None else CFG["model"]["lr"]
        self.n_samples = n_samples if n_samples is not None else CFG["dataset"]["n_samples"]
        self.workers   = {}
        self.model     = MLP(lr=self.lr)

    def run(self):
        """Start master for a normal single run.

        Use this from run_master.py. For benchmarking use
        run_and_return_metrics() so results can be collected.
        """
        self.run_and_return_metrics()

    def run_and_return_metrics(self) -> dict:
        """Run full training and return a metrics dictionary.

        Returns
        -------
        dict
            Keys: final_loss, test_loss, accuracy, f1, precision, recall,
            worker0_final_loss, worker1_final_loss, worker0_samples,
            worker1_samples, worker0_rtt, worker1_rtt, epoch_times.
        """
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(self.n_workers)
        print(f"\n[Master] Listening on {self.host}:{self.port} | "
              f"{self.n_samples:,} samples | {self.n_epochs} epochs")

        # ── accept workers ────────────────────────────────────────────────────
        for _ in range(self.n_workers):
            conn, addr = server.accept()
            msg = decode_message(conn)
            assert msg["type"] == MSG_REGISTER
            wid = msg["data"]["worker_id"]
            self.workers[wid] = {
                "conn":       conn,
                "speed":      1.0,
                "rtt":        0.0,
                "batch_size": 0,
                "last_loss":  0.0,
            }
            print(f"[Master] Worker {wid} connected from {addr}")

        # ── benchmark ─────────────────────────────────────────────────────────
        self._benchmark_workers()

        # ── load dataset and partition ────────────────────────────────────────
        X_train, X_test, y_train, y_test = generate_dataset(
            n_samples=self.n_samples
        )
        speed_weights = [
            self.workers[i]["speed"] for i in sorted(self.workers)
        ]
        shards = partition_data(X_train, y_train, self.n_workers, speed_weights)

        for i, (X_shard, _) in enumerate(shards):
            self.workers[i]["batch_size"] = X_shard.shape[0]

        # ── send data + initial weights to each worker ────────────────────────
        initial_weights = self.model.get_weights()
        for wid, (X_shard, y_shard) in enumerate(shards):
            payload = {
                "X":        X_shard.tolist(),
                "y":        y_shard.tolist(),
                "weights":  initial_weights,
                "lr":       self.lr,
                "n_epochs": self.n_epochs,
            }
            self.workers[wid]["conn"].sendall(
                encode_message(MSG_DATA_SHARD, 0, payload)
            )
            print(f"[Master] Shard sent -> Worker {wid} "
                  f"({X_shard.shape[0]:,} samples)")

        # ── training loop ─────────────────────────────────────────────────────
        epoch_times, avg_losses = self._training_loop()

        # ── evaluate ──────────────────────────────────────────────────────────
        y_pred    = self.model.forward(X_test)
        test_loss = self.model.compute_loss(y_pred, y_test)
        metrics   = compute_metrics(y_pred, y_test)
        print(f"\n[Master] Test loss: {test_loss:.4f} | {metrics}")

        # ── shut down workers ─────────────────────────────────────────────────
        for w in self.workers.values():
            w["conn"].sendall(encode_message(MSG_DONE, 0, {}))
            w["conn"].close()
        server.close()

        return {
            "final_loss":         round(avg_losses[-1], 6) if avg_losses else None,
            "test_loss":          round(float(test_loss), 6),
            "accuracy":           metrics["accuracy"],
            "f1":                 metrics["f1"],
            "precision":          metrics["precision"],
            "recall":             metrics["recall"],
            "worker0_final_loss": round(self.workers.get(0, {}).get("last_loss", 0.0), 6),
            "worker1_final_loss": round(self.workers.get(1, {}).get("last_loss", 0.0), 6),
            "worker0_samples":    self.workers.get(0, {}).get("batch_size", 0),
            "worker1_samples":    self.workers.get(1, {}).get("batch_size", 0),
            "worker0_rtt":        round(self.workers.get(0, {}).get("rtt", 0.0), 4),
            "worker1_rtt":        round(self.workers.get(1, {}).get("rtt", 0.0), 4),
            "epoch_times":        epoch_times,
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _benchmark_workers(self):
        """Send a 100x100 matrix to each worker and measure RTT.

        RTT determines relative speed ratios used for proportional
        data partitioning. Stored in self.workers[id]['rtt'].
        """
        bench_data = np.random.randn(100, 100).tolist()
        times = {}

        for wid, w in self.workers.items():
            w["conn"].sendall(
                encode_message(MSG_BENCHMARK, 0, {"matrix": bench_data})
            )
            t0 = time.time()
            msg = decode_message(w["conn"])
            assert msg["type"] == MSG_BENCH_RESULT
            elapsed    = time.time() - t0
            times[wid] = elapsed
            w["rtt"]   = round(elapsed, 4)
            print(f"[Master] Worker {wid} RTT: {elapsed:.4f}s")

        min_time = max(min(times.values()), 1e-6)
        for wid in self.workers:
            self.workers[wid]["speed"] = min_time / times[wid]

        print(f"[Master] Speed ratios: "
              f"{ {k: round(v['speed'], 4) for k, v in self.workers.items()} }")

    def _training_loop(self):
        """Run synchronous gradient aggregation for n_epochs.

        Uses SyncBarrier to collect gradients from all workers concurrently,
        and AdaptiveAggregator to apply norm clipping and staleness penalties.

        The aggregated gradient structure from AdaptiveAggregator is:
            {"layer1": {"dW": ndarray, "db": ndarray}, ...}

        This is converted to the structure MLP.apply_gradients() expects:
            {"layer1": {"dW": ndarray, "db": ndarray}, ...}
        which is identical — no conversion needed.

        Returns
        -------
        epoch_times : list of float
            Wall-clock seconds per epoch.
        avg_losses : list of float
            Master average loss per epoch.
        """
        logger      = PerformanceLogger()
        epoch_times = []
        avg_losses  = []

        barrier = SyncBarrier(
            n_workers=len(self.workers),
            timeout_sec=CFG["communication"]["timeout_sec"]
        )
        aggregator = AdaptiveAggregator(
            norm_clip_threshold=10.0,
            staleness_penalty=0.5
        )

        for epoch in range(self.n_epochs):
            logger.start_epoch()

            try:
                # 1. Broadcast SYNCHRONIZE to all workers
                for w in self.workers.values():
                    w["conn"].sendall(
                        encode_message(MSG_SYNCHRONIZE, 0, {"epoch": epoch})
                    )

                # 2. Collect gradients concurrently via SyncBarrier
                barrier_result = barrier.collect_gradients(
                    {wid: w["conn"] for wid, w in self.workers.items()},
                    recv_gradient_message
                )

                # 3. Adaptive aggregation
                #    aggregator.aggregate() returns:
                #      agg_gradients: {"layer1": {"dW":..,"db":..}, ...}
                #      worker_weights: {worker_id: float}
                agg_gradients, worker_weights = aggregator.aggregate(
                    gradients=barrier_result["gradients"],
                    batch_sizes=barrier_result["batch_sizes"],
                    losses=barrier_result["losses"],
                    arrival_times=barrier_result["timing"],
                )

                # 4. Update last_loss per worker for reporting
                for wid, loss in barrier_result["losses"].items():
                    self.workers[wid]["last_loss"] = loss

                # 5. Apply aggregated gradients to global model
                #    MLP.apply_gradients expects the same structure
                #    AdaptiveAggregator already produces.
                self.model.apply_gradients(agg_gradients)

                # 6. Broadcast updated weights to all workers
                updated_weights = self.model.get_weights()
                for w in self.workers.values():
                    w["conn"].sendall(
                        encode_message(MSG_MODEL_UPDATE, 0,
                                       {"weights": updated_weights})
                    )

                # 7. Log metrics
                avg_loss = aggregator.history[-1]["avg_loss"]
                avg_losses.append(avg_loss)

                duration = logger.log(
                    epoch + 1, avg_loss,
                    {wid: loss for wid, loss in barrier_result["losses"].items()}
                )
                epoch_times.append(duration)

                print(f"[Master] Epoch {epoch+1}/{self.n_epochs} | "
                      f"loss: {avg_loss:.4f} | "
                      f"time: {duration:.2f}s | "
                      f"weights: { {k: round(v, 4) for k, v in worker_weights.items()} }")

            except Exception as e:
                print(f"[Master] Error in epoch {epoch + 1}: "
                      f"{type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                raise

        return epoch_times, avg_losses
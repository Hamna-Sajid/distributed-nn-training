"""
coordination/master.py — Master node for synchronous distributed training.

  - Docker-aware: reads MASTER_HOST from environment variable
  - Post-training: saves final model to saved_models/
  - Post-training: generates performance report to benchmark_results/
  - run_and_return_metrics() used by both run_master.py and benchmarking.py
"""

import os
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
from coordination.model_saver import ModelSaver
from coordination.performance_report import PerformanceReporter
from tests.evaluate import compute_metrics
from config_loader import CFG
from coordination.sync_barrier import SyncBarrier
from coordination.adaptive_aggregator import AdaptiveAggregator


class Master:
    """Coordinates distributed training across multiple worker nodes.

    Docker-aware: MASTER_HOST env var overrides config.yaml host so the
    master binds to 0.0.0.0 inside Docker and localhost locally.

    Parameters
    ----------
    n_workers : int or None
        Number of workers to wait for. None = read from config.
    host : str or None
        Bind address. None = read from env var, then config.
    port : int or None
        TCP port. None = read from config.
    n_epochs : int or None
        Training epochs. None = read from config.
    lr : float or None
        Learning rate. None = read from config.
    n_samples : int or None
        MNIST samples to load. None = read from config.
    """

    def __init__(self, n_workers=None, host=None, port=None,
                 n_epochs=None, lr=None, n_samples=None):
        self.n_workers = (n_workers if n_workers is not None
                          else CFG["training"]["n_workers"])
        # Docker sets MASTER_HOST=0.0.0.0; local runs use localhost
        self.host      = (host
                          or os.environ.get("MASTER_HOST")
                          or CFG["communication"]["host"])
        self.port      = (port
                          or int(os.environ.get("MASTER_PORT",
                                 CFG["communication"]["port"])))
        self.n_epochs  = (n_epochs if n_epochs is not None
                          else CFG["training"]["n_epochs"])
        self.lr        = (lr if lr is not None
                          else CFG["model"]["lr"])
        self.n_samples = (n_samples if n_samples is not None
                          else CFG["dataset"]["n_samples"])
        self.workers   = {}
        self.model     = MLP(lr=self.lr)

    def run(self):
        """Start master for a normal single run (from run_master.py)."""
        self.run_and_return_metrics()

    def run_and_return_metrics(self) -> dict:
        """Run full training, save model, generate report, return metrics.

        Returns
        -------
        dict
            All training and evaluation metrics.
        """
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(self.n_workers)
        print(f"\n[Master] Listening on {self.host}:{self.port} | "
              f"{self.n_samples:,} samples | {self.n_epochs} epochs | "
              f"{self.n_workers} workers", flush=True)

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

        # ── send data + initial weights ───────────────────────────────────────
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

        # ── evaluate on test set ──────────────────────────────────────────────
        y_pred    = self.model.forward(X_test)
        test_loss = self.model.compute_loss(y_pred, y_test)
        metrics   = compute_metrics(y_pred, y_test)
        print(f"\n[Master] Test loss: {test_loss:.4f} | {metrics}")

        # ── save final model ──────────────────────────────────────────────────
        saver      = ModelSaver(save_dir="saved_models")
        model_meta = {
            "n_samples":  self.n_samples,
            "n_epochs":   self.n_epochs,
            "n_workers":  self.n_workers,
            "final_loss": round(avg_losses[-1], 6) if avg_losses else None,
            "test_loss":  round(float(test_loss), 6),
            "accuracy":   metrics["accuracy"],
        }
        saver.save(self.model, metadata=model_meta)
        saver.save_npz(self.model, metadata=model_meta)

        # ── generate performance report ───────────────────────────────────────
        reporter = PerformanceReporter(
            n_samples=self.n_samples,
            n_epochs=self.n_epochs,
            n_workers=self.n_workers,
        )
        reporter.add_training_history(epoch_times, avg_losses)
        reporter.add_evaluation(y_pred, y_test, test_loss)
        reporter.add_worker_stats(self.workers)
        reporter.finalize()

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
        """Send 100x100 matrix to each worker, measure RTT, compute speed ratios."""
        bench_data = np.random.randn(100, 100).tolist()
        times = {}

        for wid in sorted(self.workers.keys()):
            w = self.workers[wid]
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
        """Synchronous gradient aggregation using SyncBarrier + AdaptiveAggregator.

        Returns
        -------
        epoch_times : list of float
        avg_losses  : list of float
        """
        logger      = PerformanceLogger()
        epoch_times = []
        avg_losses  = []

        barrier    = SyncBarrier(
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
                for w in self.workers.values():
                    w["conn"].sendall(
                        encode_message(MSG_SYNCHRONIZE, 0, {"epoch": epoch})
                    )

                barrier_result = barrier.collect_gradients(
                    {wid: w["conn"] for wid, w in self.workers.items()},
                    recv_gradient_message
                )

                agg_gradients, worker_weights = aggregator.aggregate(
                    gradients=barrier_result["gradients"],
                    batch_sizes=barrier_result["batch_sizes"],
                    losses=barrier_result["losses"],
                    arrival_times=barrier_result["timing"],
                )

                for wid, loss in barrier_result["losses"].items():
                    self.workers[wid]["last_loss"] = loss

                self.model.apply_gradients(agg_gradients)

                updated_weights = self.model.get_weights()
                for w in self.workers.values():
                    w["conn"].sendall(
                        encode_message(MSG_MODEL_UPDATE, 0,
                                       {"weights": updated_weights})
                    )

                avg_loss = aggregator.history[-1]["avg_loss"]
                avg_losses.append(avg_loss)

                duration = logger.log(
                    epoch + 1, avg_loss,
                    {wid: loss for wid, loss in barrier_result["losses"].items()}
                )
                epoch_times.append(duration)

                print(f"[Master] Epoch {epoch+1}/{self.n_epochs} | "
                      f"loss: {avg_loss:.4f} | time: {duration:.2f}s | "
                      f"weights: { {k: round(v,4) for k,v in worker_weights.items()} }")

            except Exception as e:
                print(f"[Master] Error in epoch {epoch+1}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                raise

        return epoch_times, avg_losses
"""
coordination/master.py — Master node for synchronous distributed training.

All default settings are read from config.yaml via config_loader.
Individual parameters can be overridden in the constructor so that
benchmark_suite.py can sweep different sample sizes without editing config.

After training, run_and_return_metrics() returns a dict of all key results
so benchmark_suite.py can save them to JSON, CSV, and Excel.
"""

import socket
import time
import numpy as np

from communication.protocol import (
    encode_message, decode_message,
    MSG_REGISTER, MSG_BENCHMARK, MSG_BENCH_RESULT,
    MSG_DATA_SHARD, MSG_GRADIENT, MSG_MODEL_UPDATE,
    MSG_SYNCHRONIZE, MSG_DONE
)
from neural_network.mlp import MLP
from data.loader import generate_dataset, partition_data
from coordination.logger import PerformanceLogger
from tests.evaluate import compute_metrics
from config_loader import CFG


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
                "conn": conn, "speed": 1.0,
                "rtt": 0.0, "batch_size": 0, "last_loss": 0.0,
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
                "X": X_shard.tolist(), "y": y_shard.tolist(),
                "weights": initial_weights,
                "lr": self.lr, "n_epochs": self.n_epochs,
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

        min_time = min(times.values())
        for wid in self.workers:
            self.workers[wid]["speed"] = min_time / times[wid]

        print(f"[Master] Speed ratios: "
              f"{ {k: round(v['speed'], 4) for k, v in self.workers.items()} }")

    def _training_loop(self):
        """Run synchronous gradient aggregation for n_epochs.

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

        for epoch in range(self.n_epochs):
            logger.start_epoch()

            for w in self.workers.values():
                w["conn"].sendall(
                    encode_message(MSG_SYNCHRONIZE, 0, {"epoch": epoch})
                )

            all_gradients = {}
            for wid, w in self.workers.items():
                msg = decode_message(w["conn"])
                assert msg["type"] == MSG_GRADIENT
                all_gradients[wid] = {
                    "gradients":  msg["data"]["gradients"],
                    "batch_size": msg["data"]["batch_size"],
                    "loss":       msg["data"]["loss"],
                }
                self.workers[wid]["last_loss"] = msg["data"]["loss"]

            avg_loss = float(np.mean(
                [v["loss"] for v in all_gradients.values()]
            ))
            avg_losses.append(avg_loss)

            averaged = self._aggregate_gradients(all_gradients)
            self.model.apply_gradients(averaged)

            updated_weights = self.model.get_weights()
            for w in self.workers.values():
                w["conn"].sendall(
                    encode_message(MSG_MODEL_UPDATE, 0,
                                   {"weights": updated_weights})
                )

            duration = logger.log(
                epoch + 1, avg_loss,
                {wid: v["loss"] for wid, v in all_gradients.items()}
            )
            epoch_times.append(duration)

        return epoch_times, avg_losses

    def _aggregate_gradients(self, all_gradients: dict) -> dict:
        """Compute batch-size-weighted average of gradients across workers.

        Parameters
        ----------
        all_gradients : dict
            Maps worker_id -> {gradients, batch_size, loss}.

        Returns
        -------
        dict
            Averaged gradients in the same structure as MLP.backward().
        """
        total    = sum(v["batch_size"] for v in all_gradients.values())
        averaged = None

        for wid, data in all_gradients.items():
            weight = data["batch_size"] / total
            grads  = data["gradients"]

            if averaged is None:
                averaged = {
                    layer: {
                        p: np.array(val) * weight
                        for p, val in params.items()
                    }
                    for layer, params in grads.items()
                }
            else:
                for layer in averaged:
                    for p in averaged[layer]:
                        averaged[layer][p] += (
                            np.array(grads[layer][p]) * weight
                        )

        return averaged
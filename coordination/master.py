"""
Master node for synchronous distributed training.

The master is responsible for:
  1. Accepting worker connections and benchmarking their speed
  2. Partitioning the dataset proportionally to worker speed
  3. Broadcasting initial model weights and data shards
  4. Collecting gradients from all workers each iteration
  5. Computing the weighted average gradient
  6. Broadcasting updated model weights back to workers
"""

import socket
import threading
import time
import numpy as np

from coordination.logger import PerformanceLogger
from tests.evaluate import compute_metrics
from communication.protocol import (
    encode_message, decode_message,
    MSG_REGISTER, MSG_BENCHMARK, MSG_BENCH_RESULT,
    MSG_DATA_SHARD, MSG_GRADIENT, MSG_MODEL_UPDATE,
    MSG_SYNCHRONIZE, MSG_DONE
)
from neural_network.mlp import MLP
from data.loader import generate_dataset, partition_data


class Master:
    """Coordinates distributed training across multiple worker nodes.

    Parameters
    ----------
    n_workers : int
        Number of worker processes to wait for before training begins.
    host : str
        IP address or hostname to bind the master's server socket.
    port : int
        TCP port to listen on for incoming worker connections.
    n_epochs : int
        Number of full training epochs to run.
    lr : float
        Learning rate passed to the MLP and workers.

    Attributes
    ----------
    workers : dict
        Maps worker_id (int) → {'conn': socket, 'speed': float, 'batch_size': int}
    model : MLP
        The global model maintained by the master.
    """

    def __init__(self, n_workers: int = 2, host: str = "localhost",
                 port: int = 5000, n_epochs: int = 20, lr: float = 0.01):
        self.n_workers = n_workers
        self.host = host
        self.port = port
        self.n_epochs = n_epochs
        self.lr = lr
        self.workers = {}
        self.model = MLP(lr=lr)

    def run(self):
        """Start the master: accept workers, train, shut down.

        This is the main entry point. It sequentially:
        - Accepts n_workers TCP connections
        - Benchmarks each worker
        - Loads and partitions the dataset
        - Runs the distributed training loop
        - Sends DONE to all workers
        """
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(self.n_workers)
        print(f"[Master] Listening on {self.host}:{self.port} ...")

        # Accept all workers sequentially for simplicity
        for _ in range(self.n_workers):
            conn, addr = server.accept()
            msg = decode_message(conn)
            assert msg["type"] == MSG_REGISTER
            worker_id = msg["data"]["worker_id"]
            self.workers[worker_id] = {"conn": conn, "speed": 1.0}
            print(f"[Master] Worker {worker_id} connected from {addr}")

        # Benchmark each worker to measure relative speed
        self._benchmark_workers()

        # Load dataset and partition proportionally to speed
        X_train, X_test, y_train, y_test = generate_dataset()
        speed_weights = [self.workers[i]["speed"] for i in sorted(self.workers)]
        shards = partition_data(X_train, y_train, self.n_workers, speed_weights)

        # Assign batch sizes based on shard sizes
        for i, (X_shard, _) in enumerate(shards):
            self.workers[i]["batch_size"] = X_shard.shape[0]

        # Send initial weights + data shard to each worker
        initial_weights = self.model.get_weights()
        for worker_id, (X_shard, y_shard) in enumerate(shards):
            payload = {
                "X": X_shard.tolist(),
                "y": y_shard.tolist(),
                "weights": initial_weights,
                "lr": self.lr,
                "n_epochs": self.n_epochs,
            }
            self.workers[worker_id]["conn"].sendall(
                encode_message(MSG_DATA_SHARD, 0, payload)
            )
            print(f"[Master] Sent shard of size {X_shard.shape[0]} to worker {worker_id}")

        # Run training loop
        self._training_loop()

        # Evaluate on test set
        y_pred = self.model.forward(X_test)
        test_loss = self.model.compute_loss(y_pred, y_test)
        print(f"\n[Master] Final test loss: {test_loss:.4f}")
        metrics = compute_metrics(y_pred, y_test)   
        print(f"[Master] Evaluation metrics: {metrics}")

        # Shut down workers
        for worker_id, w in self.workers.items():
            w["conn"].sendall(encode_message(MSG_DONE, 0, {}))
            w["conn"].close()
        server.close()

    def _benchmark_workers(self):
        """Send a small matrix task to each worker and measure round-trip time.

        The master sends a 100x100 random matrix to each worker, asks it to
        compute the column sums, and records the elapsed time. Relative speeds
        are computed as the inverse of elapsed time, normalized so the fastest
        worker has speed 1.0.
        """
        times = {}
        bench_data = np.random.randn(100, 100).tolist()

        for worker_id, w in self.workers.items():
            w["conn"].sendall(
                encode_message(MSG_BENCHMARK, 0, {"matrix": bench_data})
            )
            t_start = time.time()
            msg = decode_message(w["conn"])
            assert msg["type"] == MSG_BENCH_RESULT
            times[worker_id] = time.time() - t_start
            print(f"[Master] Worker {worker_id} benchmark RTT: {times[worker_id]:.4f}s")

        # Normalize: faster worker gets proportionally more data
        min_time = min(times.values())
        for worker_id in self.workers:
            self.workers[worker_id]["speed"] = min_time / times[worker_id]
        print(f"[Master] Speed ratios: { {k: round(v['speed'], 3) for k, v in self.workers.items()} }")

    def _training_loop(self):
        """Run synchronous gradient aggregation for n_epochs.

        Each epoch:
          1. Send SYNCHRONIZE to all workers (start signal)
          2. Collect one GRADIENT message from every worker
          3. Compute weighted average gradient
          4. Update global model
          5. Broadcast updated weights via MODEL_UPDATE
        """
        logger = PerformanceLogger()

        for epoch in range(self.n_epochs):
            # Signal all workers to start this epoch
            for w in self.workers.values():
                w["conn"].sendall(encode_message(MSG_SYNCHRONIZE, 0, {"epoch": epoch}))

            # Collect gradients from all workers
            all_gradients = {}
            for worker_id, w in self.workers.items():
                msg = decode_message(w["conn"])
                assert msg["type"] == MSG_GRADIENT
                all_gradients[worker_id] = {
                    "gradients": msg["data"]["gradients"],
                    "batch_size": msg["data"]["batch_size"],
                    "loss": msg["data"]["loss"],
                }

            # Log per-epoch losses
            avg_loss = np.mean([v["loss"] for v in all_gradients.values()])
            print(f"[Master] Epoch {epoch+1}/{self.n_epochs} | Avg Loss: {avg_loss:.4f}")

            logger.log(epoch + 1, avg_loss, {wid: v["loss"] for wid, v in all_gradients.items()})

            # Weighted gradient average (weight = batch size)
            averaged = self._aggregate_gradients(all_gradients)

            # Update global model
            self.model.apply_gradients(averaged)

            # Broadcast updated weights to all workers
            updated_weights = self.model.get_weights()
            for w in self.workers.values():
                w["conn"].sendall(
                    encode_message(MSG_MODEL_UPDATE, 0, {"weights": updated_weights})
                )

    def _aggregate_gradients(self, all_gradients: dict) -> dict:
        """Compute a batch-size-weighted average of gradients across workers.

        Weighted average formula:
            g_avg = sum(batch_i * g_i) / sum(batch_i)

        Parameters
        ----------
        all_gradients : dict
            Maps worker_id → {'gradients': dict, 'batch_size': int, 'loss': float}

        Returns
        -------
        dict
            Averaged gradients in the same nested structure as MLP.backward().
        """
        total_samples = sum(v["batch_size"] for v in all_gradients.values())
        averaged = None

        for worker_id, data in all_gradients.items():
            weight = data["batch_size"] / total_samples
            grads = data["gradients"]

            if averaged is None:
                averaged = {
                    layer: {
                        param: np.array(val) * weight
                        for param, val in params.items()
                    }
                    for layer, params in grads.items()
                }
            else:
                for layer in averaged:
                    for param in averaged[layer]:
                        averaged[layer][param] += np.array(grads[layer][param]) * weight

        return averaged
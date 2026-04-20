"""
Worker node for synchronous distributed training.

Each worker:
  1. Connects to the master and registers with its worker ID
  2. Completes a benchmark task
  3. Receives its data shard and initial model weights
  4. Loops: waits for SYNCHRONIZE, computes gradients, sends them back,
     then receives and applies updated model weights
"""

import socket
import time
import numpy as np

from communication.protocol import (
    encode_message, decode_message,
    send_gradient_message, recv_gradient_message,
    MSG_REGISTER, MSG_BENCHMARK, MSG_BENCH_RESULT,
    MSG_DATA_SHARD, MSG_GRADIENT, MSG_MODEL_UPDATE,
    MSG_SYNCHRONIZE, MSG_DONE
)
from neural_network.mlp import MLP


class Worker:
    """Performs local training and participates in gradient synchronization.

    Parameters
    ----------
    worker_id : int
        Unique integer ID for this worker (0-indexed).
    master_host : str
        Hostname or IP address of the master node.
    master_port : int
        TCP port the master is listening on.
    artificial_delay : float
        Seconds to sleep before each backward pass, simulating a slow node.
        Set to 0.0 for no delay (default). Use to test heterogeneity handling.

    Attributes
    ----------
    conn : socket.socket
        TCP connection to the master node.
    model : MLP or None
        Local copy of the model, initialized after receiving DATA_SHARD.
    X : np.ndarray or None
        Local data shard feature matrix.
    y : np.ndarray or None
        Local data shard label matrix.
    """

    def __init__(self, worker_id: int, master_host: str = "localhost",
                 master_port: int = 5000, artificial_delay: float = 0.0):
        self.worker_id = worker_id
        self.master_host = master_host
        self.master_port = master_port
        self.artificial_delay = artificial_delay
        self.conn = None
        self.model = None
        self.X = None
        self.y = None

    def run(self):
        """Connect to master and run the full worker lifecycle.

        Lifecycle steps:
          1. Open TCP connection to master
          2. Send REGISTER message
          3. Handle benchmark task
          4. Receive data shard and initialize local model
          5. Training loop: SYNCHRONIZE → compute → send GRADIENT → receive update
          6. Terminate on DONE
        """
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((self.master_host, self.master_port))
        print(f"[Worker {self.worker_id}] Connected to master.")

        # Register with master
        self.conn.sendall(
            encode_message(MSG_REGISTER, self.worker_id, {"worker_id": self.worker_id})
        )

        # Complete benchmark
        self._handle_benchmark()

        # Receive dataset shard and initial weights
        msg = decode_message(self.conn)
        assert msg["type"] == MSG_DATA_SHARD
        self.X = np.array(msg["data"]["X"], dtype=np.float32)
        self.y = np.array(msg["data"]["y"], dtype=np.float32)
        lr = msg["data"]["lr"]
        n_epochs = msg["data"]["n_epochs"]
        self.model = MLP(
            input_dim=self.X.shape[1],
            output_dim=self.y.shape[1],
            lr=lr
        )
        self.model.set_weights(msg["data"]["weights"])
        print(f"[Worker {self.worker_id}] Received shard: {self.X.shape}, "
              f"starting {n_epochs} epochs.")

        # Training loop
        self._training_loop(n_epochs)
        self.conn.close()
        print(f"[Worker {self.worker_id}] Shutting down.")

    def _handle_benchmark(self):
        """Receive a matrix from master, compute column sums, return result.

        The master times the round-trip to estimate this worker's speed.
        Artificial delay is applied here so slow-node simulation affects
        the benchmark, causing the master to assign fewer samples.
        """
        msg = decode_message(self.conn)
        assert msg["type"] == MSG_BENCHMARK

        # Simulate slow node before responding
        if self.artificial_delay > 0:
            print(f"[Worker {self.worker_id}] Simulating delay: {self.artificial_delay}s")
            time.sleep(self.artificial_delay)

        matrix = np.array(msg["data"]["matrix"])
        result = np.sum(matrix, axis=0).tolist()

        self.conn.sendall(
            encode_message(MSG_BENCH_RESULT, self.worker_id, {"result": result})
        )

    def _training_loop(self, n_epochs: int):
        """Wait for SYNCHRONIZE, compute gradients, send to master, apply update.

        Parameters
        ----------
        n_epochs : int
            Number of training epochs, must match master's n_epochs.
        """
        for epoch in range(n_epochs):
            try:
                # Wait for master to signal start of this epoch
                msg = decode_message(self.conn)

                if msg["type"] == MSG_DONE:
                    print(f"[Worker {self.worker_id}] Received DONE early, stopping.")
                    return

                assert msg["type"] == MSG_SYNCHRONIZE

                # Optional artificial delay to simulate slow computation
                if self.artificial_delay > 0:
                    time.sleep(self.artificial_delay)

                # Forward + backward pass on local shard
                y_pred = self.model.forward(self.X)
                loss = self.model.compute_loss(y_pred, self.y)
                gradients = self.model.backward(y_pred, self.y)

                print(f"[Worker {self.worker_id}] Epoch {epoch+1} | Loss: {loss:.4f}")

                # Send gradients and metadata to master (with optional compression)
                from config_loader import load_config
                config = load_config()
                compression_config = config.get("communication", {}).get("compression", {})
                
                send_gradient_message(
                    self.conn, self.worker_id,
                    gradients=gradients,
                    batch_size=self.X.shape[0],
                    loss=loss,
                    compression_enabled=compression_config.get("enabled", False),
                    compression_type=compression_config.get("type", "int8"),
                    log_stats=compression_config.get("log_stats", False)
                )

                # Receive updated model weights from master
                msg = decode_message(self.conn)
                if msg["type"] == MSG_DONE:
                    print(f"[Worker {self.worker_id}] Received DONE, stopping.")
                    return
                assert msg["type"] == MSG_MODEL_UPDATE
                self.model.set_weights(msg["data"]["weights"])
            except ConnectionError as e:
                print(f"[Worker {self.worker_id}] Connection error in epoch {epoch+1}: {e}")
                raise
            except Exception as e:
                print(f"[Worker {self.worker_id}] Unexpected error in epoch {epoch+1}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                raise
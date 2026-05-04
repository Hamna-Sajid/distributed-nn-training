"""
Socket communication protocol for the distributed training system.

All messages are length-prefixed JSON frames. A 4-byte big-endian integer
header precedes each message body, allowing the receiver to know exactly
how many bytes to read regardless of TCP packet boundaries.

Message types
-------------
REGISTER    : Worker → Master  — worker announces itself on connection
BENCHMARK   : Master → Worker  — small task to measure worker speed
BENCH_RESULT: Worker → Master  — worker returns benchmark timing
DATA_SHARD  : Master → Worker  — initial dataset shard + model weights
GRADIENT    : Worker → Master  — local gradients after backward pass (uncompressed)
MSG_GRADIENT_COMPRESSED: Worker → Master  — local gradients (int8 quantized)
MODEL_UPDATE: Master → Worker  — averaged weights after aggregation
SYNCHRONIZE : Master → Worker  — barrier signal to begin next iteration
DONE        : Master → Worker  — training complete, worker may shut down
"""

import json
import struct
import socket
import numpy as np
import time


# ---------------------------------------------------------------------------
# Message type constants
# ---------------------------------------------------------------------------
MSG_REGISTER = "REGISTER"
MSG_BENCHMARK = "BENCHMARK"
MSG_BENCH_RESULT = "BENCH_RESULT"
MSG_DATA_SHARD = "DATA_SHARD"
MSG_GRADIENT = "GRADIENT"
MSG_GRADIENT_COMPRESSED = "GRADIENT_COMPRESSED"
MSG_MODEL_UPDATE = "MODEL_UPDATE"
MSG_SYNCHRONIZE = "SYNCHRONIZE"
MSG_DONE = "DONE"

# Global compression stats (for monitoring)
_compression_stats = {
    "total_messages": 0,
    "total_original_bytes": 0,
    "total_compressed_bytes": 0,
}


def get_compression_stats() -> dict:
    """Get cumulative compression statistics."""
    return dict(_compression_stats)



def _numpy_to_lists(obj):
    """Recursively convert numpy arrays to nested Python lists for JSON.

    Parameters
    ----------
    obj : any
        Object to convert. Can be a dict, list, np.ndarray, or scalar.

    Returns
    -------
    any
        Same structure with np.ndarrays replaced by nested lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _numpy_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_lists(v) for v in obj]
    return obj


def encode_message(msg_type: str, sender_id: int, payload: dict) -> bytes:
    """Encode a message into a length-prefixed byte frame.

    Frame layout:
        [4 bytes: body length (big-endian uint32)] [body: UTF-8 JSON]

    Parameters
    ----------
    msg_type : str
        One of the MSG_* constants defined in this module.
    sender_id : int
        Numeric ID of the sending node (0 = master, 1+ = workers).
    payload : dict
        Arbitrary JSON-serializable data to include in the message.

    Returns
    -------
    bytes
        Complete framed message ready to be sent over a socket.
    """
    envelope = {
        "type": msg_type,
        "sender": sender_id,
        "data": _numpy_to_lists(payload)
    }
    body = json.dumps(envelope).encode("utf-8")
    header = struct.pack(">I", len(body))
    return header + body


def decode_message(sock: socket.socket) -> dict:
    """Read and decode one message from a blocking socket.

    Reads the 4-byte length header first, then reads exactly that many
    bytes for the body, handling partial TCP reads correctly.

    Parameters
    ----------
    sock : socket.socket
        A connected TCP socket to read from.

    Returns
    -------
    dict
        Decoded message with keys 'type', 'sender', and 'data'.

    Raises
    ------
    ConnectionError
        If the socket closes before a complete message is received.
    """
    # Read exactly 4 bytes for the header
    raw_len = _recv_exact(sock, 4)
    if not raw_len:
        raise ConnectionError("Socket closed before header received.")
    length = struct.unpack(">I", raw_len)[0]

    # Read exactly `length` bytes for the body
    body = _recv_exact(sock, length)
    if not body:
        raise ConnectionError("Socket closed mid-message.")

    return json.loads(body.decode("utf-8"))


def _recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
    """Read exactly n_bytes from a socket, handling partial reads.

    TCP does not guarantee that a single recv() call returns all requested
    bytes, so this function loops until exactly n_bytes have been read.

    Parameters
    ----------
    sock : socket.socket
        Connected TCP socket to read from.
    n_bytes : int
        Exact number of bytes to receive.

    Returns
    -------
    bytes
        Exactly n_bytes of data, or empty bytes if the connection closed.
    """
    buf = b""
    while len(buf) < n_bytes:
        chunk = sock.recv(n_bytes - len(buf))
        if not chunk:
            return b""
        buf += chunk
    return buf


# =============================================================================
# Gradient Compression Helpers (Communication Optimization)
# =============================================================================

def send_gradient_message(sock: socket.socket, worker_id: int, gradients: dict,
                         batch_size: int, loss: float, compression_enabled: bool = False,
                         compression_type: str = 'int8', log_stats: bool = False) -> None:
    """Send gradient message with optional compression.
    
    Parameters
    ----------
    sock : socket.socket
        Connected socket to send to
    worker_id : int
        ID of sending worker
    gradients : dict
        Gradient dictionary {"layer1": {"dW": ndarray, ...}, ...}
    batch_size : int
        Batch size for this gradient
    loss : float
        Training loss for this batch
    compression_enabled : bool
        Whether to compress gradients
    compression_type : str
        Type of compression ('int8', 'int16', 'none')
    log_stats : bool
        Whether to log compression statistics
    """
    if not compression_enabled or compression_type == 'none':
        # Send uncompressed
        payload = {
            "gradients": gradients,
            "batch_size": batch_size,
            "loss": loss,
            "compressed": False
        }
        msg = encode_message(MSG_GRADIENT, worker_id, payload)
    else:
        # Compress and send
        from communication.compression import compress_gradients
        
        compressed_grad = compress_gradients(gradients, compression_type=compression_type)
        
        # Track stats
        original_bytes = sum(
            np.array(v["quantized"]).nbytes * 4  # Approximate original float32 size
            for layer in compressed_grad["layers"].values()
            for v in layer.values()
        )
        compressed_bytes = sum(
            np.array(v["quantized"]).nbytes
            for layer in compressed_grad["layers"].values()
            for v in layer.values()
        )
        
        _compression_stats["total_messages"] += 1
        _compression_stats["total_original_bytes"] += original_bytes
        _compression_stats["total_compressed_bytes"] += compressed_bytes
        
        if log_stats:
            ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
            print(f"[Compression] Worker {worker_id}: {original_bytes:,} → {compressed_bytes:,} bytes ({ratio:.2f}x)")
        
        payload = {
            "gradients": compressed_grad,
            "batch_size": batch_size,
            "loss": loss,
            "compressed": True,
            "compression_type": compression_type
        }
        msg = encode_message(MSG_GRADIENT_COMPRESSED, worker_id, payload)
    
    sock.sendall(msg)


def recv_gradient_message(sock: socket.socket) -> dict:
    """Receive gradient message and automatically decompress if needed.
    
    Returns
    -------
    dict
        Dictionary with keys: "worker_id", "gradients", "batch_size", "loss"
    """
    msg = decode_message(sock)
    
    msg_type = msg["type"]
    worker_id = msg["sender"]
    data = msg["data"]
    
    # Handle both compressed and uncompressed
    if msg_type == MSG_GRADIENT_COMPRESSED and data.get("compressed", False):
        from communication.compression import decompress_gradients
        
        # Decompress
        gradients = decompress_gradients(data["gradients"])
    elif msg_type == MSG_GRADIENT or not data.get("compressed", False):
        # Uncompressed
        gradients = data["gradients"]
    else:
        raise ValueError(f"Unknown gradient message type: {msg_type}")
    
    return {
        "worker_id": worker_id,
        "gradients": gradients,
        "batch_size": data["batch_size"],
        "loss": data["loss"]
    }
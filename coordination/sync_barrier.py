# coordination/sync_barrier.py
import time
import threading

class SyncBarrier:
    """
    Synchronous barrier that waits for all workers to submit gradients
    before the master proceeds with aggregation.
    
    Supports per-worker timeout tracking, which feeds into adaptive aggregation.
    """

    def __init__(self, n_workers, timeout_sec=300):
        self.n_workers = n_workers
        self.timeout_sec = timeout_sec

        # These are populated by collect_gradients()
        self._gradients = {}       # worker_id -> gradient dict
        self._batch_sizes = {}     # worker_id -> int
        self._losses = {}          # worker_id -> float
        self._arrival_times = {}   # worker_id -> float (seconds since epoch start)

    def collect_gradients(self, worker_connections, recv_message_fn):
        """
        Wait for GRADIENT messages from all workers (compressed or uncompressed).
        
        Args:
            worker_connections: dict of {worker_id: socket}
            recv_message_fn: the protocol.recv_gradient_message function (handles compression)
        
        Returns:
            dict with keys 'gradients', 'batch_sizes', 'losses', 'timing'
            Returns None for any worker that timed out.
        """
        self._gradients = {}
        self._batch_sizes = {}
        self._losses = {}
        self._arrival_times = {}

        barrier_start = time.time()
        results = {}

        # Use threads so we receive from all workers concurrently
        # (important: if we receive sequentially, the slow worker
        #  blocks us even from getting the fast worker's gradients)
        threads = []
        lock = threading.Lock()

        def recv_from_worker(worker_id, conn):
            try:
                # recv_gradient_message handles both compressed and uncompressed
                grad_msg = recv_message_fn(conn)
                arrival = time.time() - barrier_start
                with lock:
                    self._gradients[worker_id] = grad_msg['gradients']
                    self._batch_sizes[worker_id] = grad_msg['batch_size']
                    self._losses[worker_id] = grad_msg['loss']
                    self._arrival_times[worker_id] = arrival
            except Exception as e:
                print(f"[SyncBarrier] Worker {worker_id} failed: {e}")
                with lock:
                    self._gradients[worker_id] = None  # mark as failed

        for wid, conn in worker_connections.items():
            t = threading.Thread(target=recv_from_worker, args=(wid, conn))
            t.daemon = True
            threads.append(t)
            t.start()

        # Wait for all threads with a global timeout
        deadline = barrier_start + self.timeout_sec
        for t in threads:
            remaining = deadline - time.time()
            if remaining <= 0:
                print(f"[SyncBarrier] Timeout reached, some workers may not have completed")
                break
            t.join(timeout=remaining)

        # Report how long we waited (for monitoring)
        total_wait = time.time() - barrier_start
        print(f"[SyncBarrier] All workers received in {total_wait:.3f}s")
        if self._arrival_times:
            fastest = min(self._arrival_times.values())
            slowest = max(self._arrival_times.values())
            print(f"[SyncBarrier] Fastest: {fastest:.3f}s, Slowest: {slowest:.3f}s, "
                  f"Straggler lag: {slowest - fastest:.3f}s")

        return {
            'gradients': self._gradients,
            'batch_sizes': self._batch_sizes,
            'losses': self._losses,
            'timing': self._arrival_times
        }
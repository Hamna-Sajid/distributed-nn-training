"""
Entry point to start a worker node.

Usage
-----
    python run_worker.py <worker_id> [artificial_delay]

Examples
--------
    python run_worker.py 0          # fast worker, no delay
    python run_worker.py 1 2.0      # slow worker, 2s artificial delay
"""
import sys
from coordination.worker import Worker

if __name__ == "__main__":
    worker_id = int(sys.argv[1])
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    worker = Worker(worker_id=worker_id, artificial_delay=delay)
    worker.run()
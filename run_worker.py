"""
run_worker.py — Entry point to start a worker node.

Docker-aware: reads WORKER_ID, WORKER_DELAY, MASTER_HOST, MASTER_PORT
from environment variables. Command-line arguments override env vars
so local testing still works exactly as before.

Usage
-----
    # Local (same as before):
    python run_worker.py 0
    python run_worker.py 1 2.0

    # Docker (docker-compose sets env vars automatically):
    docker-compose up
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coordination.worker import Worker
from config_loader import CFG

if __name__ == "__main__":
    # Command-line args take priority, then env vars, then config.yaml
    worker_id   = int(sys.argv[1]) if len(sys.argv) > 1 \
                  else int(os.environ.get("WORKER_ID", 0))

    delay       = float(sys.argv[2]) if len(sys.argv) > 2 \
                  else float(os.environ.get("WORKER_DELAY", 0.0))

    master_host = os.environ.get("MASTER_HOST",
                      CFG["communication"]["host"])

    master_port = int(os.environ.get("MASTER_PORT",
                      CFG["communication"]["port"]))

    print(f"[Worker {worker_id}] Connecting to {master_host}:{master_port} "
          f"| delay={delay}s")

    worker = Worker(
        worker_id=worker_id,
        master_host=master_host,
        master_port=master_port,
        artificial_delay=delay,
    )
    worker.run()
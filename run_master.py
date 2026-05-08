"""
run_master.py — Entry point to start the master node.

Docker-aware: reads N_WORKERS, MASTER_HOST, MASTER_PORT, N_EPOCHS,
N_SAMPLES from environment variables so docker-compose.yml can
configure the run without editing source code.

For local runs (no Docker), all settings fall back to config.yaml.

Usage
-----
    # Local (reads config.yaml):
    python run_master.py

    # Docker (docker-compose sets env vars automatically):
    docker-compose up
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coordination.master import Master
from config_loader import CFG

if __name__ == "__main__":
    # Read from environment (Docker) or fall back to config.yaml
    n_workers = int(os.environ.get("N_WORKERS",
                    CFG["training"]["n_workers"]))
    host      = os.environ.get("MASTER_HOST",
                    CFG["communication"]["host"])
    port      = int(os.environ.get("MASTER_PORT",
                    CFG["communication"]["port"]))
    n_epochs  = int(os.environ.get("N_EPOCHS",
                    CFG["training"]["n_epochs"]))
    n_samples = int(os.environ.get("N_SAMPLES",
                    CFG["dataset"]["n_samples"]))

    print(f"[Master] Config: n_workers={n_workers} | host={host} | "
          f"port={port} | n_epochs={n_epochs} | n_samples={n_samples:,}", flush=True)

    master = Master(
        n_workers=n_workers,
        host=host,
        port=port,
        n_epochs=n_epochs,
        n_samples=n_samples,
    )
    master.run()
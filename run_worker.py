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
import argparse

from coordination.worker import Worker
from project_config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distributed training worker.")
    parser.add_argument("worker_id", type=int, help="Worker id.")
    parser.add_argument(
        "delay",
        nargs="?",
        type=float,
        default=None,
        help="Optional artificial delay in seconds.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    return parser.parse_args()

if __name__ == "__main__":
    # Keep compatibility with existing positional invocation while supporting YAML config.
    args = parse_args()
    cfg = load_config(args.config)

    worker_cfg = cfg["workers"]
    master_cfg = cfg["master"]

    delays = worker_cfg.get("artificial_delays", [])
    config_delay = delays[args.worker_id] if args.worker_id < len(delays) else 0.0
    delay = args.delay if args.delay is not None else config_delay

    worker = Worker(
        worker_id=args.worker_id,
        master_host=master_cfg["host"],
        master_port=master_cfg["port"],
        artificial_delay=delay,
    )
    worker.run()
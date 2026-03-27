"""Entry point to start the master node."""
import argparse

from coordination.master import Master
from project_config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distributed training master.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override training sample size.")
    parser.add_argument("--training-log", default=None, help="Override training log CSV path.")
    parser.add_argument("--run-summary", default=None, help="Override run summary JSON path.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    master_cfg = cfg["master"]
    data_cfg = cfg["data"]
    logging_cfg = cfg["logging"]

    n_samples = args.n_samples if args.n_samples is not None else data_cfg["n_samples"]
    training_log_path = args.training_log or logging_cfg["training_log_path"]
    run_summary_path = args.run_summary or logging_cfg["run_summary_path"]

    master = Master(
        n_workers=master_cfg["n_workers"],
        host=master_cfg["host"],
        port=master_cfg["port"],
        n_epochs=master_cfg["n_epochs"],
        lr=master_cfg["lr"],
        n_samples=n_samples,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        benchmark_matrix_size=master_cfg["benchmark_matrix_size"],
        training_log_path=training_log_path,
        run_summary_path=run_summary_path,
    )
    master.run()
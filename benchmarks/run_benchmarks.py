"""Benchmark orchestrator for distributed training experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from project_config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark sweeps for sample sizes.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def _write_benchmark_row(results_csv: Path, row: Dict[str, object]) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    write_header = not results_csv.exists()
    fieldnames = [
        "timestamp",
        "run_id",
        "sample_size",
        "repeat",
        "n_workers",
        "n_epochs",
        "lr",
        "worker_delays",
        "total_runtime_sec",
        "final_test_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "training_log_path",
        "run_summary_path",
    ]

    with results_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _terminate_if_running(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_single_benchmark(
    config_path: str,
    sample_size: int,
    repeat_idx: int,
    startup_wait_sec: float,
    worker_delays: List[float],
    n_workers: int,
) -> Dict[str, object]:
    now_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"s{sample_size}_r{repeat_idx}_{now_label}"

    run_dir = Path("logs") / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    training_log_path = run_dir / f"training_{run_id}.csv"
    run_summary_path = run_dir / f"summary_{run_id}.json"

    master_cmd = [
        sys.executable,
        "run_master.py",
        "--config",
        config_path,
        "--n-samples",
        str(sample_size),
        "--training-log",
        str(training_log_path),
        "--run-summary",
        str(run_summary_path),
    ]

    worker_cmds = []
    for worker_id in range(n_workers):
        delay = worker_delays[worker_id] if worker_id < len(worker_delays) else 0.0
        worker_cmds.append(
            [
                sys.executable,
                "run_worker.py",
                str(worker_id),
                str(delay),
                "--config",
                config_path,
            ]
        )

    master_proc = None
    worker_procs: List[subprocess.Popen] = []

    try:
        print(f"[Benchmark] Starting run {run_id}")
        master_proc = subprocess.Popen(master_cmd)

        time.sleep(startup_wait_sec)

        for cmd in worker_cmds:
            worker_procs.append(subprocess.Popen(cmd))

        master_code = master_proc.wait()
        worker_codes = [proc.wait() for proc in worker_procs]

        if master_code != 0:
            raise RuntimeError(f"Master failed with exit code {master_code}")

        failed_workers = [code for code in worker_codes if code != 0]
        if failed_workers:
            raise RuntimeError(f"At least one worker failed: {failed_workers}")

        if not run_summary_path.exists():
            raise FileNotFoundError(f"Run summary missing: {run_summary_path}")

        with run_summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

        metrics = summary.get("metrics", {})
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "sample_size": sample_size,
            "repeat": repeat_idx,
            "n_workers": summary.get("n_workers", n_workers),
            "n_epochs": summary.get("n_epochs"),
            "lr": summary.get("lr"),
            "worker_delays": ",".join(str(v) for v in worker_delays[:n_workers]),
            "total_runtime_sec": summary.get("total_runtime_sec"),
            "final_test_loss": summary.get("final_test_loss"),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "training_log_path": str(training_log_path),
            "run_summary_path": str(run_summary_path),
        }
    finally:
        if master_proc is not None:
            _terminate_if_running(master_proc)
        for proc in worker_procs:
            _terminate_if_running(proc)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    benchmark_cfg = cfg["benchmark"]
    master_cfg = cfg["master"]
    workers_cfg = cfg["workers"]

    sample_sizes = benchmark_cfg["sample_sizes"]
    repeats = int(benchmark_cfg["repeats"])
    startup_wait_sec = float(benchmark_cfg["master_startup_wait_sec"])
    results_csv = Path(benchmark_cfg["results_csv_path"])

    n_workers = int(master_cfg["n_workers"])
    worker_delays = workers_cfg.get("artificial_delays", [])

    print("[Benchmark] Configuration")
    print(f"  sample_sizes={sample_sizes}")
    print(f"  repeats={repeats}")
    print(f"  n_workers={n_workers}")

    for sample_size in sample_sizes:
        for repeat_idx in range(1, repeats + 1):
            row = run_single_benchmark(
                config_path=args.config,
                sample_size=int(sample_size),
                repeat_idx=repeat_idx,
                startup_wait_sec=startup_wait_sec,
                worker_delays=worker_delays,
                n_workers=n_workers,
            )
            _write_benchmark_row(results_csv, row)
            print(
                "[Benchmark] Completed "
                f"sample_size={sample_size}, repeat={repeat_idx}, "
                f"total_runtime_sec={row['total_runtime_sec']}"
            )

    print(f"[Benchmark] Results written to {results_csv}")


if __name__ == "__main__":
    main()

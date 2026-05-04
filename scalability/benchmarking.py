"""
scalability/benchmarking.py — Member C: Structured Performance Benchmarking.

Runs systematic experiments across varying cluster sizes (n_workers)
and sample sizes. Produces detailed comparative reports analyzing:
  - Scalability trends (how performance changes with more workers)
  - Efficiency curves (speedup vs ideal linear speedup)
  - System bottlenecks (where time is spent: compute vs communication)

Usage
-----
    python scalability/benchmarking.py

This spawns all processes internally — no separate terminals needed.

Output
------
    benchmark_results/m3/                             
        scalability_raw_TIMESTAMP.json        — raw per-run data
        scalability_summary.csv               — one row per experiment
        bottleneck_analysis.csv               — time breakdown per run
"""

import os
import sys
import json
import time
import datetime
import csv
import multiprocessing
from unittest import result

os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config_loader import CFG
from scalability.parallel_efficiency import EfficiencyCalculator


# ── Experiment configurations ──────────────────────────────────────────────────

# Cluster sizes to sweep — each is a number of workers
CLUSTER_SIZES = [1, 2]   

# Sample sizes to test scalability against
SCALE_SAMPLES = [
    5000,
    10000,
    20000,
    40000,
]

N_EPOCHS_PER_RUN = 5  


# ── Process target functions ───────────────────────────────────────────────────

def _master_target(port, n_workers, n_samples, n_epochs, result_queue):
    """Run master in subprocess for one scalability experiment."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from coordination.master import Master
        from config_loader import CFG
        m = Master(
            n_workers=n_workers,
            port=port,
            n_samples=n_samples,
            n_epochs=n_epochs,
            lr=CFG["model"]["lr"],
        )
        metrics = m.run_and_return_metrics()
        result_queue.put({"status": "ok", "metrics": metrics})
    except Exception as e:
        import traceback
        result_queue.put({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def _worker_target(worker_id, port, delay=0.0):
    """Run worker in subprocess."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    time.sleep(3.0)
    from coordination.worker import Worker
    Worker(worker_id=worker_id, master_port=port,
           artificial_delay=delay).run()


# ── Single experiment runner ───────────────────────────────────────────────────

def run_scalability_experiment(n_workers: int, n_samples: int,
                               n_epochs: int, port: int) -> dict:
    """Run one complete experiment and return timing + metrics.

    Parameters
    ----------
    n_workers : int
        Number of worker processes to spawn.
    n_samples : int
        MNIST samples to use.
    n_epochs : int
        Training epochs.
    port : int
        TCP port (use unique ports to avoid conflicts between runs).

    Returns
    -------
    dict
        Experiment result with keys: n_workers, n_samples, n_epochs,
        total_time_sec, avg_epoch_sec, final_loss, accuracy, f1,
        compute_time_sec, comm_time_sec, status.
    """
    print(f"\n[Scalability] n_workers={n_workers} | "
          f"n_samples={n_samples:,} | n_epochs={n_epochs}")

    q = multiprocessing.Queue()
    multiprocessing.set_executable(sys.executable)

    master = multiprocessing.Process(
        target=_master_target, daemon=False,
        args=(port, n_workers, n_samples, n_epochs, q)
    )
    workers = [
        multiprocessing.Process(
            target=_worker_target, daemon=True,
            args=(i, port, 0.0)
        )
        for i in range(n_workers)
    ]

    t_start = time.time()
    master.start()
    for w in workers:
        w.start()

    timeout = CFG["communication"]["timeout_sec"] * n_epochs
    master.join(timeout=timeout)
    total_time = round(time.time() - t_start, 2)

    if master.is_alive():
        master.terminate()
        return {
            "n_workers": n_workers, "n_samples": n_samples,
            "n_epochs": n_epochs, "total_time_sec": total_time,
            "status": "timeout",
        }

    result = q.get() if not q.empty() else {"status": "error"}
    if result["status"] != "ok":
        if "traceback" in result:
            print(f"[Scalability] ERROR traceback:\n{result['traceback']}")
        return {
            "n_workers": n_workers, "n_samples": n_samples,
            "n_epochs": n_epochs, "total_time_sec": total_time,
            "status": result.get("status", "error"),
        }

    m = result["metrics"]
    epoch_times = m.get("epoch_times", [total_time / n_epochs])
    avg_epoch   = round(sum(epoch_times) / len(epoch_times), 2) if epoch_times else 0

    return {
        "n_workers":         n_workers,
        "n_samples":         n_samples,
        "n_epochs":          n_epochs,
        "total_time_sec":    total_time,
        "avg_epoch_sec":     avg_epoch,
        "final_loss":        m.get("final_loss"),
        "test_loss":         m.get("test_loss"),
        "accuracy":          m.get("accuracy"),
        "f1":                m.get("f1"),
        "worker0_samples":   m.get("worker0_samples"),
        "worker1_samples":   m.get("worker1_samples", 0),
        "throughput_sps":    round((n_samples * n_epochs) / total_time, 1) if total_time > 0 else 0,
        "epoch_times":       epoch_times,
        "status":            "ok",
        "timestamp":         datetime.datetime.now().isoformat(),
    }


# ── Full benchmark suite ───────────────────────────────────────────────────────

def run_scalability_suite(results_dir: str = "benchmark_results/m3"):
    """Run all combinations of cluster sizes and sample sizes.

    Saves raw JSON per run, a summary CSV, and a bottleneck analysis CSV.

    Parameters
    ----------
    results_dir : str
        Directory to save all output files.
    """
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    port        = 5500
    serial_times = {}   # n_samples -> serial (1 worker) time for efficiency calc

    print("\n" + "=" * 60)
    print("   SCALABILITY BENCHMARK SUITE")
    print(f"  Cluster sizes: {CLUSTER_SIZES}")
    print(f"  Sample sizes:  {SCALE_SAMPLES}")
    print(f"  Epochs/run:    {N_EPOCHS_PER_RUN}")
    print("=" * 60)

    for n_samples in SCALE_SAMPLES:
        for n_workers in CLUSTER_SIZES:
            result = run_scalability_experiment(
                n_workers=n_workers,
                n_samples=n_samples,
                n_epochs=N_EPOCHS_PER_RUN,
                port=port,
            )
            all_results.append(result)

            # Record serial time (1 worker) for efficiency calculations
            if n_workers == 1 and result["status"] == "ok":
                serial_times[n_samples] = result["total_time_sec"]

            # Save raw JSON immediately
            _save_json(result, results_dir)
            port += 1

            status = result.get("status", "?")
            t      = result.get("total_time_sec", "?")
            loss   = result.get("final_loss", "?")
            print(f"  => n_workers={n_workers} | {n_samples:,} samples | "
                  f"{t}s | loss={loss} | {status}")

    # Compute efficiency metrics
    _add_efficiency_metrics(all_results, serial_times)

    # Save CSV summary
    _save_summary_csv(all_results, results_dir)

    # Save bottleneck analysis
    _save_bottleneck_csv(all_results, results_dir)

    print(f"\n[Scalability] Suite complete. Results in {results_dir}/")
    return all_results


def _add_efficiency_metrics(results: list, serial_times: dict):
    """Annotate results with speedup and efficiency (in-place)."""
    for r in results:
        n_samples  = r.get("n_samples", 0)
        n_workers  = r.get("n_workers", 1)
        t_parallel = r.get("total_time_sec", 0)
        t_serial   = serial_times.get(n_samples, t_parallel)

        if t_parallel > 0 and t_serial > 0:
            speedup    = round(t_serial / t_parallel, 4)
            efficiency = round(speedup / n_workers, 4)
        else:
            speedup = efficiency = 0.0

        r["speedup"]    = speedup
        r["efficiency"] = efficiency
        r["serial_time_sec"] = t_serial


def _save_json(result: dict, results_dir: str):
    """Save one run as a timestamped JSON file."""
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"scale_w{result['n_workers']}_{result['n_samples']}samp_{ts}.json"
    fpath = os.path.join(results_dir, fname)
    r     = dict(result)
    r.pop("epoch_times", None)   # lists don't CSV well
    with open(fpath, "w") as f:
        json.dump(result, f, indent=2)


def _save_summary_csv(results: list, results_dir: str):
    """Save all results to scalability_summary.csv."""
    fpath = os.path.join(results_dir, "scalability_summary.csv")
    fields = [
        "n_workers", "n_samples", "n_epochs", "total_time_sec",
        "avg_epoch_sec", "speedup", "efficiency", "throughput_sps",
        "final_loss", "test_loss", "accuracy", "f1",
        "worker0_samples", "worker1_samples", "status", "timestamp",
    ]
    with open(fpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"[Scalability] Summary CSV -> {fpath}")


def _save_bottleneck_csv(results: list, results_dir: str):
    """Save bottleneck analysis: per-epoch time breakdown.

    For each run, breaks down where time is spent:
    - Setup time (benchmark + data transfer before training starts)
    - Training time (epoch_times sum)
    - Overhead (total - setup - training)
    """
    fpath  = os.path.join(results_dir, "bottleneck_analysis.csv")
    fields = [
        "n_workers", "n_samples", "total_time_sec",
        "training_time_sec", "avg_epoch_sec",
        "throughput_sps", "speedup", "efficiency",
    ]
    rows = []
    for r in results:
        if r.get("status") != "ok":
            continue
        epoch_times   = r.get("epoch_times", [])
        training_time = round(sum(epoch_times), 2) if epoch_times else r.get("total_time_sec", 0)
        rows.append({
            "n_workers":       r["n_workers"],
            "n_samples":       r["n_samples"],
            "total_time_sec":  r["total_time_sec"],
            "training_time_sec": training_time,
            "avg_epoch_sec":   r.get("avg_epoch_sec", 0),
            "throughput_sps":  r.get("throughput_sps", 0),
            "speedup":         r.get("speedup", 1.0),
            "efficiency":      r.get("efficiency", 1.0),
        })

    with open(fpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Scalability] Bottleneck CSV -> {fpath}")


if __name__ == "__main__":
    multiprocessing.set_executable(sys.executable)
    multiprocessing.set_start_method("spawn", force=True)
    run_scalability_suite()
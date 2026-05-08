"""
scalability/resource_monitor.py — Member B: Resource Utilization Analysis.

Monitors CPU utilization and memory footprint per worker and master
to ensure efficient scaling. Samples resource usage at regular intervals
and writes results to a structured CSV for graph generation.

Usage
-----
    monitor = ResourceMonitor(worker_id=0)
    monitor.start()
    # ... run training ...
    monitor.stop()
    summary = monitor.get_summary()
"""

import os
import csv
import time
import threading
import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ResourceMonitor:
    """Samples CPU and memory usage of the current process at intervals.

    Runs a background thread that polls psutil every sample_interval_sec
    seconds. Results are collected in memory and can be written to CSV
    after the monitored operation finishes.

    Falls back gracefully if psutil is not installed — records zeros
    so the rest of the pipeline still works.

    Parameters
    ----------
    worker_id : int or str
        Label for the process being monitored (e.g., 0, 1, "master").
    sample_interval_sec : float
        How often to sample CPU/memory (default 1.0 second).
    output_dir : str
        Directory to write CSV files into.

    Attributes
    ----------
    samples : list of dict
        Collected resource samples. Each dict has:
        timestamp, cpu_percent, rss_mb, vms_mb, elapsed_sec.
    """

    def __init__(self, worker_id, sample_interval_sec: float = 1.0,
                 output_dir: str = "logs/resources"):
        self.worker_id           = worker_id
        self.sample_interval     = sample_interval_sec
        self.output_dir          = output_dir
        self.samples             = []
        self._stop_event         = threading.Event()
        self._thread             = None
        self._start_time         = None

        if not PSUTIL_AVAILABLE:
            print("[ResourceMonitor] psutil not installed. "
                  "Install with: pip install psutil\n"
                  "Recording zeros — pipeline will still work.")

    def start(self):
        """Begin background sampling thread."""
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True
        )
        self._thread.start()
        print(f"[ResourceMonitor] Worker {self.worker_id}: monitoring started")

    def stop(self):
        """Stop the sampling thread and flush results to CSV."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._write_csv()
        print(f"[ResourceMonitor] Worker {self.worker_id}: "
              f"{len(self.samples)} samples collected")

    def get_summary(self) -> dict:
        """Return summary statistics over all collected samples.

        Returns
        -------
        dict
            Keys: worker_id, n_samples, avg_cpu_pct, peak_cpu_pct,
            avg_rss_mb, peak_rss_mb, duration_sec.
        """
        if not self.samples:
            return {
                "worker_id": self.worker_id, "n_samples": 0,
                "avg_cpu_pct": 0.0, "peak_cpu_pct": 0.0,
                "avg_rss_mb": 0.0, "peak_rss_mb": 0.0,
                "duration_sec": 0.0,
            }

        cpu_vals = [s["cpu_percent"] for s in self.samples]
        rss_vals = [s["rss_mb"]      for s in self.samples]

        return {
            "worker_id":    self.worker_id,
            "n_samples":    len(self.samples),
            "avg_cpu_pct":  round(sum(cpu_vals) / len(cpu_vals), 2),
            "peak_cpu_pct": round(max(cpu_vals), 2),
            "avg_rss_mb":   round(sum(rss_vals) / len(rss_vals), 2),
            "peak_rss_mb":  round(max(rss_vals), 2),
            "duration_sec": round(self.samples[-1]["elapsed_sec"], 2),
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _sample_loop(self):
        """Background thread: poll CPU/memory every sample_interval_sec."""
        if PSUTIL_AVAILABLE:
            proc = psutil.Process(os.getpid())
        else:
            proc = None

        while not self._stop_event.is_set():
            elapsed = time.time() - self._start_time

            if proc is not None:
                try:
                    cpu = proc.cpu_percent(interval=None)
                    mem = proc.memory_info()
                    rss_mb = mem.rss / (1024 * 1024)
                    vms_mb = mem.vms / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    cpu = rss_mb = vms_mb = 0.0
            else:
                cpu = rss_mb = vms_mb = 0.0

            self.samples.append({
                "timestamp":   datetime.datetime.now().isoformat(),
                "elapsed_sec": round(elapsed, 2),
                "cpu_percent": round(cpu, 2),
                "rss_mb":      round(rss_mb, 2),
                "vms_mb":      round(vms_mb, 2),
            })

            self._stop_event.wait(timeout=self.sample_interval)

    def _write_csv(self):
        """Write all samples to a CSV file in output_dir."""
        if not self.samples:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"resources_worker{self.worker_id}_{ts}.csv"
        fpath = os.path.join(self.output_dir, fname)

        with open(fpath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.samples[0].keys())
            writer.writeheader()
            writer.writerows(self.samples)

        print(f"[ResourceMonitor] CSV -> {fpath}")


class MultiWorkerResourceReport:
    """Aggregates resource summaries from multiple workers into one report.

    Parameters
    ----------
    summaries : list of dict
        Each dict is the output of ResourceMonitor.get_summary().
    output_dir : str
        Directory to write the aggregated CSV.
    """

    def __init__(self, summaries: list, output_dir: str = "benchmark_results"):
        self.summaries  = summaries
        self.output_dir = output_dir

    def save_csv(self, label: str = ""):
        """Write all worker summaries to a single aggregated CSV.

        Parameters
        ----------
        label : str
            Optional label appended to the filename (e.g., "n2_10k").
        """
        os.makedirs(self.output_dir, exist_ok=True)
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"resource_summary_{label}_{ts}.csv" if label else f"resource_summary_{ts}.csv"
        fpath = os.path.join(self.output_dir, fname)

        fields = ["worker_id", "n_samples", "avg_cpu_pct", "peak_cpu_pct",
                  "avg_rss_mb", "peak_rss_mb", "duration_sec"]
        with open(fpath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.summaries)

        print(f"[ResourceReport] Aggregated CSV -> {fpath}")
        return fpath

    def print_table(self):
        """Print a formatted resource utilization table."""
        print("\n" + "=" * 70)
        print(f"  {'Worker':>8} {'Avg CPU%':>10} {'Peak CPU%':>11} "
              f"{'Avg RSS MB':>12} {'Peak RSS MB':>13}")
        print("=" * 70)
        for s in self.summaries:
            print(f"  {str(s['worker_id']):>8} {s['avg_cpu_pct']:>10.1f} "
                  f"{s['peak_cpu_pct']:>11.1f} {s['avg_rss_mb']:>12.1f} "
                  f"{s['peak_rss_mb']:>13.1f}")
        print("=" * 70 + "\n")
"""
heterogeneity/scenarios.py — Controlled heterogeneity scenarios for M2 testing.

Defines named scenarios with variable batch sizes, artificial compute
delays, and uneven workload distribution. Used by benchmark_suite.py
and test_m2_heterogeneity.py to run structured comparative analysis.

Usage
-----
    from heterogeneity.scenarios import SCENARIOS, get_scenario, run_scenario

    scenario = get_scenario("one_slow_worker")
    print(scenario["worker_delays"])   # [0.0, 2.0]
"""

import os
import sys
import json
import time
import datetime
import multiprocessing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ── Scenario definitions ──────────────────────────────────────────────────────

SCENARIOS = [
    {
        "name":            "equal_fast_workers",
        "description":     "Both workers run at full speed with no artificial delay. "
                           "Ideal homogeneous baseline case.",
        "worker_delays":   [0.0, 0.0],
        "variable_batch":  False,
        "n_samples":       5000,
        "expected_effect": "Equal batch sizes. Fastest possible convergence. "
                           "Establishes lower bound on epoch time.",
    },
    {
        "name":            "one_slow_worker",
        "description":     "Worker 1 has a 2-second artificial delay per epoch. "
                           "Simulates a slow machine or resource-constrained node.",
        "worker_delays":   [0.0, 2.0],
        "variable_batch":  True,
        "n_samples":       5000,
        "expected_effect": "Worker 1 gets fewer samples due to proportional allocation. "
                           "Epoch time dominated by slow worker's delay.",
    },
    {
        "name":            "very_slow_worker",
        "description":     "Worker 1 has a 5-second delay. Extreme heterogeneity "
                           "to test robustness of the variable batch strategy.",
        "worker_delays":   [0.0, 5.0],
        "variable_batch":  True,
        "n_samples":       5000,
        "expected_effect": "Worker 1 gets near-zero samples. Almost all computation "
                           "done by Worker 0. Tests proportional allocation limits.",
    },
    {
        "name":            "moderate_heterogeneity",
        "description":     "Worker 1 has a 0.5-second delay. Subtle speed difference "
                           "— realistic for CPU vs memory bottleneck.",
        "worker_delays":   [0.0, 0.5],
        "variable_batch":  True,
        "n_samples":       5000,
        "expected_effect": "Small but measurable difference in sample allocation. "
                           "Tests system sensitivity to mild heterogeneity.",
    },
    {
        "name":            "variable_batch_disabled",
        "description":     "Worker 1 has a 2-second delay but variable_batch is OFF. "
                           "Comparison case: equal batches despite unequal speeds.",
        "worker_delays":   [0.0, 2.0],
        "variable_batch":  False,
        "n_samples":       5000,
        "expected_effect": "Equal batch sizes despite speed difference. Epoch time "
                           "worse than one_slow_worker — shows cost of NOT adapting.",
    },
]

_SCENARIO_MAP = {s["name"]: s for s in SCENARIOS}


def get_scenario(name: str) -> dict:
    """Retrieve a scenario dict by name.

    Parameters
    ----------
    name : str
        Scenario name as defined in SCENARIOS.

    Returns
    -------
    dict
        Scenario configuration dictionary.

    Raises
    ------
    KeyError
        If the name is not found.
    """
    if name not in _SCENARIO_MAP:
        raise KeyError(
            f"Scenario '{name}' not found. "
            f"Available: {list(_SCENARIO_MAP.keys())}"
        )
    return _SCENARIO_MAP[name]


def list_scenarios() -> list:
    """Return all scenario names.

    Returns
    -------
    list of str
    """
    return list(_SCENARIO_MAP.keys())


# ── Scenario runner ───────────────────────────────────────────────────────────

def _master_proc(port, n_samples, n_epochs, variable_batch, result_queue):
    """Run master in a subprocess."""
    try:
        from coordination.master import Master
        from config_loader import CFG
        m = Master(
            n_workers=2, port=port,
            n_samples=n_samples, n_epochs=n_epochs,
            lr=CFG["model"]["lr"],
        )
        # Override variable batch setting
        m._variable_batch_override = variable_batch
        metrics = m.run_and_return_metrics()
        result_queue.put({"status": "ok", "metrics": metrics})
    except Exception as e:
        result_queue.put({"status": "error", "error": str(e)})


def _worker_proc(worker_id, port, delay):
    """Run worker in a subprocess."""
    time.sleep(1.0)
    from coordination.worker import Worker
    Worker(worker_id=worker_id, master_port=port,
           artificial_delay=delay).run()


def run_scenario(scenario_name: str, port: int = 5300,
                 n_epochs: int = 3) -> dict:
    """Run one named scenario and return its results.

    Parameters
    ----------
    scenario_name : str
        Name from SCENARIOS list.
    port : int
        TCP port to use (use unique ports per scenario to avoid conflicts).
    n_epochs : int
        How many epochs to train for this scenario.

    Returns
    -------
    dict
        Result dictionary with scenario metadata + training metrics.
    """
    s = get_scenario(scenario_name)
    print(f"\n[Scenario] Running: {scenario_name}")
    print(f"           Delays: {s['worker_delays']} | "
          f"VarBatch: {s['variable_batch']}")

    q = multiprocessing.Queue()
    master  = multiprocessing.Process(
        target=_master_proc, daemon=False,
        args=(port, s["n_samples"], n_epochs,
              s["variable_batch"], q)
    )
    worker0 = multiprocessing.Process(
        target=_worker_proc, daemon=True,
        args=(0, port, s["worker_delays"][0])
    )
    worker1 = multiprocessing.Process(
        target=_worker_proc, daemon=True,
        args=(1, port, s["worker_delays"][1])
    )

    t_start = time.time()
    master.start(); worker0.start(); worker1.start()
    master.join(timeout=600)
    total_time = round(time.time() - t_start, 2)

    if master.is_alive():
        master.terminate()
        return {**s, "status": "timeout", "total_time_sec": total_time}

    result = q.get() if not q.empty() else {"status": "error"}
    if result["status"] != "ok":
        return {**s, "status": result.get("status", "error"),
                "total_time_sec": total_time}

    m = result["metrics"]
    return {
        "scenario_name":      scenario_name,
        "description":        s["description"],
        "worker_delays":      s["worker_delays"],
        "variable_batch":     s["variable_batch"],
        "n_samples":          s["n_samples"],
        "total_time_sec":     total_time,
        "avg_epoch_sec":      round(total_time / n_epochs, 2),
        "final_loss":         m.get("final_loss"),
        "test_loss":          m.get("test_loss"),
        "accuracy":           m.get("accuracy"),
        "f1":                 m.get("f1"),
        "worker0_samples":    m.get("worker0_samples"),
        "worker1_samples":    m.get("worker1_samples"),
        "worker0_rtt":        m.get("worker0_rtt"),
        "worker1_rtt":        m.get("worker1_rtt"),
        "status":             "ok",
    }


def run_all_scenarios(results_dir: str = "benchmark_results/scenarios",
                      n_epochs: int = 3):
    """Run all scenarios sequentially and save results to JSON + CSV.

    Parameters
    ----------
    results_dir : str
        Directory to save scenario results into.
    n_epochs : int
        Epochs per scenario.
    """
    import csv
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    port = 5300

    for i, scenario in enumerate(SCENARIOS):
        result = run_scenario(scenario["name"], port=port + i,
                              n_epochs=n_epochs)
        all_results.append(result)

        # Save individual JSON
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_{scenario['name']}.json"
        with open(os.path.join(results_dir, fname), "w") as f:
            json.dump(result, f, indent=2)

        print(f"[Scenario] Done: {scenario['name']} | "
              f"{result.get('total_time_sec')}s | "
              f"loss={result.get('final_loss')}")

    # Save summary CSV
    csv_path = os.path.join(results_dir, "scenario_summary.csv")
    fields = [
        "scenario_name", "worker_delays", "variable_batch",
        "n_samples", "total_time_sec", "avg_epoch_sec",
        "final_loss", "test_loss", "accuracy", "f1",
        "worker0_samples", "worker1_samples", "status",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            row = dict(r)
            row["worker_delays"] = str(row.get("worker_delays", ""))
            writer.writerow(row)

    print(f"\n[Scenarios] Results saved to {results_dir}/")
    return all_results


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run_all_scenarios()

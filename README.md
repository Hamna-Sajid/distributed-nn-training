# Distributed Neural Network Training from Scratch on Heterogeneous Compute Nodes

A distributed neural network training system built from scratch using NumPy and Python sockets — no PyTorch, no TensorFlow.

Built as a course project for **Parallel and Distributed Computing**, this system implements synchronous data-parallel training across multiple compute nodes with differing performance characteristics.

---

## What This Project Does

- Trains a multi-layer perceptron (MLP) for **multi-label classification**
- Splits training data across multiple worker nodes
- Each worker computes gradients on its local shard
- A master node **aggregates gradients** and broadcasts updated weights
- Handles **heterogeneous nodes** (nodes with different speeds) via benchmarking and proportional data allocation

---

## Project Structure
```
distributed-nn-training/
├── neural_network/        # MLP, layers, activations, loss functions
├── communication/         # Socket protocol and message framing
├── coordination/          # Master and worker node logic
├── data/                  # Dataset generation and partitioning
├── heterogeneity/         # Speed benchmarking and workload adjustment
├── tests/                 # Unit and integration tests
├── docs/                  # Design documents and diagrams
├── run_master.py          # Entry point: start master node
├── run_worker.py          # Entry point: start a worker node
└── requirements.txt
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Neural network | NumPy + native Python |
| Communication | Python `socket` (TCP) |
| Dataset | `sklearn.make_multilabel_classification` |
| Version control | Git / GitHub |

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/distributed-nn-training.git
cd distributed-nn-training
pip install -r requirements.txt
```

### Global Configuration
The project uses a root-level `config.yaml` for global settings.

Key sections:
- `master`: host, port, worker count, epochs, learning rate
- `data`: dataset sample size and split settings
- `workers`: per-worker artificial delays
- `logging`: training CSV and run summary JSON paths
- `benchmark`: sample size sweep + output CSV path

Edit `config.yaml` once, then run your experiments without changing code.

### Run Single-Node Test (verify NN correctness first)
```bash
python tests/test_single_node.py
```

### Run Distributed Training (3 terminals)
```bash
# Terminal 1 — Master
python run_master.py

# Terminal 2 — Worker 0 (fast node)
python run_worker.py 0

# Terminal 3 — Worker 1 (slow node, 2s artificial delay)
python run_worker.py 1 2.0
```

You can override sample size per run (useful for quick experiments):
```bash
python run_master.py --n-samples 20000
```

Each run now stores:
- Per-epoch training CSV (`logs/training_log.csv` by default)
- Run summary JSON (`logs/last_run_summary.json` by default)

### Benchmark Sweep (Multiple Sample Sizes)
Run a complete benchmark sweep from `config.yaml`:
```bash
python benchmarks/run_benchmarks.py
```

By default, this script:
- Iterates over `benchmark.sample_sizes`
- Repeats each case `benchmark.repeats` times
- Launches master + workers automatically
- Writes per-run files under `logs/runs/`
- Appends comparison rows to `logs/benchmark_results.csv`

Example comparison columns:
- `sample_size`
- `total_runtime_sec`
- `final_test_loss`
- `accuracy`, `precision`, `recall`, `f1`

---

## Gradient Synchronization Strategies

This project implements two strategies:

**Parameter Server** — all workers send gradients to the master, which averages them and broadcasts updated weights. Simple but creates a bottleneck at the master.

**All-Reduce (Ring)** — workers exchange gradients directly in a ring topology. Eliminates the central bottleneck but is more complex to implement.

---

## Heterogeneous Node Handling

On startup, the master benchmarks each worker by sending a small matrix task and measuring round-trip time. Faster workers receive proportionally larger data shards. Artificial delays can be injected via `run_worker.py` to simulate slow nodes without needing different hardware.

---

## Team

| Name | 
|---|
| Hamna Sajid | 
| Anusha Randhawa | 
| Zarmeen Rahman | 
| Nayab Dhanani | 

---

## License

This project is for academic purposes only.
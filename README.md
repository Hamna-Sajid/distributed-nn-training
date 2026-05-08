# Distributed Neural Network Training from Scratch on Heterogeneous Compute Nodes

A distributed neural network training system built from scratch using **NumPy and Python sockets** — no PyTorch, no TensorFlow. Built as a course project for **Parallel and Distributed Computing**, implementing synchronous data-parallel training across heterogeneous compute nodes with adaptive aggregation, compression, and performance scaling.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
  - [Baseline Distributed Training](#baseline-distributed-training)
  - [Sync Barrier, Adaptive Aggregation & Compression](#sync-barrier-adaptive-aggregation--compression)
  - [Performance Scaling & Optimization](#performance-scaling--optimization)
- [Benchmark Suite](#benchmark-suite)
- [Running Tests](#running-tests)
- [Neural Network Architecture](#neural-network-architecture)
- [Gradient Synchronisation Strategies](#gradient-synchronisation-strategies)
- [Heterogeneity Handling](#heterogeneity-handling)
- [Results Summary](#results-summary)

---

## Project Overview

This project trains a multi-label classification neural network across distributed compute nodes:

- **Baseline**: Basic distributed training with parameter server architecture
- **Enhanced**: Adds synchronous barriers, adaptive gradient aggregation, and compression
- **Optimized**: Performance scaling benchmarks and optimization across varying worker counts and data sizes

**Key Features:**
- Synchronous gradient aggregation with barrier synchronization
- Adaptive gradient weighting using norm clipping
- Gradient compression (quantization, sparsification)
- Heterogeneous node support with speed benchmarking
- Docker containerization for multi-node deployment
- Comprehensive performance benchmarking and scaling analysis

---

## Architecture

### System Overview
```
┌─────────────┐
│   Master    │  ← Orchestrates training
│   Node      │  ← Aggregates gradients
└──────┬──────┘  ← Broadcasts weights
       │
   ┌───┴───────┬──────────┬──────────┐
   │           │          │          │
┌──▼──┐    ┌──▼──┐   ┌──▼──┐   ┌──▼──┐
│ W0  │    │ W1  │   │ W2  │   │ W3  │  ← Worker Nodes
│(fast)    │(std)│   │(std)│   │(slow)   ← Heterogeneous
└─────┘    └─────┘   └─────┘   └─────┘
```

### Communication Protocol
- **Message Format**: Length-prefixed JSON over TCP sockets
- **Phases**: Benchmarking → Data Distribution → Training Loop
- **Synchronization**: Barriers ensure all workers finish before aggregation
- **Compression**: Optional gradient quantization and sparsification

---

## Repository Structure

```
distributed-nn-training/
├── neural_network/              # Core neural network implementation
│   ├── __init__.py
│   ├── mlp.py                   # Multi-layer perceptron
│   ├── layers.py                # Dense layer implementation
│   ├── activations.py           # ReLU, softmax
│   ├── loss.py                  # Binary cross-entropy loss
│   └── ...
│
├── communication/               # Socket protocol & compression
│   ├── __init__.py
│   ├── protocol.py              # TCP message framing
│   ├── compression.py           # Quantization, sparsification
│   └── ...
│
├── coordination/                # Master & worker logic
│   ├── __init__.py
│   ├── master.py                # Master node orchestrator
│   ├── worker.py                # Worker node trainer
│   ├── sync_barrier.py          # Synchronization mechanism
│   ├── adaptive_aggregator.py   # Gradient aggregation & weighting
│   ├── model_saver.py           # Checkpoint management
│   ├── logger.py                # Training metrics
│   └── performance_report.py    # Benchmarking results
│
├── data/                        # Dataset handling
│   ├── __init__.py
│   └── loader.py                # Data generation & partitioning
│
├── heterogeneity/               # Heterogeneous node handling
│   ├── __init__.py
│   ├── scenarios.py             # Test scenarios
│   ├── batch_size.py            # Batch size variations
│   ├── data_skew.py             # Data distribution skew
│   ├── network_simulator.py     # Network delay simulation
│   ├── metrics.py               # Heterogeneity metrics
│   └── ...
│
├── scalability/                 # Performance scaling analysis
│   ├── __init__.py
│   ├── benchmarking.py          # Multi-worker benchmarks
│   ├── parallel_efficiency.py   # Speedup & efficiency metrics
│   └── resource_monitor.py      # CPU/memory profiling
│
├── tests/                       # Comprehensive test suite
│   ├── test_single_node.py      # Neural network correctness
│   ├── test_stability.py        # Training stability (1-4 workers)
│   ├── test_heterogeneity.py    # Heterogeneous node handling
│   ├── test_compression.py      # Gradient compression
│   ├── test_sync_aggregation.py # Synchronization & aggregation
│   └── evaluate.py              # Multi-run evaluation
│
├── graphs/                      # Generated performance graphs
│   └── m3/                      # Milestone 3 results
│
├── run_master.py                # Start master node
├── run_worker.py                # Start worker node
├── config.yaml                  # Configuration (seeds, learning rates, etc.)
├── config_loader.py             # Config parsing singleton
├── benchmark_suite.py           # Automated benchmarking
├── generate_graphs.py           # Graph generation from results
│
├── docker-compose.yml           # Docker orchestration (basic)
├── docker-compose.4workers.yml  # Docker orchestration (4 workers)
├── Dockerfile.master            # Master container
├── Dockerfile.worker            # Worker container
│
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── LICENSE                      # Project license
└── README.md                    # This file
```

---

## Installation

### Prerequisites
- **Python 3.8+** (3.9+ recommended)
- **pip** or **conda**
- **Docker & Docker Compose** (optional, for containerized deployment)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Hamna-Sajid/distributed-nn-training.git
cd distributed-nn-training
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python tests/test_single_node.py
```

---

## Configuration

All system settings are defined in [`config.yaml`](config.yaml):

```yaml
# Training
epochs: 10
learning_rate: 0.01
batch_size: 32

# Network
n_features: 100
n_samples: 10000
n_labels: 5

# Master
master_host: localhost
master_port: 5000
num_workers: 2

# Logging
log_interval: 1
verbose: true
```

**To override in code:**
```python
from config_loader import CFG
CFG.learning_rate = 0.005
CFG.epochs = 20
```

---

## Running the System

### Baseline Distributed Training

**Single-Node Test** (verify correctness):
```bash
python tests/test_single_node.py
```

**Distributed Training** (3+ terminals):

**Terminal 1 — Master:**
```bash
python run_master.py
```

**Terminal 2 — Worker 0 (fast node):**
```bash
python run_worker.py 0
```

**Terminal 3 — Worker 1 (slow node, 2s delay):**
```bash
python run_worker.py 1 2.0
```

Expected output: Training progresses, metrics logged each epoch.

---

### Sync Barrier, Adaptive Aggregation & Compression

**With Barrier Synchronization:**
```bash
# Terminal 1: Master (auto-enables M2 features in config)
python run_master.py

# Terminal 2 & 3: Worker 0 (fast) & Worker 1 (slow):
python run_worker.py 0
python run_worker.py 1 2.0
```

**Features Activated:**
- Synchronous barrier before each aggregation
- Adaptive gradient weighting (norm clipping)
- Optional compression (quantization/sparsification)

**Test Components:**
```bash
python tests/test_sync_aggregation.py
python tests/test_compression.py
python tests/test_heterogeneity.py
```

---

### Performance Scaling & Optimization

**Scalability Benchmark** (4 to 32 workers):
```bash
python benchmark_suite.py
```

**Run Tests for Different Worker Counts:**
```bash
# Baseline (1 worker):
python tests/test_single_node.py

# 2 workers:
python tests/test_stability.py -k test_4_scalability_2workers

# 4 workers:
python tests/test_stability.py -k test_8_scalability_4workers

# Full suite:
python tests/test_stability.py
```

**Docker Deployment** (automated 4-worker setup):
```bash
docker-compose -f docker-compose.4workers.yml up
```

**Generate Performance Graphs:**
```bash
python generate_graphs.py
```

Output: Speedup, efficiency, and scalability plots in `graphs/m3/`

---

## Benchmark Suite

The [`benchmark_suite.py`](benchmark_suite.py) script automates performance testing across configurations:

```bash
python benchmark_suite.py
```

**Measures:**
- Training time per epoch
- Throughput (samples/sec)
- Gradient compression ratio
- Communication overhead
- Scalability (1–32 workers)

**Output:** JSON results + CSV log + Excel summary in project root

---

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run specific test:
```bash
pytest tests/test_stability.py -v
```

**Core Stability Tests** (8 comprehensive tests):

| # | Test | Purpose | Time |
|---|---|---|---|
| 1 | `test_1_single_node_regression` | MLP correctness on single node | ~5s |
| 2 | `test_2_distributed_convergence` | 2-worker distributed training | ~8s |
| 3 | `test_3_mini_batch_convergence` | Mini-batch SGD convergence | ~6s |
| 4 | `test_4_compression_correctness` | Gradient compression accuracy | ~7s |
| 5 | `test_5_resource_monitor` | CPU/memory monitoring | ~5s |
| 6 | `test_6_scalability_2workers_faster` | 2-worker speedup measurement | ~10s |
| 7 | `test_7_full_pipeline_integration` | End-to-end pipeline test | ~12s |
| 8 | `test_8_scalability_4workers` | 4-worker scalability benchmark | ~15s |

Run stability tests:
```bash
pytest tests/test_stability.py -v
```

**Additional Test Modules:**

| Module | Purpose | Time |
|---|---|---|
| `test_single_node.py` | Neural network correctness | ~5s |
| `test_heterogeneity.py` | Heterogeneous node handling (5 tests) | ~20s |
| `test_compression.py` | Gradient compression variants (3 tests) | ~10s |
| `test_sync_aggregation.py` | Barrier & aggregation logic (2 tests) | ~8s |

---

## Docker & Scalability Commands

### Docker Compose (Basic Setup)

**Build Docker images:**
```bash
# Build master and worker Docker images
docker-compose build
```

**Start containers:**
```bash
# Start master and worker services in background
docker-compose up
```

**Stop containers:**
```bash
# Bring down all running services
docker-compose down
```

### Docker Compose (4-Worker Setup)

**Build images for 4-worker deployment:**
```bash
# Build images configured for 4 workers
docker-compose -f docker-compose.4workers.yml build
```

**Start 4-worker cluster:**
```bash
# Launch master and 4 worker nodes in parallel
docker-compose -f docker-compose.4workers.yml up
```

**Stop 4-worker cluster:**
```bash
# Bring down all 4-worker services
docker-compose -f docker-compose.4workers.yml down
```

### Scalability & Performance Benchmarking

**Run full scalability benchmark:**
```bash
# Benchmarks across multiple worker counts and data sizes
# Measures throughput, speedup, efficiency, compression ratios
# Results: JSON + CSV + Excel summary
python scalability/benchmarking.py
```

**Run stability tests (all 8):**
```bash
# Executes all stability tests: single-node, 2-worker, 4-worker, compression, resource monitoring
python tests/test_stability.py
```

---

## Neural Network Architecture

**Multi-Layer Perceptron (MLP)** for multi-label classification:

```
Input (n_features=100)
   ↓
Dense (256 units) + ReLU
   ↓
Dense (128 units) + ReLU
   ↓
Dense (64 units) + ReLU
   ↓
Output (n_labels=5) + Sigmoid
   ↓
Binary Cross-Entropy Loss
```

**Implemented in [`neural_network/`](neural_network/):**
- [`mlp.py`](neural_network/mlp.py): Forward/backward pass
- [`layers.py`](neural_network/layers.py): Dense layer + gradient computation
- [`activations.py`](neural_network/activations.py): ReLU, Sigmoid, Softmax
- [`loss.py`](neural_network/loss.py): BCE loss & derivatives

---

## Gradient Synchronisation Strategies

### Parameter Server (Used in M1)
```
Worker 0      Worker 1      Worker 2
   ↓             ↓             ↓
   └─────────┬───┴─────────────┘
             ↓
        Master (aggregates)
             ↓
   ┌─────────┴───────────────┐
   ↓             ↓             ↓
Worker 0      Worker 1      Worker 2
```

**Pros:** Simple, deterministic  
**Cons:** Master becomes bottleneck at scale

### All-Reduce (Ring topology)
```
Worker 0 ↔ Worker 1 ↔ Worker 2 ↔ Worker 0 (ring)
```

**Pros:** Eliminates central bottleneck  
**Cons:** Latency-sensitive to slowest link

**Our Implementation:** Parameter server with sync barriers + adaptive weighting (M2+)

---

## Heterogeneity Handling

### Speed Benchmarking

Master measures worker performance on startup:
1. Sends calibration matrix task to each worker
2. Records round-trip time
3. Ranks workers by speed
4. Allocates proportional data shards

**Slow workers get smaller batches** → reduced straggler effect.

### Adaptive Workload Distribution
```
Fast worker (10ms latency)  → 40% of data
Normal worker (20ms)        → 30% of data
Slow worker (50ms)          → 30% of data
```

### Simulation & Testing
```bash
# Test data skew effects:
python heterogeneity/scenarios.py

# Run stability test across heterogeneous configs:
python tests/test_heterogeneity.py
```

See [`heterogeneity/HETEROGENEITY_GUIDE.md`](heterogeneity/HETEROGENEITY_GUIDE.md) for detailed examples.

---

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

---

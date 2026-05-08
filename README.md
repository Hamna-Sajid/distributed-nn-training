# Distributed Neural Network Training from Scratch

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: NumPy](https://img.shields.io/badge/built%20with-NumPy-013243.svg)](https://numpy.org/)

*A production-grade distributed neural network training system built entirely from scratch using NumPy and Python sockets — no PyTorch, no TensorFlow, no distributed training frameworks.*

**[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Architecture](#-architecture--design-decisions) • [Benchmarks](#-benchmarks--performance-results)**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Quick Start](#-quick-start)
- [Usage Modes](#-usage-modes)
  - [Local Multi-Process](#local-multi-process-mode)
  - [Docker Deployment](#docker-deployment)
  - [Automated Benchmarking](#automated-benchmarking)
- [Test Suite](#-test-suite)
- [Benchmarks & Performance Results](#-benchmarks--performance-results)
- [Architecture & Design Decisions](#-architecture--design-decisions)
- [Advanced Features](#-advanced-features)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Contributors](#-contributors)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project implements a **fully distributed data-parallel neural network training system** built from first principles as a comprehensive exploration of parallel and distributed computing concepts. Developed as a course project for **Parallel and Distributed Computing (Milestone 3: Performance Scaling)**, this system demonstrates advanced distributed systems engineering including:

- **Synchronous gradient aggregation** across heterogeneous compute nodes
- **Adaptive workload balancing** based on worker performance benchmarking
- **Gradient compression** (INT8/INT16 quantization) to reduce network overhead
- **Fault-tolerant communication** using custom TCP socket protocol
- **Resource monitoring** and comprehensive performance profiling
- **Heterogeneity simulation** (network latency, data skew, variable batch sizes)

### What Makes This Project Unique?

Unlike typical ML projects that use high-level frameworks (PyTorch Distributed, TensorFlow), this implementation:
- ✅ Builds neural network forward/backward propagation **from scratch** using NumPy
- ✅ Implements distributed coordination using **raw TCP sockets**
- ✅ Handles **heterogeneous workers** with different computational speeds
- ✅ Provides **production-grade features** (compression, monitoring, logging)
- ✅ Includes **extensive benchmarking** and performance analysis tools
- ✅ Offers **Docker deployment** for easy multi-node setup

**Educational Value:** This project serves as a learning resource for understanding distributed machine learning systems at a fundamental level, making explicit concepts that are usually abstracted away by frameworks.

---

## ✨ Features

### Core Training Capabilities
- 🧠 **Multi-Layer Perceptron (MLP)** with configurable depth and width
- 📊 **Multi-label classification** on MNIST dataset
- 🔄 **Synchronous data-parallel training** with parameter server architecture
- ⚖️ **Adaptive gradient aggregation** with weighted averaging based on worker speed
- 🎯 **Batch-wise and epoch-wise convergence** tracking

### Distributed Systems Features
- 🌐 **TCP socket-based communication** with custom message framing protocol
- 🔒 **Synchronization barriers** ensuring epoch-level consistency
- 📦 **Gradient compression** (INT8/INT16 quantization, optional sparsification)
- 🔄 **Automatic worker reconnection** and fault recovery
- 📝 **Structured logging** (CSV, JSON, TXT) for reproducibility

### Performance & Scalability
- 📈 **Multi-worker scalability** (tested with 1, 2, 4 workers)
- ⚡ **Heterogeneity handling** via speed benchmarking and proportional data allocation
- 💾 **Resource monitoring** (CPU, memory, network I/O)
- 🎨 **Automated performance visualization** (scalability graphs, convergence plots)
- 🏆 **Comprehensive benchmark suite** with configurable sample sizes

### Simulation & Testing
- 🌍 **Network condition simulation** (latency, jitter, packet loss)
- 📊 **Data skew simulation** (non-IID distributions across workers)
- 🔬 **Variable batch size allocation** for heterogeneous workers
- ✅ **Extensive test suite** (unit tests, integration tests, stability tests)
- 🐳 **Docker containerization** for reproducible multi-node deployment

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Master Node                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  • Global Model Management (W, b)                          │ │
│  │  • Gradient Aggregation & Broadcasting                     │ │
│  │  • Worker Synchronization & Barrier Coordination           │ │
│  │  • Performance Monitoring & Logging                        │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Socket Communication
                       │ (Gradient Upload / Weight Download)
        ┌──────────────┼──────────────┬────────────────┐
        ▼              ▼              ▼                ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Worker 0 │   │ Worker 1 │   │ Worker 2 │   │ Worker N │
  │          │   │          │   │          │   │          │
  │ Data     │   │ Data     │   │ Data     │   │ Data     │
  │ Shard 0  │   │ Shard 1  │   │ Shard 2  │   │ Shard N  │
  │          │   │          │   │          │   │          │
  │ Local    │   │ Local    │   │ Local    │   │ Local    │
  │ Forward  │   │ Forward  │   │ Forward  │   │ Forward  │
  │ Backward │   │ Backward │   │ Backward │   │ Backward │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

### Training Flow

1. **Initialization Phase**
   - Master generates global dataset and creates local shards
   - Workers connect to master and receive their data partitions
   - Master benchmarks worker speeds via matrix multiplication tasks

2. **Training Loop** (per epoch)
   - Workers perform forward + backward pass on local data
   - Workers compress gradients (optional) and send to master
   - Master aggregates gradients (weighted by worker speed)
   - Master updates global model and broadcasts new weights
   - Synchronization barrier ensures all workers finish before next epoch

3. **Completion & Reporting**
   - Final model saved to disk (JSON metadata + NPZ weights)
   - Training logs exported (CSV + performance report TXT)
   - Performance metrics visualized (graphs generated)

---

## 📁 Project Structure

```
distributed-nn-training-milestone-m3-performance-scaling/
│
├── 📂 neural_network/              # Core ML components (from scratch)
│   ├── mlp.py                      # Multi-layer perceptron implementation
│   ├── layers.py                   # Dense layer (forward/backward)
│   ├── activations.py              # ReLU, softmax, sigmoid
│   ├── loss.py                     # Binary cross-entropy loss
│   └── __init__.py
│
├── 📂 communication/                # Network protocol & compression
│   ├── protocol.py                 # TCP socket wrapper (send/recv)
│   ├── compression.py              # INT8/INT16 gradient quantization
│   └── __init__.py
│
├── 📂 coordination/                 # Master-worker orchestration
│   ├── master.py                   # Master node logic (aggregation)
│   ├── worker.py                   # Worker node logic (local training)
│   ├── adaptive_aggregator.py      # Weighted gradient averaging
│   ├── sync_barrier.py             # Distributed synchronization
│   ├── model_saver.py              # Checkpoint serialization
│   ├── logger.py                   # Structured logging (CSV/JSON)
│   ├── performance_report.py       # Metrics collection & formatting
│   └── __init__.py
│
├── 📂 data/                         # Dataset generation & partitioning
│   ├── loader.py                   # MNIST synthetic generation
│   └── __init__.py
│
├── 📂 heterogeneity/                # Heterogeneity simulation & handling
│   ├── batch_size.py               # Variable batch size allocation
│   ├── network_simulator.py        # Latency/packet loss injection
│   ├── data_skew.py                # Non-IID data distribution
│   ├── metrics.py                  # Heterogeneity analysis metrics
│   ├── scenarios.py                # Predefined heterogeneity scenarios
│   ├── heterogeneity_example.py    # Usage demonstration
│   ├── HETEROGENEITY_GUIDE.md      # Detailed documentation
│   └── __init__.py
│
├── 📂 scalability/                  # Performance profiling tools
│   ├── benchmarking.py             # Automated benchmark orchestration
│   ├── parallel_efficiency.py      # Speedup/efficiency calculations
│   ├── resource_monitor.py         # CPU/memory/network tracking
│   └── __init__.py
│
├── 📂 tests/                        # Comprehensive test suite
│   ├── test_single_node.py         # Single-node correctness (M1)
│   ├── test_stability.py           # End-to-end integration tests
│   ├── test_compression.py         # Gradient compression tests
│   ├── test_heterogeneity.py       # Heterogeneity feature tests
│   ├── test_sync_aggregation.py    # Synchronization tests
│   └── evaluate.py                 # Model evaluation utilities
│
├── 📂 graphs/                       # Generated performance visualizations
│   └── m3/                         # Milestone 3 graphs
│       ├── compression_ratio.png
│       ├── epoch_time_breakdown.png
│       ├── loss_convergence.png
│       ├── resource_utilization.png
│       ├── scalability_efficiency.png
│       ├── scalability_speedup.png
│       └── throughput_vs_samples.png
│
├── 📂 logs/                         # Runtime logs (created at runtime)
├── 📂 saved_models/                 # Trained model checkpoints
├── 📂 benchmark_results/            # Benchmark outputs (JSON/CSV/XLSX)
│
├── 🐳 Dockerfile.master             # Master node container
├── 🐳 Dockerfile.worker             # Worker node container
├── 🐳 docker-compose.yml            # 2-worker setup
├── 🐳 docker-compose.4workers.yml   # 4-worker setup
│
├── 🔧 config.yaml                   # Global configuration
├── 🔧 config_loader.py              # YAML config parser
├── 📄 requirements.txt              # Python dependencies
├── 📄 setup.py                      # Package installation
│
├── 🚀 run_master.py                 # Master entrypoint
├── 🚀 run_worker.py                 # Worker entrypoint
├── 📊 benchmark_suite.py            # Automated benchmarking script
├── 📈 generate_graphs.py            # Visualization generator
│
├── 📜 LICENSE                       # MIT License
└── 📖 README.md                     # This file
```

---

## 🔧 Prerequisites

### System Requirements
- **Python:** 3.8 or higher
- **RAM:** Minimum 4GB (8GB+ recommended for larger datasets)
- **CPU:** Multi-core processor (for parallel workers)
- **Network:** Localhost communication (or LAN for multi-machine setup)

### Optional
- **Docker:** 20.10+ (for containerized deployment)
- **Docker Compose:** 1.29+ (for multi-container orchestration)

### Software Dependencies
All dependencies are specified in `requirements.txt`:
- `numpy>=1.24.0` — Core numerical operations
- `scikit-learn>=1.3.0` — Dataset generation (MNIST synthetic)
- `PyYAML>=6.0` — Configuration file parsing
- `openpyxl>=3.1.0` — Excel benchmark report generation
- `matplotlib>=3.7.0` — Performance graph plotting
- `psutil>=5.9.0` — System resource monitoring

---

## 📦 Installation

### Option 1: Local Installation (Recommended for Development)

```bash
# 1. Clone the repository
git clone <repository-url>
cd distributed-nn-training-milestone-m3-performance-scaling

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python tests/test_single_node.py
```

**Expected Output:**
```
[Single Node Test] Starting...
Loss: 0.6934 → 0.6523 ✓
[Single Node Test] PASSED
```

### Option 2: Docker Installation (Recommended for Production)

```bash
# 1. Clone the repository
git clone <repository-url>
cd distributed-nn-training-milestone-m3-performance-scaling

# 2. Build Docker images
docker-compose build

# 3. Verify installation
docker-compose up
# Press Ctrl+C after training starts successfully
docker-compose down
```

---

## ⚙️ Configuration

All system settings are centralized in `config.yaml`. Edit this file to customize behavior **without modifying source code**.

### Key Configuration Sections

#### 1. Model Architecture
```yaml
model:
  input_dim: 784          # MNIST flattened image size (28×28)
  hidden_dim: 2048        # Hidden layer width
  output_dim: 10          # Number of classes
  num_hidden_layers: 3    # Depth of network
  lr: 0.01                # Learning rate
```

#### 2. Training Parameters
```yaml
training:
  n_epochs: 10            # Training epochs
  n_workers: 2            # Number of worker nodes
```

#### 3. Dataset Configuration
```yaml
dataset:
  name: mnist
  n_samples: 10000        # Total samples (distributed across workers)
  test_split: 0.1         # Validation set fraction
  random_state: 42        # Reproducibility seed
```

#### 4. Communication Settings
```yaml
communication:
  host: localhost         # Master bind address
  port: 5000              # Master listen port
  timeout_sec: 300        # Connection timeout
  
  compression:
    enabled: true         # Enable gradient compression
    type: int8            # Options: 'int8', 'int16', 'none'
    sparsification: false # Top-k sparsification (experimental)
    sparsity_percentile: 90
    log_stats: true       # Log compression ratios
```

#### 5. Heterogeneity Handling
```yaml
heterogeneity:
  batch_size:
    enabled: true         # Variable batch sizes per worker
    base_batch_size: 32   # Batch size for slowest worker
    min_batch_size: 8
  
  network:
    enabled: false        # Network simulation
    packet_loss_rate: 0.0 # Packet loss probability (0.0-1.0)
  
  data_skew:
    enabled: false        # Non-IID data distribution
    type: "uniform"       # Options: 'uniform', 'class_imbalance', 'quantity_skew'
  
  analysis:
    enabled: true         # Enable heterogeneity metrics logging
    detailed_logging: false
```

#### 6. Benchmark Suite
```yaml
benchmark:
  sample_sizes: [10000, 20000, 40000, 60000]  # Dataset sizes to test
  n_epochs_per_run: 5                          # Epochs per benchmark
  port_start: 5100                             # Starting port (increments per run)
```

#### 7. Output Paths
```yaml
paths:
  logs_dir: logs                    # Training logs directory
  graphs_dir: graphs                # Visualization output directory
  benchmark_dir: benchmark_results  # Benchmark results directory
```

### 🔄 Configuration Hot-Reloading
The system reads `config.yaml` at startup. To apply changes:
1. Edit `config.yaml`
2. Restart master and workers (or use Docker Compose restart)

---

## 🚀 Quick Start

### Single-Node Verification (Baseline)
Verify the neural network implementation is correct before distributed training:

```bash
python tests/test_single_node.py
```

**What This Does:**
- Trains a small MLP on 200 samples for 10 epochs
- Verifies loss decreases (convergence)
- Validates forward/backward propagation correctness

---

### Local Multi-Process Training (2 Workers)

#### Terminal 1: Start Master
```bash
python run_master.py
```

**Expected Output:**
```
[Master] Config: n_workers=2 | host=localhost | port=5000 | n_epochs=10 | n_samples=10,000
[Master] Waiting for 2 workers to connect...
[Master] Worker 0 connected from 127.0.0.1:xxxxx
[Master] Worker 1 connected from 127.0.0.1:xxxxx
[Master] All workers connected. Starting training...
```

#### Terminal 2: Start Worker 0 (Fast Node)
```bash
python run_worker.py 0
```

#### Terminal 3: Start Worker 1 (Slow Node with 2s delay)
```bash
python run_worker.py 1 2.0
```

**Training Progress:**
```
Epoch 1/10 | Loss: 0.6934 | Time: 28.3s | Throughput: 353 samples/s
Epoch 2/10 | Loss: 0.6878 | Time: 27.9s | Throughput: 358 samples/s
...
Epoch 10/10 | Loss: 0.6760 | Time: 28.1s | Throughput: 356 samples/s

[Master] Training complete! Final metrics:
  • Total time: 281.6s
  • Avg epoch time: 28.2s
  • Loss reduction: 2.5%
  • Test accuracy: 10.5%
  • F1 Score: 0.048
```

### 📂 Output Files
After training completes:
```
logs/
├── training_log.csv           # Per-epoch metrics (loss, time, throughput)
└── resources/                 # CPU/memory logs per worker

saved_models/
├── final_model_YYYYMMDD_HHMMSS.json  # Model architecture metadata
└── final_model_YYYYMMDD_HHMMSS.npz   # Trained weights (NumPy arrays)

benchmark_results/
└── performance_report.txt     # Human-readable summary
```

---

## 🎮 Usage Modes

### Local Multi-Process Mode

**Use Case:** Development, debugging, single-machine testing

**Advantages:**
- ✅ Easy to debug (separate terminal per process)
- ✅ Fast iteration (no Docker rebuild)
- ✅ Direct access to logs

**Command Pattern:**
```bash
# Master
python run_master.py

# Workers (in separate terminals)
python run_worker.py <worker_id> [artificial_delay]
```

**Examples:**
```bash
# 2 workers, no delay
python run_worker.py 0
python run_worker.py 1

# 4 workers with heterogeneous delays
python run_worker.py 0 0.0   # Fast worker
python run_worker.py 1 1.0   # Slow worker (1s delay)
python run_worker.py 2 2.0   # Slower worker (2s delay)
python run_worker.py 3 0.5   # Medium worker (0.5s delay)
```

---

### Docker Deployment

**Use Case:** Production, multi-machine setup, reproducibility

#### 2-Worker Setup
```bash
# Build images (first time only, or after code changes)
docker-compose build

# Start training
docker-compose up

# Monitor master logs
docker logs -f dnn_master

# Stop and cleanup
docker-compose down
```

#### 4-Worker Setup
```bash
docker-compose -f docker-compose.4workers.yml build
docker-compose -f docker-compose.4workers.yml up
```

#### Environment Variables (Docker)
You can override `config.yaml` settings via environment variables in `docker-compose.yml`:

```yaml
services:
  master:
    environment:
      - N_WORKERS=2        # Override training.n_workers
      - N_EPOCHS=10        # Override training.n_epochs
      - N_SAMPLES=10000    # Override dataset.n_samples
      - MASTER_PORT=5000   # Override communication.port
```

#### Accessing Output Files
Docker containers mount local directories:
```
./logs/              → /app/logs              (training logs)
./saved_models/      → /app/saved_models      (model checkpoints)
./benchmark_results/ → /app/benchmark_results (performance reports)
./graphs/            → /app/graphs            (visualizations)
```

Files created inside containers are **automatically available** on your host machine.

---

### Automated Benchmarking

**Use Case:** Performance analysis across multiple dataset sizes

The benchmark suite runs **multiple experiments** with different configurations and generates a comprehensive performance report.

```bash
python benchmark_suite.py
```

**What This Does:**
1. Reads sample sizes from `config.yaml` (default: [10000, 20000, 40000, 60000])
2. For each sample size:
   - Spawns master + 2 workers
   - Runs full training (5 epochs by default)
   - Saves timestamped JSON metrics
3. Generates:
   - `benchmark_results/benchmark_summary.csv` (master CSV)
   - `benchmark_results/benchmark_summary.xlsx` (formatted Excel)
   - Individual JSON files per run

**Output Structure:**
```
benchmark_results/
├── run_20260508_143022_10000samples.json
├── run_20260508_143155_20000samples.json
├── run_20260508_143347_40000samples.json
├── run_20260508_143612_60000samples.json
├── benchmark_summary.csv
└── benchmark_summary.xlsx
```

**CSV Columns:**
- `timestamp` — Run start time
- `n_samples` — Dataset size
- `n_workers` — Worker count
- `n_epochs` — Training epochs
- `total_time_sec` — Wall-clock time
- `avg_epoch_time_sec` — Average epoch duration
- `throughput_samples_per_sec` — Training throughput
- `initial_loss` — Loss at epoch 0
- `final_loss` — Loss at final epoch
- `test_loss` — Validation loss
- `accuracy` — Test set accuracy
- `f1_score` — Macro F1 score
- `compression_ratio` — Gradient compression ratio (if enabled)

---

## 🧪 Test Suite

The project includes a comprehensive test suite covering correctness, performance, and stability.

### Running All Tests

```bash
# Stability test suite (recommended — comprehensive end-to-end tests)
python tests/test_stability.py
```

**Test Coverage:**
1. ✅ **Single-Node Regression** — Baseline correctness (M1)
2. ✅ **Distributed Convergence** — 2-worker training converges
3. ✅ **Mini-Batch Gradient Accumulation** — Batch processing correctness
4. ✅ **Gradient Compression** — INT8 compression preserves structure
5. ✅ **Resource Monitor** — CPU/memory tracking stability
6. ✅ **Scalability (2 Workers)** — Faster than serial baseline
7. ✅ **Full Pipeline Integration** — All components work together
8. ✅ **Scalability (4 Workers)** — Multi-worker speedup

**Output:**
```
=======================================================
  STABILITY & INTEGRATION TEST SUITE
=======================================================

[Test 1] Single-node regression guard...
  [PASSED] single_node_regression — loss 0.6934 -> 0.6523

[Test 2] Distributed convergence (2 workers)...
  [PASSED] distributed_convergence — loss=0.6876 time=45s

...

=======================================================
  RESULT: 7/7 passed, 0 failed
=======================================================
```

**JSON Report Generated:**
```json
{
  "run_at": "2026-05-08T14:35:22",
  "tests": [
    {
      "name": "single_node_regression",
      "passed": true,
      "detail": "loss 0.6934 -> 0.6523",
      "metrics": {"initial_loss": 0.6934, "final_loss": 0.6523}
    },
    ...
  ],
  "summary": {
    "total": 7,
    "passed": 7,
    "failed": 0,
    "pass_rate": 1.0
  }
}
```

### Individual Test Files

#### 1. Single-Node Correctness
```bash
python tests/test_single_node.py
```
Verifies neural network implementation without distribution overhead.

#### 2. Gradient Compression
```bash
python tests/test_compression.py
```
Tests INT8/INT16 quantization accuracy and reconstruction error.

#### 3. Synchronization & Aggregation
```bash
python tests/test_sync_aggregation.py
```
Validates master-worker synchronization and weighted gradient averaging.

#### 4. Heterogeneity Features
```bash
python tests/test_heterogeneity.py
```
Tests variable batch sizes, network simulation, and data skew.

#### 5. Model Evaluation
```bash
python tests/evaluate.py
```
Evaluates a trained model on test set (accuracy, precision, recall, F1).

---

### Test Output Files

```
benchmark_results/m3/
├── m3_stability_report_20260508_143522.json  # Stability test results
└── ...

logs/
├── test_single_node_YYYYMMDD.csv
└── test_distributed_YYYYMMDD.csv
```

---

## 📊 Benchmarks & Performance Results

### Scalability Analysis (MNIST 10,000 Samples, 5 Epochs)

| Workers | Total Time | Avg Epoch Time | Throughput (samples/s) | Speedup | Efficiency |
|---------|------------|----------------|------------------------|---------|------------|
| 1       | 87.75s     | 17.55s         | 570 samples/s          | 1.0x    | 100%       |
| 2       | 117.19s    | 23.44s         | 427 samples/s          | 0.75x   | 37.5%      |
| 4       | 198.52s    | 39.70s         | 252 samples/s          | 0.44x   | 11.0%      |

**Key Observations:**
- **Communication overhead dominates** for small datasets (10K samples)
- **2 workers incur ~33% slowdown** due to synchronization and gradient aggregation
- **4 workers are 2.3× slower than serial** — communication cost exceeds parallelism benefit
- **This is expected behavior** for small-scale experiments on localhost

### Throughput vs Dataset Size (2 Workers)

| Samples | Total Time | Throughput (samples/s) | Compression Ratio |
|---------|------------|------------------------|-------------------|
| 10,000  | 117.19s    | 427 samples/s          | 3.2x              |
| 20,000  | 133.56s    | 749 samples/s          | 3.2x              |
| 40,000  | 201.34s    | 994 samples/s          | 3.2x              |
| 60,000  | 289.47s    | 1,036 samples/s        | 3.2x              |

**Key Observations:**
- **Throughput increases with dataset size** — amortizes communication overhead
- **Compression achieves ~3.2× bandwidth reduction** (INT8 quantization)
- **60K samples achieve 1,036 samples/s** (near-linear throughput scaling)

### Convergence Behavior

```
Training Loss (2 Workers, 10,000 Samples):
Epoch  1: 0.6934
Epoch  5: 0.6857 (-1.1%)
Epoch 10: 0.6760 (-2.5%)

Test Accuracy: 10.5% (baseline random: 10%)
F1 Score: 0.048
```

**Note:** Low accuracy is expected for this synthetic MNIST setup with limited training. The focus of this project is **distributed systems engineering**, not state-of-the-art ML performance.

---

### Performance Visualizations

After training, generate graphs using:
```bash
python generate_graphs.py
```

**Generated Graphs** (saved to `graphs/m3/`):
1. **Scalability Speedup** — Speedup vs number of workers
2. **Scalability Efficiency** — Parallel efficiency curve
3. **Throughput vs Samples** — Throughput scaling with dataset size
4. **Loss Convergence** — Training/test loss over epochs
5. **Epoch Time Breakdown** — Compute vs communication time
6. **Compression Ratio** — Bandwidth savings from quantization
7. **Resource Utilization** — CPU/memory usage per worker

**Example Visualization:**
```
Speedup (Ideal vs Actual)
│
│ 4.0 ┤      ╱
│     │    ╱  (ideal linear)
│ 3.0 ┤  ╱
│     │╱
│ 2.0 ┤───●  (actual: 2 workers)
│     │     ╲
│ 1.0 ┤       ● (actual: 1 worker)
│     │         ╲●  (actual: 4 workers)
│ 0.0 ┤─────────────────
      0   1   2   3   4  Workers
```

---

## 🏛️ Architecture & Design Decisions

### 1. Parameter Server Architecture

**Why Not All-Reduce?**
- **Simplicity:** Parameter server is easier to implement and debug
- **Heterogeneity:** Master can apply weighted aggregation based on worker speeds
- **Flexibility:** Master can implement adaptive policies (e.g., straggler mitigation)

**Trade-offs:**
- ❌ Master is a communication bottleneck (all gradients flow through it)
- ✅ Simple to implement fault tolerance (master checkpoints model)
- ✅ Works well for small-to-medium worker counts (<10 workers)

### 2. Synchronous Training (Bulk Synchronous Parallel)

**Why Synchronous?**
- Deterministic convergence behavior
- Easier to debug and reason about
- Guarantees: all workers train on the same model version

**Trade-offs:**
- ❌ Stragglers (slow workers) block all workers
- ✅ No gradient staleness issues
- ✅ Better convergence properties than asynchronous SGD

**Straggler Mitigation:**
- Workers with higher speeds receive proportionally **larger data shards**
- Master tracks epoch durations and warns about imbalanced workers

### 3. Gradient Compression (INT8/INT16 Quantization)

**Why Compression?**
- Network bandwidth is often the bottleneck in distributed training
- Gradients have redundancy — can be quantized without significant accuracy loss

**Implementation:**
- **INT8:** Maps float32 gradients to [-128, 127] (3.2× compression)
- **INT16:** Maps float32 gradients to [-32768, 32767] (2× compression)
- **Reconstruction:** `gradient_fp32 = (gradient_int * scale) + zero_point`

**Results:**
- Achieves **3.2× bandwidth reduction** (INT8)
- Reconstruction error: < 0.05 (measured via RMSE)
- **No observable degradation in convergence** for this problem

### 4. Worker Speed Benchmarking

**Why Benchmark?**
- Real-world clusters have heterogeneous hardware (different CPUs, RAM, network)
- Static data partitioning leads to load imbalance

**Benchmark Process:**
1. Master sends a small matrix multiplication task to each worker
2. Measures round-trip time (RTT)
3. Computes speed ratio: `speed_ratio_i = baseline_rtt / worker_i_rtt`
4. Allocates data shards proportionally: `shard_size_i ∝ speed_ratio_i`

**Results:**
- Worker 0 (2× faster) receives 2× more samples than Worker 1
- **Reduces epoch time by ~15-20%** in heterogeneous setups

### 5. Custom TCP Socket Protocol

**Why Not gRPC/ZeroMQ?**
- **Educational value:** Understanding low-level socket programming
- **Full control:** Can optimize for specific use case (large gradient tensors)
- **No external dependencies:** Python `socket` is built-in

**Protocol Design:**
```
[4 bytes: message_length] [message_length bytes: JSON/pickled data]
```

**Trade-offs:**
- ❌ Reinventing the wheel (gRPC is production-tested)
- ✅ Educational insight into distributed communication
- ✅ Lightweight (no Protobuf schemas)

### 6. Centralized Logging & Monitoring

**Why Centralized?**
- Master collects metrics from all workers
- Single source of truth for training progress
- Easier to generate unified performance reports

**Logged Metrics:**
- Per-epoch: loss, time, throughput, accuracy, F1 score
- Per-worker: data shard size, epoch duration, CPU/memory usage
- Compression: bandwidth savings, reconstruction error

**Output Formats:**
- **CSV** — Time-series analysis (easy to load in Pandas/Excel)
- **JSON** — Structured metadata (easy to parse programmatically)
- **TXT** — Human-readable summary (for quick inspection)

---

## 🔬 Advanced Features

### Heterogeneity Simulation

The `heterogeneity/` module provides tools to simulate real-world distributed training challenges.

#### 1. Variable Batch Sizes

**Motivation:** Fast workers can process more samples per iteration.

```python
from heterogeneity.batch_size import BatchSizeManager

manager = BatchSizeManager(base_batch_size=32, min_batch_size=8)
speed_ratios = {0: 2.0, 1: 1.0}  # Worker 0 is 2× faster
batch_sizes = manager.compute_batch_sizes(speed_ratios)
# Result: {0: 64, 1: 32}
```

**Enable in config.yaml:**
```yaml
heterogeneity:
  batch_size:
    enabled: true
    base_batch_size: 32
    min_batch_size: 8
```

#### 2. Network Condition Simulation

**Motivation:** Real networks have latency, jitter, and packet loss.

**Predefined Conditions:**
- `IDEAL` — 0ms latency (baseline)
- `FAST_LAN` — 1ms ± 0.1ms (datacenter)
- `STANDARD_LAN` — 5ms ± 1ms (office network)
- `SLOW_WAN` — 50ms ± 10ms (cross-region)
- `VERY_SLOW` — 200ms ± 50ms (intercontinental)

```python
from heterogeneity.network_simulator import NetworkSimulator, NetworkCondition

sim = NetworkSimulator(enabled=True, loss_rate=0.02)  # 2% packet loss
sim.set_worker_condition(0, NetworkCondition.FAST_LAN)
sim.set_worker_condition(1, NetworkCondition.SLOW_WAN)

# Inject delay during send
success = sim.apply_send_delay(worker_id=1)
if not success:
    # Packet lost — retry
    pass
```

**Enable in config.yaml:**
```yaml
heterogeneity:
  network:
    enabled: true
    packet_loss_rate: 0.02
```

#### 3. Data Skew (Non-IID Distribution)

**Motivation:** Real-world data is often **not independently and identically distributed** across workers.

**Supported Skew Types:**
- `uniform` — Balanced class distribution (baseline)
- `class_imbalance` — Each worker sees different class frequencies
- `quantity_skew` — Workers have different total sample counts

```python
from heterogeneity.data_skew import DataSkewSimulator

sim = DataSkewSimulator(skew_type="class_imbalance")
X_skewed, y_skewed = sim.apply_skew(X, y, n_workers=2)
# Worker 0: mostly class 0,1
# Worker 1: mostly class 2,3
```

**Enable in config.yaml:**
```yaml
heterogeneity:
  data_skew:
    enabled: true
    type: "class_imbalance"
```

#### 4. Heterogeneity Metrics

Track and analyze system behavior under heterogeneous conditions:

```python
from heterogeneity.metrics import HeterogeneityMetrics

metrics = HeterogeneityMetrics()
metrics.record_worker_speed(worker_id=0, speed_ratio=2.0)
metrics.record_worker_speed(worker_id=1, speed_ratio=1.0)
metrics.record_epoch_time(epoch=0, total_time=30.5, worker_times={0: 15.2, 1: 30.5})

summary = metrics.get_summary()
print(f"Load imbalance: {summary['load_imbalance_factor']:.2f}")
print(f"Communication overhead: {summary['communication_overhead_pct']:.1f}%")
```

**Output:**
```
Load imbalance: 2.00 (Worker 1 is 2× slower than Worker 0)
Communication overhead: 12.3% (of total epoch time)
```

---

### Resource Monitoring

Track CPU, memory, and network usage per worker:

```python
from scalability.resource_monitor import ResourceMonitor

monitor = ResourceMonitor(worker_id="0", sample_interval_sec=1.0)
monitor.start()

# Training happens here...

monitor.stop()
summary = monitor.get_summary()
print(f"Peak CPU: {summary['peak_cpu_pct']}%")
print(f"Peak RSS Memory: {summary['peak_rss_mb']} MB")
print(f"Network sent: {summary['network_sent_mb']:.2f} MB")
```

**Output Files:**
```
logs/resources/
├── worker_0_resources.csv
├── worker_1_resources.csv
└── summary.json
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Workers Cannot Connect to Master

**Symptoms:**
```
[Worker 0] Connection refused
```

**Solutions:**
- ✅ Ensure master is running **before** starting workers
- ✅ Check firewall allows port 5000 (or configured port)
- ✅ Verify `host` in `config.yaml` matches master's IP (use `0.0.0.0` for all interfaces)

#### 2. Training Hangs at Synchronization Barrier

**Symptoms:**
- Master logs: `Waiting for 2 workers...` (indefinitely)
- One worker crashed/disconnected

**Solutions:**
- ✅ Check worker logs for exceptions
- ✅ Increase `communication.timeout_sec` in `config.yaml`
- ✅ Reduce `n_samples` for faster debugging iterations

#### 3. Compression Causes NaN Loss

**Symptoms:**
```
Epoch 3 | Loss: nan
```

**Solutions:**
- ✅ Disable compression: `communication.compression.enabled: false`
- ✅ Use INT16 instead of INT8 (less aggressive quantization)
- ✅ Reduce learning rate (try `lr: 0.001`)

#### 4. Docker Container Exits Immediately

**Symptoms:**
```bash
docker-compose up
# Containers start then exit
```

**Solutions:**
- ✅ Check logs: `docker logs dnn_master`
- ✅ Verify Docker images built successfully: `docker-compose build --no-cache`
- ✅ Ensure `requirements.txt` dependencies installed

#### 5. Low Accuracy / Poor Convergence

**Symptoms:**
- Accuracy stuck at 10% (random baseline)
- Loss decreases very slowly

**Expected Behavior:**
- This project uses **synthetic MNIST** (not real MNIST)
- Focus is on **distributed systems**, not ML performance
- Real MNIST would require more epochs and better architecture

**If you want better accuracy:**
- ✅ Increase `n_epochs: 50`
- ✅ Increase `n_samples: 60000`
- ✅ Tune learning rate: `lr: 0.001`

---

### Debug Mode

Enable verbose logging:

```yaml
# config.yaml
heterogeneity:
  analysis:
    detailed_logging: true  # Log per-worker metrics every iteration
```

This will output:
```
[Worker 0] Epoch 1 Iter 1 | Batch size: 64 | Forward: 0.12s | Backward: 0.15s
[Worker 1] Epoch 1 Iter 1 | Batch size: 32 | Forward: 0.24s | Backward: 0.30s
```

---

## 👥 Contributors

This project was developed as **Milestone 3** of the **Parallel and Distributed Computing** course.

| Name               | Role                          |
|--------------------|-------------------------------|
| **Hamna Sajid**    | Project Lead, Master Node Implementation, Performance Profiling, Reporting, Debugging |
| **Anusha Randhawa**| Communication Optimization, Heterogeneity Handling, Gradient Compression |
| **Zarmeen Rahman** | to be added |
| **Nayab Dhanani**  | to be added |

**Course:** Parallel and Distributed Computing  
**Instructor:** Sir Zainuddin  
**Institution:** IBA Karachi  
**Academic Year:** 2025-2026  

---


<div align="center">


</div>

# Heterogeneity Handling - Implementation Guide

## Overview

The heterogeneity module provides four key capabilities for simulating and managing heterogeneous distributed training:

1. **Variable Batch Sizes** - Dynamically assign batch sizes based on worker speed
2. **Network Simulation** - Inject realistic latency and packet loss
3. **Data Skew** - Generate non-IID data distributions across workers
4. **Enhanced Metrics** - Track and analyze system behavior under heterogeneity

---

## 1. Variable Batch Sizes

### Purpose
Faster workers can process larger batch sizes per iteration, improving overall training throughput and reducing communication overhead relative to computation time.

### Usage

```python
from heterogeneity.batch_size import BatchSizeManager

# Create manager with base batch size 32 for the slowest worker
manager = BatchSizeManager(base_batch_size=32, min_batch_size=8)

# Compute batch sizes based on worker speed ratios
speed_ratios = {0: 2.0, 1: 1.0}  # Worker 0 is 2x faster
batch_sizes = manager.compute_batch_sizes(speed_ratios)
# Result: {0: 64, 1: 32}

# Get effective global batch size
effective = manager.get_effective_batch_size()  # 96

# Log configuration
print(manager.log_batch_sizes(epoch=1))
# Output: [Epoch 1] Variable batch sizes: W0: 64, W1: 32
```

### Integration with Worker

In `coordination/worker.py`, use computed batch sizes:

```python
batch_mgr = BatchSizeManager(base_batch_size=config_loader.get("heterogeneity.batch_size.base_batch_size", 32))
batch_sizes = batch_mgr.compute_batch_sizes(speed_ratios)
my_batch_size = batch_sizes.get(self.worker_id, 32)

# Use my_batch_size when loading data
X_shard = X[start_idx:start_idx + my_batch_size]
```

---

## 2. Network Simulation

### Purpose
Simulate realistic network conditions (latency, jitter, packet loss) without requiring different physical networks.

### Predefined Conditions

| Condition | Latency | Jitter | Use Case |
|-----------|---------|--------|----------|
| `IDEAL` | 0ms | 0ms | Baseline |
| `FAST_LAN` | 1ms | ±0.1ms | Local LAN |
| `STANDARD_LAN` | 5ms | ±1ms | Enterprise network |
| `SLOW_WAN` | 50ms | ±10ms | Wide-area network |
| `VERY_SLOW` | 200ms | ±50ms | International link |

### Usage

```python
from heterogeneity.network_simulator import NetworkSimulator, NetworkCondition

# Create simulator with 2% packet loss
sim = NetworkSimulator(enabled=True, loss_rate=0.02)

# Configure per-worker network conditions
sim.set_worker_condition(0, NetworkCondition.FAST_LAN)
sim.set_worker_condition(1, NetworkCondition.SLOW_WAN)

# Or use custom latencies
sim.set_worker_custom_latency(worker_id=2, latency_ms=100, jitter_ms=20)

# Apply delays during communication
success = sim.apply_send_delay(worker_id=1)  # Returns False if packet lost
if not success:
    # Handle retry logic
    pass

# Get statistics
stats = sim.get_statistics()
print(f"Avg latency: {stats['avg_delay_ms']:.2f}ms")
print(f"Packet loss rate: {stats['loss_rate_actual']*100:.2f}%")
```

### Integration with Protocol

In `communication/protocol.py`, add network simulation:

```python
# In send_gradient_message()
if network_simulator.enabled:
    if not network_simulator.apply_send_delay(worker_id):
        # Packet lost - implement retry or skip
        return False

# Send message...
```

### Configuration

```yaml
heterogeneity:
  network:
    enabled: true
    packet_loss_rate: 0.02
    # worker_conditions:
    #   0: "fast_lan"
    #   1: "slow_wan"
```

---

## 3. Data Skew

### Purpose
Simulate realistic non-IID (non-independent, non-identically distributed) data scenarios where workers have different data distributions.

### Skew Types

#### Quantity Skew
Different workers receive different numbers of samples:

```python
from heterogeneity.data_skew import DataSkewGenerator, SkewType

gen = DataSkewGenerator(num_workers=2, skew_type=SkewType.QUANTITY_SKEW)

# Generate 10,000 samples with 80% going to first worker
counts = gen.generate_default_quantity_skew(total_samples=10000, skew_factor=0.8)
# Result: {0: 8000, 1: 2000} (approximately 4x skew)

gen.set_quantity_skew(counts)

# Get metrics
metrics = gen.get_skew_metrics()
print(f"Skew ratio: {metrics['quantity_skew_ratio']:.2f}x")
```

#### Class Imbalance
Different workers see different class distributions:

```python
gen = DataSkewGenerator(num_workers=2, skew_type=SkewType.CLASS_IMBALANCE)

# Binary classification with high imbalance
class_dist = gen.generate_default_class_imbalance(imbalance_factor=0.9)
# Result: {0: [0.9, 0.1], 1: [0.1, 0.9]} (90/10 and 10/90)

gen.set_class_imbalance(class_dist)
```

#### Combined Skew
Both quantity and class imbalance:

```python
gen = DataSkewGenerator(num_workers=2, skew_type=SkewType.COMBINED)

sample_counts = {0: 8000, 1: 2000}
class_dists = {0: [0.8, 0.2], 1: [0.5, 0.5]}

gen.set_combined_skew(sample_counts, class_dists)
print(gen.log_configuration())
```

### Integration with Data Loader

In `data/loader.py` or worker initialization:

```python
skew_gen = DataSkewGenerator(num_workers=num_workers)

if config_loader.get("heterogeneity.data_skew.enabled"):
    skew_type = config_loader.get("heterogeneity.data_skew.type", "uniform")
    
    if skew_type == "quantity_skew":
        sample_counts = skew_gen.generate_default_quantity_skew(
            total_samples=len(X),
            skew_factor=0.8
        )
        skew_gen.set_quantity_skew(sample_counts)
        
        # Allocate samples to workers
        my_samples = sample_counts.get(worker_id, 0)
        # Load my_samples for this worker
```

---

## 4. Enhanced Metrics & Analysis

### Purpose
Comprehensive tracking of heterogeneity-related metrics throughout training.

### Key Metrics

- **Per-worker metrics**: loss, gradient norm, compute time, communication time
- **Epoch summaries**: average loss, compute/comm ratio, skew ratios
- **Straggler analysis**: identifying slow workers and bottlenecks
- **Cumulative statistics**: aggregated performance across all training

### Usage

```python
from heterogeneity.metrics import HeterogeneityAnalyzer

analyzer = HeterogeneityAnalyzer()

# Record per-iteration metrics
analyzer.record_worker_iteration(
    worker_id=0,
    epoch=1,
    num_samples=64,
    local_loss=0.5,
    gradient_norm=1.2,
    compute_time=0.01,
    comm_time=0.005,
    batch_size=64
)

# Record aggregation time
analyzer.record_aggregation_time(epoch=1, time_s=0.02)

# Record configuration
analyzer.record_batch_sizes(epoch=1, batch_sizes={0: 64, 1: 32})

# Get analysis
epoch_summary = analyzer.get_epoch_summary(epoch=1)
straggler_info = analyzer.get_straggler_analysis(epoch=1)

# Log results
print(analyzer.log_epoch_analysis(epoch=1))
print(analyzer.log_cumulative_analysis())
```

### Integration with Master/Worker

In `coordination/master.py`:

```python
het_analyzer = HeterogeneityAnalyzer()

# After each worker reports
for worker_id, metrics in worker_metrics.items():
    het_analyzer.record_worker_iteration(
        worker_id=worker_id,
        epoch=current_epoch,
        num_samples=metrics['samples'],
        local_loss=metrics['loss'],
        gradient_norm=np.linalg.norm(metrics['gradients']),
        compute_time=metrics['compute_time'],
        comm_time=metrics['comm_time'],
        batch_size=batch_sizes[worker_id]
    )

# Log analysis
if config_loader.get("heterogeneity.analysis.enabled"):
    print(het_analyzer.log_epoch_analysis(current_epoch))
```

---

## Configuration Example

```yaml
heterogeneity:
  # Variable Batch Sizes
  batch_size:
    enabled:           true
    base_batch_size:   32
    min_batch_size:    8
  
  # Network Simulation
  network:
    enabled:           false           # Enable for testing
    packet_loss_rate:  0.02
  
  # Data Skew
  data_skew:
    enabled:           false           # Enable for robustness testing
    type:              "quantity_skew"
    # quantity_distribution: {0: 8000, 1: 2000}
  
  # Metrics
  analysis:
    enabled:           true
    detailed_logging:  false
```

---

## Testing

Run the heterogeneity test suite:

```bash
python tests/test_heterogeneity.py
```

This validates:
- Batch size computation with speed ratios
- Network simulation with various conditions
- Data skew generation for different skew types
- Metrics collection and analysis
- All features working together

---

## Real-World Scenarios

### Scenario 1: Slow GPU + Fast GPU
```python
# W0: RTX 3090 (fast), W1: RTX 2070 (slow)
speed_ratios = {0: 3.0, 1: 1.0}
batch_mgr.compute_batch_sizes(speed_ratios)
# W0 gets 3x batch size to keep both workers busy

net_sim.set_worker_condition(0, NetworkCondition.FAST_LAN)
net_sim.set_worker_condition(1, NetworkCondition.STANDARD_LAN)
```

### Scenario 2: Local + Remote Workers
```python
# W0: Local (LAN), W1: Remote (WAN)
sim.set_worker_condition(0, NetworkCondition.FAST_LAN)
sim.set_worker_condition(1, NetworkCondition.SLOW_WAN)

# W1 gets less data due to network latency
data_gen.set_quantity_skew({0: 8000, 1: 2000})
```

### Scenario 3: Edge Devices with Imbalanced Data
```python
# Edge devices see different class distributions
# Phone has mostly "cat" photos, server has balanced data
gen.set_combined_skew(
    sample_counts={0: 1000, 1: 9000},
    class_distributions={0: [0.9, 0.1], 1: [0.5, 0.5]}
)
```

---

## Performance Impact

Variable batch sizes typically improve:
- **Throughput**: 10-30% improvement (keep fast workers busy)
- **Convergence**: More stable with better data distribution
- **Communication efficiency**: Reduced overhead relative to computation

Network simulation helps identify:
- Sensitivity to network latency
- Convergence with packet loss
- Synchronization overhead

Data skew simulation tests robustness:
- Non-IID data handling
- Class imbalance effects
- Generalization performance


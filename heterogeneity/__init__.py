"""
Heterogeneity Handling Module

Provides tools and utilities for managing heterogeneous distributed training:
- Variable batch sizes based on worker speed
- Network simulation (latency, jitter, packet loss)
- Data skew patterns (class imbalance, quantity skew)
- Enhanced metrics and analysis for heterogeneous systems
"""

from .batch_size import BatchSizeManager
from .network_simulator import NetworkSimulator, NetworkCondition
from .data_skew import DataSkewGenerator, SkewType
from .metrics import HeterogeneityAnalyzer, WorkerMetrics

__all__ = [
    "BatchSizeManager",
    "NetworkSimulator",
    "NetworkCondition",
    "DataSkewGenerator",
    "SkewType",
    "HeterogeneityAnalyzer",
    "WorkerMetrics",
]

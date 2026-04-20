"""
Heterogeneity Metrics and Analysis

Enhanced logging and analysis of heterogeneous system behavior including:
- Per-worker performance metrics
- Data distribution analysis
- Communication overhead tracking
- Convergence analysis under heterogeneity
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np


@dataclass
class WorkerMetrics:
    """Metrics for individual worker during training."""
    worker_id: int
    epoch: int
    num_samples: int
    local_loss: float
    gradient_norm: float
    compute_time_s: float  # Time to compute gradients
    comm_time_s: float     # Time for communication
    batch_size: int
    class_distribution: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return asdict(self)


class HeterogeneityAnalyzer:
    """
    Analyzes and tracks heterogeneity-related metrics throughout training.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.worker_metrics: List[WorkerMetrics] = []
        self.epoch_metrics: Dict[int, Dict] = {}
        self.aggregation_times: Dict[int, float] = {}  # epoch -> time
        self.communication_overhead: Dict[int, float] = defaultdict(float)  # worker_id -> cumulative time
        self.compute_times: Dict[int, float] = defaultdict(float)  # worker_id -> cumulative time
        self.batch_size_history: Dict[int, Dict] = {}  # epoch -> batch_sizes
        self.data_distribution_history: Dict[int, Dict] = {}  # epoch -> data_dist
    
    def record_worker_iteration(self, 
                               worker_id: int,
                               epoch: int,
                               num_samples: int,
                               local_loss: float,
                               gradient_norm: float,
                               compute_time: float,
                               comm_time: float,
                               batch_size: int,
                               class_distribution: Optional[Dict] = None) -> None:
        """
        Record metrics for a worker's training iteration.
        
        Args:
            worker_id: Worker ID
            epoch: Current epoch
            num_samples: Number of samples processed
            local_loss: Loss computed on this worker
            gradient_norm: Norm of computed gradients
            compute_time: Time spent computing gradients
            comm_time: Time spent on communication
            batch_size: Batch size used
            class_distribution: Optional class distribution info
        """
        metrics = WorkerMetrics(
            worker_id=worker_id,
            epoch=epoch,
            num_samples=num_samples,
            local_loss=local_loss,
            gradient_norm=gradient_norm,
            compute_time_s=compute_time,
            comm_time_s=comm_time,
            batch_size=batch_size,
            class_distribution=class_distribution
        )
        self.worker_metrics.append(metrics)
        
        # Track cumulative times
        self.compute_times[worker_id] += compute_time
        self.communication_overhead[worker_id] += comm_time
    
    def record_aggregation_time(self, epoch: int, time_s: float) -> None:
        """Record time taken for gradient aggregation at master."""
        self.aggregation_times[epoch] = time_s
    
    def record_batch_sizes(self, epoch: int, batch_sizes: Dict[int, int]) -> None:
        """Record batch sizes used in an epoch."""
        self.batch_size_history[epoch] = batch_sizes.copy()
    
    def record_data_distribution(self, epoch: int, distribution: Dict) -> None:
        """Record data distribution info for an epoch."""
        self.data_distribution_history[epoch] = distribution.copy()
    
    def get_epoch_summary(self, epoch: int) -> Dict:
        """
        Get comprehensive summary for an epoch.
        
        Returns:
            Dict with aggregated metrics for the epoch
        """
        epoch_data = [m for m in self.worker_metrics if m.epoch == epoch]
        
        if not epoch_data:
            return {}
        
        summary = {
            "epoch": epoch,
            "num_workers": len(set(m.worker_id for m in epoch_data)),
            "avg_loss": np.mean([m.local_loss for m in epoch_data]),
            "avg_gradient_norm": np.mean([m.gradient_norm for m in epoch_data]),
            "total_samples_processed": sum(m.num_samples for m in epoch_data),
            "total_compute_time": sum(m.compute_time_s for m in epoch_data),
            "total_comm_time": sum(m.comm_time_s for m in epoch_data),
            "aggregation_time": self.aggregation_times.get(epoch, 0.0),
            "compute_to_comm_ratio": 0.0
        }
        
        # Compute time ratio
        total_comm = summary["total_comm_time"]
        total_comp = summary["total_compute_time"]
        if total_comm > 0:
            summary["compute_to_comm_ratio"] = total_comp / total_comm
        
        # Per-worker breakdown
        summary["per_worker"] = {}
        for m in epoch_data:
            summary["per_worker"][m.worker_id] = {
                "loss": m.local_loss,
                "gradient_norm": m.gradient_norm,
                "samples": m.num_samples,
                "compute_time": m.compute_time_s,
                "comm_time": m.comm_time_s,
                "batch_size": m.batch_size,
                "compute_comm_ratio": m.compute_time_s / m.comm_time_s if m.comm_time_s > 0 else 0.0
            }
        
        # Heterogeneity indices
        compute_times = [m.compute_time_s for m in epoch_data]
        comm_times = [m.comm_time_s for m in epoch_data]
        
        if compute_times:
            summary["compute_time_skew"] = max(compute_times) / (min(compute_times) + 1e-10)
        if comm_times:
            summary["comm_time_skew"] = max(comm_times) / (min(comm_times) + 1e-10)
        
        return summary
    
    def get_straggler_analysis(self, epoch: int) -> Dict:
        """
        Identify stragglers (slow workers) in an epoch.
        
        Returns:
            Dict with straggler info
        """
        epoch_data = [m for m in self.worker_metrics if m.epoch == epoch]
        
        if not epoch_data:
            return {}
        
        # Workers sorted by compute time
        sorted_by_compute = sorted(epoch_data, key=lambda m: m.compute_time_s, reverse=True)
        
        analysis = {
            "epoch": epoch,
            "slowest_worker": sorted_by_compute[0].worker_id if sorted_by_compute else None,
            "slowest_compute_time": sorted_by_compute[0].compute_time_s if sorted_by_compute else 0,
            "fastest_worker": sorted_by_compute[-1].worker_id if sorted_by_compute else None,
            "fastest_compute_time": sorted_by_compute[-1].compute_time_s if sorted_by_compute else 0,
            "compute_time_variance": float(np.var([m.compute_time_s for m in epoch_data])),
            "worker_ranking": [m.worker_id for m in sorted_by_compute]
        }
        
        return analysis
    
    def get_cumulative_statistics(self) -> Dict:
        """
        Get cumulative statistics across all training.
        """
        if not self.worker_metrics:
            return {}
        
        stats = {
            "total_epochs": len(set(m.epoch for m in self.worker_metrics)),
            "total_iterations": len(self.worker_metrics),
            "per_worker_stats": {}
        }
        
        # Per-worker cumulative stats
        for worker_id in set(m.worker_id for m in self.worker_metrics):
            worker_data = [m for m in self.worker_metrics if m.worker_id == worker_id]
            
            stats["per_worker_stats"][worker_id] = {
                "iterations": len(worker_data),
                "total_samples": sum(m.num_samples for m in worker_data),
                "avg_loss": np.mean([m.local_loss for m in worker_data]),
                "total_compute_time": self.compute_times[worker_id],
                "total_comm_time": self.communication_overhead[worker_id],
                "avg_batch_size": np.mean([m.batch_size for m in worker_data]),
            }
        
        return stats
    
    def log_epoch_analysis(self, epoch: int) -> str:
        """Generate comprehensive log for epoch heterogeneity analysis."""
        summary = self.get_epoch_summary(epoch)
        straggler = self.get_straggler_analysis(epoch)
        
        if not summary:
            return f"[Heterogeneity] No data for epoch {epoch}"
        
        lines = [f"[Heterogeneity Analysis] Epoch {epoch}"]
        lines.append(f"  Avg Loss: {summary['avg_loss']:.6f}")
        lines.append(f"  Compute/Comm Ratio: {summary['compute_to_comm_ratio']:.2f}x")
        
        if "compute_time_skew" in summary:
            lines.append(f"  Compute Time Skew: {summary['compute_time_skew']:.2f}x")
        if "comm_time_skew" in summary:
            lines.append(f"  Comm Time Skew: {summary['comm_time_skew']:.2f}x")
        
        lines.append(f"  Straggler: Worker {straggler.get('slowest_worker')} "
                    f"({straggler.get('slowest_compute_time', 0):.3f}s)")
        lines.append(f"  Fastest: Worker {straggler.get('fastest_worker')} "
                    f"({straggler.get('fastest_compute_time', 0):.3f}s)")
        
        # Per-worker detail
        lines.append("  Per-worker breakdown:")
        for w_id in sorted(summary.get("per_worker", {}).keys()):
            w_info = summary["per_worker"][w_id]
            lines.append(
                f"    W{w_id}: loss={w_info['loss']:.4f}, "
                f"compute={w_info['compute_time']:.3f}s, "
                f"batch={w_info['batch_size']}"
            )
        
        return "\n".join(lines)
    
    def log_cumulative_analysis(self) -> str:
        """Generate log for cumulative heterogeneity statistics."""
        stats = self.get_cumulative_statistics()
        
        if not stats:
            return "[Heterogeneity] No training data collected"
        
        lines = ["[Heterogeneity] Cumulative Statistics"]
        lines.append(f"  Total Epochs: {stats['total_epochs']}")
        lines.append(f"  Total Iterations: {stats['total_iterations']}")
        lines.append("  Per-Worker Summary:")
        
        for w_id in sorted(stats.get("per_worker_stats", {}).keys()):
            w_stats = stats["per_worker_stats"][w_id]
            lines.append(
                f"    Worker {w_id}: "
                f"{w_stats['iterations']} iters, "
                f"{w_stats['total_samples']} samples, "
                f"avg_loss={w_stats['avg_loss']:.4f}, "
                f"compute_time={w_stats['total_compute_time']:.2f}s"
            )
        
        return "\n".join(lines)

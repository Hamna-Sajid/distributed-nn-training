"""
Variable Batch Size Management for Heterogeneous Workers

Dynamically assigns batch sizes to workers based on their computational capacity.
Faster workers receive larger batch sizes to maximize throughput and minimize
communication overhead relative to computation time.
"""

import numpy as np
from typing import Dict, List


class BatchSizeManager:
    """
    Manages variable batch sizes across heterogeneous workers.
    
    Batch size is computed as: batch_size_i = base_batch_size * speed_ratio_i
    This ensures faster workers process more data per iteration,
    improving overall throughput.
    """
    
    def __init__(self, base_batch_size: int = 32, min_batch_size: int = 8):
        """
        Args:
            base_batch_size: Base batch size for the slowest worker
            min_batch_size: Minimum batch size to prevent tiny batches
        """
        self.base_batch_size = base_batch_size
        self.min_batch_size = min_batch_size
        self.worker_batch_sizes: Dict[int, int] = {}
        self.batch_history: List[Dict] = []
    
    def compute_batch_sizes(self, speed_ratios: Dict[int, float]) -> Dict[int, int]:
        """
        Compute batch sizes for each worker based on speed ratios.
        
        Args:
            speed_ratios: Dict mapping worker_id -> speed_ratio (1.0 = baseline)
                         Higher ratio = faster worker
        
        Returns:
            Dict mapping worker_id -> batch_size
        """
        batch_sizes = {}
        
        # Find the minimum speed ratio to use as reference
        min_ratio = min(speed_ratios.values()) if speed_ratios else 1.0
        
        for worker_id, ratio in speed_ratios.items():
            # Normalize: slower worker (min ratio) gets base_batch_size
            normalized_ratio = ratio / min_ratio if min_ratio > 0 else 1.0
            
            # Batch size scales with normalized speed
            batch_size = max(
                int(self.base_batch_size * normalized_ratio),
                self.min_batch_size
            )
            batch_sizes[worker_id] = batch_size
        
        self.worker_batch_sizes = batch_sizes
        return batch_sizes
    
    def get_batch_size(self, worker_id: int) -> int:
        """Get batch size for specific worker."""
        return self.worker_batch_sizes.get(worker_id, self.base_batch_size)
    
    def get_all_batch_sizes(self) -> Dict[int, int]:
        """Get all current batch sizes."""
        return self.worker_batch_sizes.copy()
    
    def log_batch_sizes(self, epoch: int) -> str:
        """Generate log string for batch size configuration."""
        sizes_str = ", ".join(
            f"W{w_id}: {size}" 
            for w_id, size in sorted(self.worker_batch_sizes.items())
        )
        log_msg = f"[Epoch {epoch}] Variable batch sizes: {sizes_str}"
        self.batch_history.append({
            "epoch": epoch,
            "batch_sizes": self.worker_batch_sizes.copy()
        })
        return log_msg
    
    def get_effective_batch_size(self) -> int:
        """
        Get effective global batch size (sum of all worker batch sizes).
        Useful for understanding convergence behavior.
        """
        return sum(self.worker_batch_sizes.values())
    
    def get_batch_distribution_ratio(self) -> Dict[int, float]:
        """
        Get the proportion of total batch allocated to each worker.
        """
        total = self.get_effective_batch_size()
        if total == 0:
            return {}
        
        return {
            worker_id: size / total
            for worker_id, size in self.worker_batch_sizes.items()
        }

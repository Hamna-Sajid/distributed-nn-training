"""
Data Skew Patterns for Heterogeneous Data Distribution

Simulates realistic data distribution scenarios where different workers
may receive non-uniform data distributions due to:
- Statistical heterogeneity (data imbalance across classes)
- Sampling bias (some workers get different class distributions)
- Quantity skew (some workers get more/fewer total samples)
"""

import numpy as np
from typing import Dict, List, Tuple
from enum import Enum


class SkewType(Enum):
    """Types of data skew to simulate."""
    UNIFORM = "uniform"              # No skew (baseline)
    CLASS_IMBALANCE = "class_imbalance"  # Different class distributions
    QUANTITY_SKEW = "quantity_skew"    # Some workers get more samples
    COMBINED = "combined"             # Both class and quantity skew


class DataSkewGenerator:
    """
    Generates heterogeneous data distributions for workers.
    
    Enables studying the impact of non-IID (independent and identically
    distributed) data on distributed training convergence and performance.
    """
    
    def __init__(self, num_workers: int, skew_type: SkewType = SkewType.UNIFORM):
        """
        Args:
            num_workers: Number of worker nodes
            skew_type: Type of data skew to apply
        """
        self.num_workers = num_workers
        self.skew_type = skew_type
        self.skew_parameters: Dict = {}
    
    def set_class_imbalance(self, class_distribution: Dict[int, List[float]]) -> None:
        """
        Set class distribution for each worker.
        
        Args:
            class_distribution: Dict mapping worker_id -> list of class probabilities
                               e.g., {0: [0.9, 0.1], 1: [0.5, 0.5]}  (2 classes)
                               Worker 0: 90% class 0, 10% class 1
                               Worker 1: 50% class 0, 50% class 1
        """
        self.skew_parameters["class_distribution"] = class_distribution
        self.skew_type = SkewType.CLASS_IMBALANCE
    
    def set_quantity_skew(self, worker_sample_counts: Dict[int, int]) -> None:
        """
        Set the number of samples each worker receives.
        
        Args:
            worker_sample_counts: Dict mapping worker_id -> num_samples
                                 e.g., {0: 8000, 1: 100}  (80x skew)
        """
        self.skew_parameters["sample_counts"] = worker_sample_counts
        self.skew_type = SkewType.QUANTITY_SKEW
    
    def set_combined_skew(self, 
                         worker_sample_counts: Dict[int, int],
                         class_distribution: Dict[int, List[float]]) -> None:
        """
        Set both quantity and class skew simultaneously.
        """
        self.skew_parameters["sample_counts"] = worker_sample_counts
        self.skew_parameters["class_distribution"] = class_distribution
        self.skew_type = SkewType.COMBINED
    
    def generate_default_class_imbalance(self, imbalance_factor: float = 0.8) -> Dict[int, List[float]]:
        """
        Generate default class imbalance pattern.
        
        Args:
            imbalance_factor: How extreme the imbalance is (0-1)
                            0.5 = balanced, 1.0 = maximum imbalance
        
        For 2-class problem, creates a Dirichlet distribution.
        """
        num_classes = 2  # Binary classification
        distributions = {}
        
        for w_id in range(self.num_workers):
            # Create skewed distribution using Dirichlet
            alpha = [imbalance_factor, 1.0 - imbalance_factor] if w_id % 2 == 0 else [1.0 - imbalance_factor, imbalance_factor]
            dist = np.random.dirichlet(alpha).tolist()
            distributions[w_id] = dist
        
        return distributions
    
    def generate_default_quantity_skew(self, total_samples: int, 
                                       skew_factor: float = 0.8) -> Dict[int, int]:
        """
        Generate default quantity skew pattern.
        
        Args:
            total_samples: Total number of samples to distribute
            skew_factor: How extreme the skew is (0.5 = balanced, 0.95 = extreme)
        
        Faster workers get more samples (using speed_ratios if available).
        """
        # For now, use a simple exponential distribution
        sample_counts = {}
        
        # Create weights favoring first worker
        weights = np.array([skew_factor ** i for i in range(self.num_workers)])
        weights = weights / weights.sum()
        
        # Distribute samples according to weights
        allocations = np.random.multinomial(total_samples, weights)
        
        for w_id, count in enumerate(allocations):
            sample_counts[w_id] = int(count)
        
        return sample_counts
    
    def get_worker_data_distribution(self, worker_id: int) -> Dict:
        """
        Get the data distribution configuration for a specific worker.
        
        Returns:
            Dict with 'num_samples' and 'class_probs' keys
        """
        info = {}
        
        # Add sample count
        if "sample_counts" in self.skew_parameters:
            info["num_samples"] = self.skew_parameters["sample_counts"].get(worker_id, 0)
        
        # Add class probabilities
        if "class_distribution" in self.skew_parameters:
            info["class_probs"] = self.skew_parameters["class_distribution"].get(worker_id, None)
        
        return info
    
    def get_skew_metrics(self) -> Dict:
        """
        Compute metrics describing the degree of data skew.
        
        Returns:
            Dict with various skew metrics
        """
        metrics = {"skew_type": self.skew_type.value}
        
        # Quantity skew metrics
        if "sample_counts" in self.skew_parameters:
            counts = list(self.skew_parameters["sample_counts"].values())
            max_count = max(counts)
            min_count = min(counts)
            metrics["quantity_skew_ratio"] = max_count / min_count if min_count > 0 else float('inf')
            metrics["total_samples"] = sum(counts)
            metrics["samples_per_worker"] = {w_id: c for w_id, c in enumerate(counts)}
        
        # Class distribution skew (entropy-based)
        if "class_distribution" in self.skew_parameters:
            distributions = self.skew_parameters["class_distribution"]
            entropies = []
            
            for worker_id, probs in distributions.items():
                probs = np.array(probs)
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                entropies.append(entropy)
            
            max_entropy = np.log2(len(next(iter(distributions.values()))))
            avg_entropy = np.mean(entropies)
            normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0
            
            metrics["class_imbalance_entropy"] = avg_entropy
            metrics["class_balance_ratio"] = normalized_entropy  # 1.0 = fully balanced
            metrics["class_distributions"] = distributions
        
        return metrics
    
    def log_configuration(self) -> str:
        """Generate log string describing data skew configuration."""
        lines = [f"[Data Skew] Type: {self.skew_type.value}"]
        
        if self.skew_type in [SkewType.QUANTITY_SKEW, SkewType.COMBINED]:
            if "sample_counts" in self.skew_parameters:
                counts = self.skew_parameters["sample_counts"]
                lines.append("  Sample distribution:")
                for w_id, count in sorted(counts.items()):
                    lines.append(f"    Worker {w_id}: {count} samples")
                
                if counts:
                    ratio = max(counts.values()) / min(counts.values()) if min(counts.values()) > 0 else 0
                    lines.append(f"  Quantity skew ratio: {ratio:.1f}x")
        
        if self.skew_type in [SkewType.CLASS_IMBALANCE, SkewType.COMBINED]:
            if "class_distribution" in self.skew_parameters:
                dists = self.skew_parameters["class_distribution"]
                lines.append("  Class distributions:")
                for w_id, probs in sorted(dists.items()):
                    probs_str = ", ".join(f"{p*100:.1f}%" for p in probs)
                    lines.append(f"    Worker {w_id}: [{probs_str}]")
        
        return "\n".join(lines)

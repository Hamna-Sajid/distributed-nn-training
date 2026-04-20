#!/usr/bin/env python3
"""
Heterogeneity Integration Example

This script demonstrates how to integrate all heterogeneity features
into the distributed training pipeline.

Usage:
    # Enable in config.yaml, then:
    python heterogeneity_example.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from config_loader import ConfigLoader
from heterogeneity.batch_size import BatchSizeManager
from heterogeneity.network_simulator import NetworkSimulator, NetworkCondition
from heterogeneity.data_skew import DataSkewGenerator, SkewType
from heterogeneity.metrics import HeterogeneityAnalyzer


def setup_heterogeneity_components(num_workers: int, speed_ratios: dict):
    """
    Initialize all heterogeneity components based on configuration.
    
    Args:
        num_workers: Number of worker nodes
        speed_ratios: Dict mapping worker_id -> speed_ratio
    
    Returns:
        Tuple of (batch_mgr, net_sim, data_gen, analyzer)
    """
    config = ConfigLoader()
    
    # 1. Setup Variable Batch Sizes
    print("=[Heterogeneity Setup]==========================================")
    batch_mgr = None
    if config.get("heterogeneity.batch_size.enabled", False):
        base_batch = config.get("heterogeneity.batch_size.base_batch_size", 32)
        min_batch = config.get("heterogeneity.batch_size.min_batch_size", 8)
        
        batch_mgr = BatchSizeManager(base_batch_size=base_batch, min_batch_size=min_batch)
        batch_sizes = batch_mgr.compute_batch_sizes(speed_ratios)
        print(batch_mgr.log_batch_sizes(epoch=0))
    
    # 2. Setup Network Simulation
    net_sim = None
    if config.get("heterogeneity.network.enabled", False):
        loss_rate = config.get("heterogeneity.network.packet_loss_rate", 0.0)
        net_sim = NetworkSimulator(enabled=True, loss_rate=loss_rate)
        
        # Set per-worker conditions if configured
        worker_conditions = config.get("heterogeneity.network.worker_conditions", {})
        condition_map = {
            "fast_lan": NetworkCondition.FAST_LAN,
            "standard_lan": NetworkCondition.STANDARD_LAN,
            "slow_wan": NetworkCondition.SLOW_WAN,
            "very_slow": NetworkCondition.VERY_SLOW,
        }
        
        for w_id, cond_name in worker_conditions.items():
            if cond_name in condition_map:
                net_sim.set_worker_condition(w_id, condition_map[cond_name])
            else:
                net_sim.set_worker_condition(w_id, NetworkCondition.STANDARD_LAN)
        
        print(net_sim.log_configuration())
    
    # 3. Setup Data Skew
    data_gen = None
    if config.get("heterogeneity.data_skew.enabled", False):
        skew_type_str = config.get("heterogeneity.data_skew.type", "uniform").upper()
        skew_type = SkewType[skew_type_str] if skew_type_str in SkewType.__members__ else SkewType.UNIFORM
        
        data_gen = DataSkewGenerator(num_workers=num_workers, skew_type=skew_type)
        
        # Load custom distributions if provided
        if skew_type in [SkewType.QUANTITY_SKEW, SkewType.COMBINED]:
            qty_dist = config.get("heterogeneity.data_skew.quantity_distribution", None)
            if qty_dist:
                data_gen.set_quantity_skew(qty_dist)
        
        if skew_type in [SkewType.CLASS_IMBALANCE, SkewType.COMBINED]:
            class_dist = config.get("heterogeneity.data_skew.class_distribution", None)
            if class_dist:
                data_gen.set_class_imbalance(class_dist)
        
        print(data_gen.log_configuration())
    
    # 4. Setup Metrics Analyzer
    analyzer = HeterogeneityAnalyzer()
    if config.get("heterogeneity.analysis.enabled", False):
        print("[Heterogeneity Analysis] Enabled - tracking metrics")
    
    print("============================================================\n")
    
    return batch_mgr, net_sim, data_gen, analyzer


def example_training_loop():
    """Example integration in master training loop."""
    
    print("Example: Master Training Loop with Heterogeneity\n")
    
    # Simulate worker speed ratios from benchmarking
    speed_ratios = {0: 2.0, 1: 1.0}
    num_workers = 2
    
    # Setup heterogeneity components
    batch_mgr, net_sim, data_gen, analyzer = setup_heterogeneity_components(
        num_workers, speed_ratios
    )
    
    # Simulate 2 epochs of training
    for epoch in range(2):
        print(f"\n--- Epoch {epoch + 1} ---")
        
        # Get batch sizes for this epoch
        if batch_mgr:
            batch_sizes = batch_mgr.get_all_batch_sizes()
            analyzer.record_batch_sizes(epoch, batch_sizes)
            print(f"Batch sizes: {batch_sizes}")
        else:
            batch_sizes = {0: 32, 1: 32}
        
        # Collect metrics from workers
        epoch_start = __import__('time').time()
        
        for worker_id in range(num_workers):
            # Simulate gradient computation and communication
            compute_time = 0.01 * (2 - worker_id) * 2  # W0 faster than W1
            comm_time = 0.005
            
            # Simulate network delay if enabled
            if net_sim:
                net_sim.apply_send_delay(worker_id)
                comm_time *= 2  # Longer due to network
            
            # Simulate loss and gradient norm
            local_loss = 0.5 - epoch * 0.05 + np.random.rand() * 0.1
            gradient_norm = 1.2 - epoch * 0.1
            
            # Record metrics
            analyzer.record_worker_iteration(
                worker_id=worker_id,
                epoch=epoch,
                num_samples=batch_sizes.get(worker_id, 32),
                local_loss=local_loss,
                gradient_norm=gradient_norm,
                compute_time=compute_time,
                comm_time=comm_time,
                batch_size=batch_sizes.get(worker_id, 32)
            )
            
            print(f"  W{worker_id}: loss={local_loss:.4f}, "
                  f"compute={compute_time:.3f}s, comm={comm_time:.3f}s")
        
        # Simulate aggregation
        aggregation_time = 0.02
        analyzer.record_aggregation_time(epoch, aggregation_time)
        
        # Print epoch analysis
        print(analyzer.log_epoch_analysis(epoch))
        
        # Print network statistics if enabled
        if net_sim and net_sim.enabled:
            print(net_sim.log_statistics())
    
    # Print cumulative statistics
    print("\n" + analyzer.log_cumulative_analysis())


def example_data_skew_impact():
    """Example showing data skew impact."""
    
    print("\n\nExample: Data Skew Impact Analysis\n")
    
    # Test different skew scenarios
    scenarios = [
        ("Uniform", SkewType.UNIFORM),
        ("Quantity Skew (8:2)", SkewType.QUANTITY_SKEW),
        ("Class Imbalance", SkewType.CLASS_IMBALANCE),
        ("Combined Skew", SkewType.COMBINED),
    ]
    
    for scenario_name, skew_type in scenarios:
        print(f"\n--- {scenario_name} ---")
        
        gen = DataSkewGenerator(num_workers=2, skew_type=skew_type)
        
        if skew_type == SkewType.QUANTITY_SKEW:
            counts = {0: 8000, 1: 2000}
            gen.set_quantity_skew(counts)
        elif skew_type == SkewType.CLASS_IMBALANCE:
            dists = gen.generate_default_class_imbalance(imbalance_factor=0.8)
            gen.set_class_imbalance(dists)
        elif skew_type == SkewType.COMBINED:
            counts = {0: 8000, 1: 2000}
            dists = {0: [0.8, 0.2], 1: [0.5, 0.5]}
            gen.set_combined_skew(counts, dists)
        
        metrics = gen.get_skew_metrics()
        
        # Print metrics
        if "quantity_skew_ratio" in metrics:
            print(f"Quantity Skew Ratio: {metrics['quantity_skew_ratio']:.2f}x")
        if "class_balance_ratio" in metrics:
            print(f"Class Balance Ratio: {metrics['class_balance_ratio']:.3f} (1.0 = balanced)")


def example_network_conditions():
    """Example showing different network conditions."""
    
    print("\n\nExample: Network Conditions\n")
    
    # Setup different network scenarios
    scenarios = [
        ("No Network Effect", [], 0.0),
        ("Local LAN", [(0, NetworkCondition.FAST_LAN), (1, NetworkCondition.FAST_LAN)], 0.0),
        ("Mixed LAN/WAN", [(0, NetworkCondition.FAST_LAN), (1, NetworkCondition.SLOW_WAN)], 0.01),
        ("High Loss WAN", [(0, NetworkCondition.VERY_SLOW), (1, NetworkCondition.VERY_SLOW)], 0.05),
    ]
    
    for scenario_name, conditions, loss_rate in scenarios:
        print(f"\n--- {scenario_name} ---")
        
        sim = NetworkSimulator(enabled=True, loss_rate=loss_rate)
        
        for w_id, condition in conditions:
            sim.set_worker_condition(w_id, condition)
        
        # Simulate 1000 message sends
        for _ in range(1000):
            for worker_id in range(2):
                sim.simulate_send_delay(worker_id)
        
        stats = sim.get_statistics()
        print(f"Avg latency: {stats['avg_delay_ms']:.2f}ms")
        print(f"Packet loss rate: {stats['loss_rate_actual']*100:.2f}%")
        print(f"Total simulated delay: {stats['total_delay_s']:.2f}s")


if __name__ == "__main__":
    # Run examples
    example_training_loop()
    example_data_skew_impact()
    example_network_conditions()
    
    print("\n" + "="*60)
    print("Heterogeneity Examples Complete")
    print("="*60)

#!/usr/bin/env python3
"""
Integration tests for heterogeneity features.

Tests variable batch sizes, network simulation, and data skew independently
and in combination.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from heterogeneity.batch_size import BatchSizeManager
from heterogeneity.network_simulator import NetworkSimulator, NetworkCondition
from heterogeneity.data_skew import DataSkewGenerator, SkewType
from heterogeneity.metrics import HeterogeneityAnalyzer


def test_batch_size_manager():
    """Test variable batch size computation."""
    print("[Test 1] Batch Size Manager")
    
    manager = BatchSizeManager(base_batch_size=32, min_batch_size=8)
    
    # Speed ratios: W0 is 2x faster than W1
    speed_ratios = {0: 2.0, 1: 1.0}
    batch_sizes = manager.compute_batch_sizes(speed_ratios)
    
    assert batch_sizes[0] == 64, f"Expected 64 for W0, got {batch_sizes[0]}"
    assert batch_sizes[1] == 32, f"Expected 32 for W1, got {batch_sizes[1]}"
    
    log_msg = manager.log_batch_sizes(epoch=1)
    print(f"  {log_msg}")
    
    effective_batch = manager.get_effective_batch_size()
    print(f"  Effective global batch size: {effective_batch}")
    
    distribution = manager.get_batch_distribution_ratio()
    print(f"  Batch distribution: {distribution}")
    
    print("  ✓ PASSED\n")


def test_network_simulator():
    """Test network simulation with various conditions."""
    print("[Test 2] Network Simulator")
    
    sim = NetworkSimulator(enabled=True, loss_rate=0.05)
    
    # Set different conditions for different workers
    sim.set_worker_condition(0, NetworkCondition.FAST_LAN)
    sim.set_worker_condition(1, NetworkCondition.SLOW_WAN)
    
    print(f"  {sim.log_configuration()}\n")
    
    # Simulate 100 message sends
    total_delay = 0.0
    losses = 0
    for _ in range(100):
        for worker_id in [0, 1]:
            delay, lost = sim.simulate_send_delay(worker_id)
            if not lost:
                total_delay += delay
            else:
                losses += 1
    
    print(f"  {sim.log_statistics()}")
    print(f"  Total simulated delay for 100 sends: {total_delay:.4f}s")
    print("  ✓ PASSED\n")


def test_data_skew_generator():
    """Test data skew pattern generation."""
    print("[Test 3] Data Skew Generator")
    
    generator = DataSkewGenerator(num_workers=2, skew_type=SkewType.QUANTITY_SKEW)
    
    # Test quantity skew
    sample_counts = generator.generate_default_quantity_skew(
        total_samples=10000,
        skew_factor=0.8
    )
    generator.set_quantity_skew(sample_counts)
    
    print("  Quantity Skew Configuration:")
    print(f"  {generator.log_configuration()}")
    
    metrics = generator.get_skew_metrics()
    print(f"  Quantity skew ratio: {metrics['quantity_skew_ratio']:.2f}x")
    print(f"  Total samples: {metrics['total_samples']}")
    
    # Test class imbalance
    print("\n  Class Imbalance Configuration:")
    class_dist = generator.generate_default_class_imbalance(imbalance_factor=0.9)
    generator.set_class_imbalance(class_dist)
    print(f"  {generator.log_configuration()}")
    
    print("  ✓ PASSED\n")


def test_heterogeneity_analyzer():
    """Test heterogeneity metrics collection and analysis."""
    print("[Test 4] Heterogeneity Analyzer")
    
    analyzer = HeterogeneityAnalyzer()
    
    # Simulate 2 workers, 2 epochs
    for epoch in range(2):
        batch_sizes = {0: 64, 1: 32}
        analyzer.record_batch_sizes(epoch, batch_sizes)
        
        # Worker 0 (faster)
        analyzer.record_worker_iteration(
            worker_id=0,
            epoch=epoch,
            num_samples=64,
            local_loss=0.5,
            gradient_norm=1.2,
            compute_time=0.01,
            comm_time=0.005,
            batch_size=64
        )
        
        # Worker 1 (slower)
        analyzer.record_worker_iteration(
            worker_id=1,
            epoch=epoch,
            num_samples=32,
            local_loss=0.6,
            gradient_norm=1.5,
            compute_time=0.05,
            comm_time=0.01,
            batch_size=32
        )
        
        analyzer.record_aggregation_time(epoch, 0.02)
    
    # Test epoch summary
    print("  Epoch 0 Summary:")
    summary = analyzer.get_epoch_summary(0)
    print(f"    Avg loss: {summary['avg_loss']:.4f}")
    print(f"    Compute/Comm ratio: {summary['compute_to_comm_ratio']:.2f}x")
    print(f"    Compute time skew: {summary['compute_time_skew']:.2f}x")
    
    # Test straggler analysis
    print("\n  Straggler Analysis:")
    straggler = analyzer.get_straggler_analysis(0)
    print(f"    Slowest: Worker {straggler['slowest_worker']} ({straggler['slowest_compute_time']:.3f}s)")
    print(f"    Fastest: Worker {straggler['fastest_worker']} ({straggler['fastest_compute_time']:.3f}s)")
    
    # Test logging
    print("\n  " + analyzer.log_epoch_analysis(0).replace("\n", "\n  "))
    
    print("  ✓ PASSED\n")


def test_combined_heterogeneity():
    """Test all features working together."""
    print("[Test 5] Combined Heterogeneity Features")
    
    # Setup batch size management
    batch_mgr = BatchSizeManager(base_batch_size=32)
    speed_ratios = {0: 2.0, 1: 1.0}  # W0 is 2x faster
    batch_sizes = batch_mgr.compute_batch_sizes(speed_ratios)
    
    # Setup network simulation
    net_sim = NetworkSimulator(enabled=True, loss_rate=0.02)
    net_sim.set_worker_condition(0, NetworkCondition.FAST_LAN)
    net_sim.set_worker_condition(1, NetworkCondition.SLOW_WAN)
    
    # Setup data skew
    data_gen = DataSkewGenerator(num_workers=2, skew_type=SkewType.COMBINED)
    sample_counts = {0: 8000, 1: 2000}
    class_dists = {0: [0.8, 0.2], 1: [0.5, 0.5]}
    data_gen.set_combined_skew(sample_counts, class_dists)
    
    # Setup metrics
    analyzer = HeterogeneityAnalyzer()
    analyzer.record_batch_sizes(0, batch_sizes)
    analyzer.record_data_distribution(0, data_gen.get_skew_metrics())
    
    print("  Combined Configuration:")
    print(f"    Batch sizes: {batch_sizes}")
    print(f"    {net_sim.log_configuration()}")
    print(f"    {data_gen.log_configuration()}")
    
    print("\n  ✓ PASSED\n")


if __name__ == "__main__":
    print("=" * 70)
    print("HETEROGENEITY FEATURES TEST SUITE")
    print("=" * 70 + "\n")
    
    try:
        test_batch_size_manager()
        test_network_simulator()
        test_data_skew_generator()
        test_heterogeneity_analyzer()
        test_combined_heterogeneity()
        
        print("=" * 70)
        print("ALL HETEROGENEITY TESTS PASSED ✓")
        print("=" * 70)
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

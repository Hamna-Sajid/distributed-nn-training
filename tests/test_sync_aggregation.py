# tests/test_sync_aggregation.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from coordination.adaptive_aggregator import AdaptiveAggregator

def make_fake_gradients(seed, layers=('layer1', 'layer2')):
    """Helper to create fake gradient dicts for testing."""
    np.random.seed(seed)
    return {
        layer: {
            'weights': np.random.randn(10, 10),
            'biases': np.random.randn(10)
        }
        for layer in layers
    }

def test_basic_aggregation():
    agg = AdaptiveAggregator()
    grads = {0: make_fake_gradients(0), 1: make_fake_gradients(1)}
    batch_sizes = {0: 9000, 1: 1000}  # realistic split
    losses = {0: 0.65, 1: 0.70}
    arrival_times = {0: 0.1, 1: 2.0}

    result, weights = agg.aggregate(grads, batch_sizes, losses, arrival_times)

    # Worker 0 should dominate (large batch + early arrival)
    assert weights[0] > weights[1], "Worker 0 should have higher weight"
    # Weights must sum to 1
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    print(f"PASS: weights = {weights}")

def test_norm_clipping():
    agg = AdaptiveAggregator(norm_clip_threshold=1.5)  # Lower threshold to trigger clipping
    # Make worker 1's gradients much larger
    g0 = make_fake_gradients(0)
    g1 = {k: {'weights': v['weights'] * 100, 'biases': v['biases'] * 100}
          for k, v in make_fake_gradients(1).items()}
    
    _, weights = agg.aggregate(
        {0: g0, 1: g1},
        batch_sizes={0: 500, 1: 500},  # equal batches
        losses={0: 0.65, 1: 0.65},
        arrival_times={0: 0.1, 1: 0.1}
    )
    # Worker 1's inflated norms should get clipped down
    print(f"PASS norm clip: w0={weights[0]:.3f}, w1={weights[1]:.3f}")
    assert weights[0] > weights[1], "Norm clip should reduce worker 1's influence"

if __name__ == '__main__':
    test_basic_aggregation()
    test_norm_clipping()
    print("All tests passed.")
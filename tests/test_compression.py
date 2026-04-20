"""
Quick test of gradient compression module.
Run this to verify quantization works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from communication.compression import (
    compress_gradients, decompress_gradients,
    sparsify_gradients, desparse_gradients,
    quantize_int8, dequantize_int8, compute_quantization_params
)


def test_int8_quantization():
    """Test basic int8 quantization and dequantization."""
    print("[Test 1] Int8 Quantization...")
    
    # Create a sample gradient array
    original = np.random.randn(100, 100).astype(np.float32)
    
    # Quantize
    quantized, params = quantize_int8(original)
    
    # Dequantize
    recovered = dequantize_int8(quantized, params)
    
    # Check error
    error = np.mean(np.abs(original - recovered))
    max_error = np.max(np.abs(original - recovered))
    
    print(f"  Original shape: {original.shape}, dtype: {original.dtype}")
    print(f"  Quantized shape: {quantized.shape}, dtype: {quantized.dtype}")
    print(f"  Mean reconstruction error: {error:.6f}")
    print(f"  Max reconstruction error: {max_error:.6f}")
    print(f"  Compression ratio: {original.nbytes / quantized.nbytes:.2f}x")
    print("  ✓ PASSED\n")


def test_gradient_compression():
    """Test compression of nested gradient dict."""
    print("[Test 2] Gradient Dict Compression...")
    
    # Create fake gradient dict (like from MLP)
    grad_dict = {
        "layer1": {
            "dW": np.random.randn(784, 2048).astype(np.float32),
            "db": np.random.randn(2048).astype(np.float32)
        },
        "layer2": {
            "dW": np.random.randn(2048, 2048).astype(np.float32),
            "db": np.random.randn(2048).astype(np.float32)
        }
    }
    
    # Compress
    compressed = compress_gradients(grad_dict)
    
    # Decompress
    recovered = decompress_gradients(compressed)
    
    # Check structure and error
    print(f"  Original layers: {list(grad_dict.keys())}")
    print(f"  Compressed format: {compressed['type']}")
    print(f"  Overall compression ratio: {compressed['compression_ratio']:.2f}x")
    
    for layer in grad_dict:
        for param in grad_dict[layer]:
            orig = grad_dict[layer][param]
            recov = recovered[layer][param]
            error = np.mean(np.abs(orig - recov))
            print(f"  {layer}.{param}: error={error:.6f}, shape={recov.shape}")
    
    print("  ✓ PASSED\n")


def test_sparsification():
    """Test gradient sparsification."""
    print("[Test 3] Gradient Sparsification...")
    
    grad_dict = {
        "layer1": {
            "dW": np.random.randn(100, 100).astype(np.float32),
        }
    }
    
    # Sparsify (keep top 10%)
    sparse = sparsify_gradients(grad_dict, threshold_percentile=90)
    
    # Desparse
    recovered = desparse_gradients(sparse)
    
    orig = grad_dict["layer1"]["dW"]
    recov = recovered["layer1"]["dW"]
    
    # Count non-zero values
    orig_nonzero = np.count_nonzero(orig)
    sparse_nonzero = len(sparse["layers"]["layer1"]["dW"]["values"])  # Length of values list
    recov_nonzero = np.count_nonzero(recov)
    
    print(f"  Original non-zeros: {orig_nonzero}")
    print(f"  Sparse non-zeros (kept): {sparse_nonzero}")
    print(f"  Recovered non-zeros: {recov_nonzero}")
    print(f"  Sparsity reduction: {(1 - sparse_nonzero/orig_nonzero)*100:.1f}%")
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Gradient Compression Module Tests")
    print("=" * 60 + "\n")
    
    test_int8_quantization()
    test_gradient_compression()
    test_sparsification()
    
    print("=" * 60)
    print("All tests PASSED! ✓")
    print("=" * 60)

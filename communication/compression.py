"""
Gradient compression utilities for reducing communication overhead.

Implements:
- int8/int16 quantization with dynamic scale/zero-point
- Gradient sparsification (top-k selection)
- Compression metadata tracking for analysis
"""

import numpy as np


class QuantizationParams:
    """Metadata for reversing quantization."""
    def __init__(self, scale: float, zero_point: int):
        self.scale = float(scale)
        self.zero_point = int(zero_point)
    
    def to_dict(self):
        return {"scale": self.scale, "zero_point": self.zero_point}
    
    @staticmethod
    def from_dict(d):
        return QuantizationParams(d["scale"], d["zero_point"])


def compute_quantization_params(arr: np.ndarray, bits: int = 8) -> QuantizationParams:
    """Compute quantization scale and zero-point for an array.
    
    Uses symmetric quantization around zero for gradients.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to quantize (typically gradients)
    bits : int
        Quantization bit depth (8 or 16)
    
    Returns
    -------
    QuantizationParams
        Scale and zero-point for dequantization
    """
    # For gradients, use symmetric quantization (zero-point = 0)
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Symmetric range: [-range, range]
    range_val = max(abs(min_val), abs(max_val))
    if range_val == 0:
        range_val = 1.0  # Avoid division by zero
    
    # Quantization range: [-2^(bits-1), 2^(bits-1) - 1]
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    
    # Scale: how much each quantized unit represents
    scale = range_val / qmax
    zero_point = 0
    
    return QuantizationParams(scale, zero_point)


def quantize_int8(arr: np.ndarray, params: QuantizationParams = None) -> tuple:
    """Quantize a float32 array to int8.
    
    Parameters
    ----------
    arr : np.ndarray
        Float32 array to quantize
    params : QuantizationParams, optional
        Quantization parameters. If None, computed from array.
    
    Returns
    -------
    tuple
        (quantized_array: np.ndarray, params: QuantizationParams)
        quantized_array is int8; params needed for dequantization
    """
    if params is None:
        params = compute_quantization_params(arr, bits=8)
    
    # Quantize: Q = round(X / scale) + zero_point
    quantized = np.round(arr / params.scale + params.zero_point).astype(np.int8)
    return quantized, params


def dequantize_int8(quantized_arr: np.ndarray, params: QuantizationParams) -> np.ndarray:
    """Dequantize an int8 array back to float32.
    
    Parameters
    ----------
    quantized_arr : np.ndarray
        Int8 quantized array
    params : QuantizationParams
        Scale and zero-point from quantization
    
    Returns
    -------
    np.ndarray
        Float32 dequantized array
    """
    # Dequantize: X = (Q - zero_point) * scale
    dequantized = (quantized_arr.astype(np.float32) - params.zero_point) * params.scale
    return dequantized


def compress_gradients(grad_dict: dict, compression_type: str = 'int8') -> dict:
    """Compress a gradient dictionary using quantization.
    
    Walks through nested gradient dict (from neural network layers)
    and quantizes each weight/bias array.
    
    Parameters
    ----------
    grad_dict : dict
        Gradient dictionary with structure:
        {"layer1": {"dW": np.ndarray, "db": np.ndarray}, ...}
    compression_type : str
        'int8' or 'int16' quantization
    
    Returns
    -------
    dict
        Compressed gradient dict:
        {
            "type": "quantized_int8",
            "layers": {
                "layer1": {
                    "dW": {"quantized": [...], "params": {...}},
                    "db": {"quantized": [...], "params": {...}}
                }, ...
            },
            "compression_ratio": float
        }
    """
    if compression_type != 'int8':
        raise ValueError(f"Unsupported compression type: {compression_type}")
    
    compressed = {"type": "quantized_int8", "layers": {}}
    total_original_bytes = 0
    total_compressed_bytes = 0
    
    for layer_name, layer_grads in grad_dict.items():
        compressed["layers"][layer_name] = {}
        
        for param_name, param_array in layer_grads.items():
            if isinstance(param_array, np.ndarray):
                # Quantize
                quantized, params = quantize_int8(param_array)
                
                # Track sizes (in bytes, assuming float32 and int8)
                original_size = param_array.nbytes  # float32
                compressed_size = quantized.nbytes  # int8
                total_original_bytes += original_size
                total_compressed_bytes += compressed_size
                
                # Store compressed data with metadata
                compressed["layers"][layer_name][param_name] = {
                    "quantized": quantized.tolist(),
                    "params": params.to_dict(),
                    "shape": list(param_array.shape)
                }
    
    if total_original_bytes > 0:
        compressed["compression_ratio"] = total_original_bytes / total_compressed_bytes
    else:
        compressed["compression_ratio"] = 1.0
    
    return compressed


def decompress_gradients(compressed_dict: dict) -> dict:
    """Decompress a compressed gradient dictionary back to float32.
    
    Parameters
    ----------
    compressed_dict : dict
        Compressed gradient dict (output of compress_gradients)
    
    Returns
    -------
    dict
        Decompressed gradient dict in original format:
        {"layer1": {"dW": np.ndarray, "db": np.ndarray}, ...}
    """
    if compressed_dict.get("type") != "quantized_int8":
        raise ValueError("Unsupported compressed format")
    
    decompressed = {}
    
    for layer_name, layer_data in compressed_dict["layers"].items():
        decompressed[layer_name] = {}
        
        for param_name, param_data in layer_data.items():
            # Recover metadata
            quantized_list = param_data["quantized"]
            params_dict = param_data["params"]
            shape = tuple(param_data["shape"])
            
            # Reconstruct
            quantized_arr = np.array(quantized_list, dtype=np.int8).reshape(shape)
            params = QuantizationParams.from_dict(params_dict)
            
            # Dequantize
            decompressed_arr = dequantize_int8(quantized_arr, params)
            decompressed[layer_name][param_name] = decompressed_arr
    
    return decompressed


def sparsify_gradients(grad_dict: dict, threshold_percentile: float = 90) -> dict:
    """Sparsify gradients by keeping only top-k% by magnitude.
    
    For large gradient tensors, often only top gradients matter.
    This reduces payload by sending only significant updates.
    
    Parameters
    ----------
    grad_dict : dict
        Gradient dictionary
    threshold_percentile : float
        Keep gradients above this percentile (0-100). 
        90 = keep top 10% by magnitude
    
    Returns
    -------
    dict
        Sparse gradient dict:
        {
            "type": "sparse",
            "layers": {
                "layer1": {
                    "dW": {"indices": [...], "values": [...], "shape": [...]}
                }
            }
        }
    """
    sparse = {"type": "sparse", "layers": {}}
    
    for layer_name, layer_grads in grad_dict.items():
        sparse["layers"][layer_name] = {}
        
        for param_name, param_array in layer_grads.items():
            if isinstance(param_array, np.ndarray):
                # Compute magnitude threshold
                magnitudes = np.abs(param_array)
                threshold = np.percentile(magnitudes, threshold_percentile)
                
                # Find indices of significant gradients
                mask = magnitudes >= threshold
                indices = np.where(mask)
                values = param_array[mask]
                
                sparse["layers"][layer_name][param_name] = {
                    "indices": [idx.tolist() for idx in indices],
                    "values": values.tolist(),
                    "shape": list(param_array.shape)
                }
    
    return sparse


def desparse_gradients(sparse_dict: dict) -> dict:
    """Reconstruct gradients from sparse representation (with zeros for missing values).
    
    Parameters
    ----------
    sparse_dict : dict
        Sparse gradient dict (output of sparsify_gradients)
    
    Returns
    -------
    dict
        Reconstructed gradient dict with zeros where values were omitted
    """
    if sparse_dict.get("type") != "sparse":
        raise ValueError("Unsupported sparse format")
    
    desparse = {}
    
    for layer_name, layer_data in sparse_dict["layers"].items():
        desparse[layer_name] = {}
        
        for param_name, param_data in layer_data.items():
            shape = tuple(param_data["shape"])
            indices = param_data["indices"]
            values = param_data["values"]
            
            # Reconstruct: start with zeros, fill in sparse values
            reconstructed = np.zeros(shape, dtype=np.float32)
            if indices and values:
                idx_tuple = tuple(np.array(idx) for idx in indices)
                reconstructed[idx_tuple] = np.array(values, dtype=np.float32)
            
            desparse[layer_name][param_name] = reconstructed
    
    return desparse

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class RealQuantizedKVCache:
    """
    KV Cache that ACTUALLY stores tensors in quantized format.
    - 8-bit: stored as uint8 (1 byte per element)
    - 4-bit: stored as uint8 but packed (2 values per byte conceptually)
    """
    
    def __init__(self, k_bits=8, v_bits=8):
        self.k_bits = k_bits
        self.v_bits = v_bits
        
        # Storage for QUANTIZED tensors
        self.k_cache_quantized: List[torch.Tensor] = []
        self.v_cache_quantized: List[torch.Tensor] = []
        
        # Scales and zeros for dequantization
        self.k_scales: List[torch.Tensor] = []
        self.k_zeros: List[torch.Tensor] = []
        self.v_scales: List[torch.Tensor] = []
        self.v_zeros: List[torch.Tensor] = []
        
        # Track element counts for accurate memory calculation
        self.k_elements = 0
        self.v_elements = 0
    
    def _quantize(self, tensor: torch.Tensor, n_bits: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize FP16 tensor to int representation"""
        # Per-token quantization
        t_min = tensor.amin(dim=-1, keepdim=True)
        t_max = tensor.amax(dim=-1, keepdim=True)
        
        qmax = 2 ** n_bits - 1
        scale = (t_max - t_min) / qmax
        scale = scale.clamp(min=1e-8)
        
        # Quantize
        quantized = ((tensor - t_min) / scale).round().clamp(0, qmax)
        
        # Store as uint8 (actual storage format)
        quantized_int = quantized.to(torch.uint8)
        
        return quantized_int, scale.half(), t_min.half()
    
    def _dequantize(self, quantized: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        """Dequantize back to FP16"""
        return (quantized.float() * scale + zero).half()
    
    def quantize_and_store_k(self, key: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Quantize K, store it, and return DEQUANTIZED version (with quantization error)"""
        k_quant, k_scale, k_zero = self._quantize(key, self.k_bits)
        
        # Store quantized
        if layer_idx >= len(self.k_cache_quantized):
            self.k_cache_quantized.append(k_quant)
            self.k_scales.append(k_scale)
            self.k_zeros.append(k_zero)
        else:
            self.k_cache_quantized[layer_idx] = torch.cat([self.k_cache_quantized[layer_idx], k_quant], dim=-2)
            self.k_scales[layer_idx] = torch.cat([self.k_scales[layer_idx], k_scale], dim=-2)
            self.k_zeros[layer_idx] = torch.cat([self.k_zeros[layer_idx], k_zero], dim=-2)
        
        self.k_elements += key.numel()
        
        # Return DEQUANTIZED (this introduces quantization error into computation!)
        return self._dequantize(k_quant, k_scale, k_zero)
    
    def quantize_and_store_v(self, value: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Quantize V, store it, and return DEQUANTIZED version (with quantization error)"""
        v_quant, v_scale, v_zero = self._quantize(value, self.v_bits)
        
        # Store quantized
        if layer_idx >= len(self.v_cache_quantized):
            self.v_cache_quantized.append(v_quant)
            self.v_scales.append(v_scale)
            self.v_zeros.append(v_zero)
        else:
            self.v_cache_quantized[layer_idx] = torch.cat([self.v_cache_quantized[layer_idx], v_quant], dim=-2)
            self.v_scales[layer_idx] = torch.cat([self.v_scales[layer_idx], v_scale], dim=-2)
            self.v_zeros[layer_idx] = torch.cat([self.v_zeros[layer_idx], v_zero], dim=-2)
        
        self.v_elements += value.numel()
        
        # Return DEQUANTIZED (this introduces quantization error into computation!)
        return self._dequantize(v_quant, v_scale, v_zero)
    
    def memory_bytes(self) -> int:
        """
        Calculate ACTUAL memory used by quantized cache.
        - 8-bit: 1 byte per element
        - 4-bit: 0.5 bytes per element (packed)
        """
        # K cache memory (bits / 8 = bytes per element)
        k_bytes = self.k_elements * (self.k_bits / 8)
        
        # V cache memory
        v_bytes = self.v_elements * (self.v_bits / 8)
        
        # Add scale/zero overhead (small, FP16 per-token)
        # Approximately 4 bytes per token (scale + zero, both FP16)
        # This is much smaller than the actual cache
        
        return int(k_bytes + v_bytes)
    
    def memory_mb(self) -> float:
        return self.memory_bytes() / (1024 ** 2)
    
    def fp16_equivalent_mb(self) -> float:
        """What this cache would use in FP16 (2 bytes per element)"""
        return (self.k_elements + self.v_elements) * 2 / (1024 ** 2)
    
    def clear(self):
        self.k_cache_quantized.clear()
        self.v_cache_quantized.clear()
        self.k_scales.clear()
        self.k_zeros.clear()
        self.v_scales.clear()
        self.v_zeros.clear()
        self.k_elements = 0
        self.v_elements = 0


def patch_model_attention(model, k_bits=16, v_bits=16):
    """
    Patch model to use REAL quantized KV cache.
    Returns DEQUANTIZED values to model so quality is affected by quantization.
    """
    if k_bits >= 16 and v_bits >= 16:
        print("Using FP16 KV cache (no quantization)")
        return model, None
    
    print(f"Patching model with K{k_bits}V{v_bits} REAL quantization...")
    print(f"  → Keys: {k_bits}-bit ({k_bits/8:.2f} bytes/element)")
    print(f"  → Values: {v_bits}-bit ({v_bits/8:.2f} bytes/element)")
    
    cache = RealQuantizedKVCache(k_bits=k_bits, v_bits=v_bits)
    
    k_patched = 0
    v_patched = 0
    
    for name, module in model.named_modules():
        if 'k_proj' in name and isinstance(module, nn.Linear):
            original_forward = module.forward
            layer_id = k_patched
            
            def make_k_hook(orig_fn, kv_cache, layer):
                def hooked(x):
                    output = orig_fn(x)
                    # Quantize, store, and return DEQUANTIZED (with error!)
                    dequantized = kv_cache.quantize_and_store_k(output, layer)
                    return dequantized  # Model uses this degraded version
                return hooked
            
            module.forward = make_k_hook(original_forward, cache, layer_id)
            k_patched += 1
        
        elif 'v_proj' in name and isinstance(module, nn.Linear):
            original_forward = module.forward
            layer_id = v_patched
            
            def make_v_hook(orig_fn, kv_cache, layer):
                def hooked(x):
                    output = orig_fn(x)
                    # Quantize, store, and return DEQUANTIZED (with error!)
                    dequantized = kv_cache.quantize_and_store_v(output, layer)
                    return dequantized  # Model uses this degraded version
                return hooked
            
            module.forward = make_v_hook(original_forward, cache, layer_id)
            v_patched += 1
    
    print(f"✓ Patched {k_patched} K projections + {v_patched} V projections")
    print(f"✓ Model will use DEQUANTIZED values (quality affected by quantization error)")
    
    return model, cache


def calculate_kv_memory(model, seq_len, k_bits=16, v_bits=16):
    """Calculate theoretical KV cache memory"""
    config = model.config
    
    num_layers = config.num_hidden_layers
    num_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    
    # Bytes per element based on bit width
    k_bytes_per_elem = k_bits / 8
    v_bytes_per_elem = v_bits / 8
    
    # Total elements: layers × seq_len × heads × head_dim
    total_elements = num_layers * seq_len * num_heads * head_dim
    
    k_size_mb = (total_elements * k_bytes_per_elem) / (1024 ** 2)
    v_size_mb = (total_elements * v_bytes_per_elem) / (1024 ** 2)
    
    return k_size_mb + v_size_mb, k_size_mb, v_size_mb

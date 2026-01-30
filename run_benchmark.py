import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc
import time
from kv_cache_hook import patch_model_attention, calculate_kv_memory


def load_model():
    """Load Llama-2-7B with 4-bit weights"""
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    print(f"\nLoading {model_name}...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    
    print("✓ Model loaded (4-bit weights)")
    return model, tokenizer


def get_gpu_memory():
    """Get current GPU memory in MB"""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 2)


def benchmark_config(k_bits, v_bits, seq_len=512):
    """Benchmark a single KV cache configuration"""
    
    config_name = f"K{k_bits}V{v_bits}"
    print(f"\n{'='*70}")
    print(f"Testing: {config_name} (sequence length: {seq_len})")
    print(f"{'='*70}")
    
    # Load fresh model for each config to avoid hook stacking
    print("Loading fresh model...")
    model, tokenizer = load_model()
    
    # Apply KV cache quantization (REAL quantization!)
    model, quant_cache = patch_model_attention(model, k_bits, v_bits)
    
    # Test prompts
    prompts = [
        "Explain quantum computing in simple terms:",
        "Write a Python function to calculate fibonacci:",
        "What is the theory of relativity?"
    ]
    
    results = {
        "config": config_name,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "seq_len": seq_len,
        "memory_mb": 0,
        "theoretical_kv_mb": 0,
        "outputs": []
    }
    
    # Clear cache before running
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Generate with each prompt
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=seq_len,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results["outputs"].append(generated[:100] + "...")
    
    # Calculate theoretical KV cache size  
    theoretical_kv, k_cache, v_cache = calculate_kv_memory(model, seq_len, k_bits, v_bits)
    fp16_baseline, _, _ = calculate_kv_memory(model, seq_len, 16, 16)
    results["theoretical_kv_mb"] = theoretical_kv
    
    # Get ACTUAL memory from our quantized cache
    if quant_cache is not None:
        actual_cache_mb = quant_cache.memory_mb()
        fp16_equivalent = quant_cache.fp16_equivalent_mb()
        num_layers = len(quant_cache.k_cache_quantized)
        
        # Show actual tensor storage info
        if num_layers > 0:
            print(f"\n🔍 REAL Quantization Stats:")
            print(f"  K elements: {quant_cache.k_elements:,} @ {k_bits}-bit = {quant_cache.k_elements * k_bits / 8 / 1024**2:.2f} MB")
            print(f"  V elements: {quant_cache.v_elements:,} @ {v_bits}-bit = {quant_cache.v_elements * v_bits / 8 / 1024**2:.2f} MB")
            print(f"  Layers cached: {num_layers}")
    else:
        actual_cache_mb = fp16_baseline
        fp16_equivalent = fp16_baseline
    
    results["memory_mb"] = actual_cache_mb
    
    # Peak GPU memory
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    print(f"\n📊 KV Cache Memory:")
    print(f"  FP16 baseline:       {fp16_baseline:.2f} MB")
    if quant_cache is not None:
        print(f"  Quantized (K{k_bits}V{v_bits}): {actual_cache_mb:.2f} MB")
        print(f"  FP16 equivalent:     {fp16_equivalent:.2f} MB")
        savings = fp16_equivalent - actual_cache_mb
        savings_pct = (savings / fp16_equivalent) * 100 if fp16_equivalent > 0 else 0
        print(f"  💾 REAL Savings:     {savings:.2f} MB ({savings_pct:.1f}%)")
    print(f"\nPeak GPU memory: {peak_mem:.2f} MB")
    print(f"\nSample output: {results['outputs'][0]}")
    
    # Clean up model before next config
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


def run_full_benchmark():
    """Run complete benchmark across all configurations"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    print("\n" + "="*70)
    print("MIXED QUANTIZATION BENCHMARK: KV Cache")
    print("="*70)
    
    # Configurations to test
    configs = [
        (16, 16),  # FP16 baseline
        (8, 8),    # K8V8
        (4, 8),    # K4V8
        (8, 4),    # K8V4
        (4, 4),    # K4V4
    ]
    
    seq_len = 512  # Tokens to generate
    
    all_results = []
    
    for k_bits, v_bits in configs:
        # Load fresh model for each config
        result = benchmark_config(k_bits, v_bits, seq_len)
        all_results.append(result)
        
        # Pause between tests
        print("\n⏳ Waiting 3 seconds before next config...")
        time.sleep(3)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - REAL QUANTIZED KV CACHE MEMORY")
    print("="*70)
    
    baseline = all_results[0]
    baseline_kv = baseline["theoretical_kv_mb"]  # FP16 baseline
    
    print(f"\n{'Config':<10} {'FP16 Baseline':<15} {'REAL Storage':<15} {'Memory Saved':<20}")
    print("-" * 70)
    
    for result in all_results:
        config = result["config"]
        actual_mem = result["memory_mb"]
        
        savings_mb = baseline_kv - actual_mem
        savings_pct = (savings_mb / baseline_kv) * 100 if baseline_kv > 0 else 0
        
        print(f"{config:<10} {baseline_kv:>7.2f} MB      "
              f"{actual_mem:>7.2f} MB      "
              f"{savings_mb:>6.2f} MB ({savings_pct:>5.1f}%)")
    
    print("\n" + "="*70)
    print("QUALITY COMPARISON (first 100 chars of output)")
    print("="*70)
    
    for result in all_results:
        print(f"\n{result['config']}:")
        print(f"  {result['outputs'][0]}")
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    run_full_benchmark()

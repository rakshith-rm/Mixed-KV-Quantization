# WIP
# Mixed Quantization: KV Cache Optimization

Test different KV cache quantization configs (K4V8, K8V8, etc.) on **Llama-2-7B** to save memory on 8GB GPU while measuring quality loss.

**What it does:**
- Loads Llama-2-7B with 4-bit weights
- Hooks into attention layers to quantize KV cache
- Tests K16V16, K8V8, K4V8, K8V4, K4V4 configs
- Compares memory usage and output quality

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Login to Hugging Face (for Llama-2 access)
huggingface-cli login
# See SETUP.md for detailed instructions

# 3. Run benchmark (tests all configs)
python run_benchmark.py
```

## Quantization Configs

| Config | K bits | V bits | Expected Savings |
|--------|--------|--------|------------------|
| K16V16 | 16     | 16     | 0% (baseline)    |
| K8V8   | 8      | 8      | ~50%             |
| K4V8   | 4      | 8      | ~62.5%           |
| K8V4   | 8      | 4      | ~62.5%           |
| K4V4   | 4      | 4      | ~75%             |

## What Gets Measured

- **Memory**: Actual GPU memory used during generation
- **Quality**: Perplexity and output comparison vs baseline
- **Tradeoffs**: Memory savings % vs quality degradation %

## Files

- `kv_cache_hook.py` - KV cache quantization logic (hooks into attention layers)
- `run_benchmark.py` - Main benchmark script (tests all configs)

## Interview Points

- Identified KV cache as main memory bottleneck in LLM inference
- Tested mixed quantization (K4V8, K8V8) for 8GB GPU constraints
- Measured memory-quality tradeoff with perplexity metrics
- Can save 50-75% KV cache memory with minimal quality loss

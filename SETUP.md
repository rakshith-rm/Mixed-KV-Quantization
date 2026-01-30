# Setup Instructions

## Important: Hugging Face Access Token

Llama-2 models require accepting Meta's license and using an access token.

### Step 1: Get Llama-2 Access

1. Go to: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
2. Click "Request Access" and accept the license
3. Wait for approval (usually instant)

### Step 2: Create Access Token

1. Go to: https://huggingface.co/settings/tokens
2. Create a new token with "Read" access
3. Copy the token

### Step 3: Login

```bash
# Install huggingface-cli
pip install huggingface-hub

# Login with your token
huggingface-cli login
```

Paste your token when prompted.

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run Tests

```bash
# Quick test (512 tokens)
python test_quantization.py

# Stress test (256-2048 tokens)
python stress_test.py
```

## Alternative: Use Llama-3.2 (No License Required)

If you don't want to wait for Llama-2 access, edit the scripts and change:

```python
model_name = "meta-llama/Llama-2-7b-chat-hf"
```

To:

```python
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # No license needed
```

## GPU Requirements

- **8GB GPU**: Recommended (tested)
- **4GB GPU**: May OOM on longer sequences
- **16GB GPU**: All tests will pass easily

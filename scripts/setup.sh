#!/bin/bash
# Setup script for SARM-based feature control experiments
# Run this first: bash scripts/setup.sh

set -e

echo "=============================================="
echo "Setting up SARM Feature Control Environment"
echo "=============================================="

# Set HuggingFace token for gated model access
# Either export HF_TOKEN in your environment or set it here
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Please set it: export HF_TOKEN=your_token"
    echo "Get your token from: https://huggingface.co/settings/tokens"
fi

# ============================================
# Step 1: Install Python dependencies
# ============================================
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q \
    "torch>=2.1" \
    "transformers>=4.51.0" \
    "accelerate>=0.30" \
    "trl>=0.7.4" \
    "datasets>=2.18" \
    "numpy>=1.24" \
    "tqdm>=4.66" \
    "pyyaml>=6.0" \
    "scipy>=1.10" \
    "pandas" \
    "scikit-learn" \
    "matplotlib" \
    "seaborn" \
    "safetensors" \
    "huggingface_hub"

echo "Dependencies installed"

# ============================================
# Step 2: Login to HuggingFace
# ============================================
echo ""
echo "[2/4] Logging into HuggingFace..."
python3 -c "
from huggingface_hub import login
login(token='$HF_TOKEN', add_to_git_credential=True)
print('Logged in successfully!')
"

# ============================================
# Step 3: Skip Flash Attention (takes too long to compile)
# ============================================
echo ""
echo "[3/5] Skipping Flash Attention (using eager attention instead)..."
echo "  Note: You can install flash-attn later for 2-3x speedup"

# ============================================
# Step 4: Download models
# ============================================
echo ""
echo "[4/5] Downloading models (this may take several minutes)..."

python3 << PYEOF
import os
# Use HF_TOKEN from environment
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
import torch

# Download SARM (without flash attention requirement)
print("Downloading Schrieffer/Llama-SARM-4B...")
tok = AutoTokenizer.from_pretrained("Schrieffer/Llama-SARM-4B")
model = AutoModelForSequenceClassification.from_pretrained(
    "Schrieffer/Llama-SARM-4B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cpu",
    attn_implementation="eager"  # Don't require flash attention
)
print(f"  SARM: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
del model, tok

# Download Llama-3.1-8B-Instruct
print("Downloading meta-llama/Llama-3.1-8B-Instruct...")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    attn_implementation="eager"  # Don't require flash attention
)
print(f"  Llama-3.1-8B: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
del model, tok

print("Models downloaded successfully!")
PYEOF

# ============================================
# Step 5: Create directories
# ============================================
echo ""
echo "[5/5] Creating output directories..."
mkdir -p outputs/{probes,ppo,eval}
mkdir -p data/{train,test}

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Prepare data:  python scripts/prepare_data.py"
echo "  2. Run Phase 2:   bash scripts/run_phase2_probes.sh"
echo "  3. Run Phase 3:   bash scripts/run_phase3_ppo.sh"
echo "  4. Evaluate:      bash scripts/run_evaluation.sh"
echo ""
echo "Or run everything: bash scripts/run_all.sh"

#!/bin/bash
# Phase 3: PPO Training with Feature Controls
# Uses SARM reward model + in-loop feature controls from Phase 2
#
# Run: bash scripts/run_phase3_ppo.sh

set -e

echo "=============================================="
echo "Phase 3: PPO with Feature-Level Controls"
echo "=============================================="

# Configuration - Use all 4 H100s
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Set your HuggingFace token (or export HF_TOKEN in your environment)
export HF_TOKEN="${HF_TOKEN:-your_hf_token_here}"

POLICY_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SARM_MODEL="Schrieffer/Llama-SARM-4B"
OUTPUT_DIR="outputs/ppo"
mkdir -p $OUTPUT_DIR

# ============================================
# Step 1: Build human latent buffer
# ============================================
echo ""
echo "[Step 1/3] Building human latent buffer from chosen responses..."
echo ""

# Get features from Phase 2 (REQUIRED)
if [ ! -f "outputs/probes/selected_features.json" ]; then
    echo "ERROR: outputs/probes/selected_features.json not found!"
    echo "Run Phase 2 first: bash scripts/run_phase2_probes.sh"
    exit 1
fi

LATENT_INDICES=$(python -c "
import json
f = json.load(open('outputs/probes/selected_features.json'))
# Use both promoting and harming features for density ratio
all_features = f['safety_promoting'][:5] + f['safety_harming'][:5]
print(','.join(map(str, all_features)))
")
echo "Using Phase 2 discovered features: $LATENT_INDICES"

# Build human latent buffer from chosen (safe) responses
if [ ! -f "$OUTPUT_DIR/human_latents.pt" ]; then
    echo "Building human latents from data/train/demos.jsonl..."
    python build_human_latents.py \
        --demos_jsonl data/train/demos.jsonl \
        --out $OUTPUT_DIR/human_latents.pt \
        --latent_indices $LATENT_INDICES \
        --batch_size 8
else
    echo "Human latents already exist at $OUTPUT_DIR/human_latents.pt"
fi

# ============================================
# Step 2: Configure feature controls from Phase 2
# ============================================
echo ""
echo "[Step 2/3] Configuring feature controls..."
echo ""

# Configure feature controls from Phase 2 (already verified above)
PENALIZE_FEATURES=$(python -c "
import json
f = json.load(open('outputs/probes/selected_features.json'))
print(','.join(map(str, f['safety_harming'][:3])))
")
# Clamp the worst safety-harming feature
CLAMP_FEATURES=$(python -c "
import json
f = json.load(open('outputs/probes/selected_features.json'))
print(f['safety_harming'][0])
")
echo "Using Phase 2 penalty features: $PENALIZE_FEATURES"

TAU_VALUES="3.0,3.0,3.0"
ALPHA_VALUES="0.2,0.15,0.1"
CLAMP_MAX_VALUES="2.0"

echo "Configuration:"
echo "  Penalty features:  $PENALIZE_FEATURES"
echo "  Tau values:        $TAU_VALUES"
echo "  Alpha values:      $ALPHA_VALUES"
echo "  Clamp features:    $CLAMP_FEATURES"
echo "  Clamp max values:  $CLAMP_MAX_VALUES"
echo "  KL beta:           0.05"
echo "  Latent indices:    $LATENT_INDICES"

# ============================================
# Step 3: Run PPO training
# ============================================
echo ""
echo "[Step 3/3] Starting PPO training..."
echo "Using 4x H100 GPUs with DeepSpeed ZeRO-2"
echo ""

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --config_file configs/accelerate_4gpu.yaml \
    --main_process_port 29500 \
    ppo_feature_control_sarm.py \
    --policy_model $POLICY_MODEL \
    --sarm_model $SARM_MODEL \
    --prompts_jsonl data/train/prompts.jsonl \
    --output_dir $OUTPUT_DIR/checkpoint \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --ppo_epochs 2 \
    --total_steps 200 \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_new_tokens 256 \
    --penalize_features $PENALIZE_FEATURES \
    --tau_values $TAU_VALUES \
    --alpha_values $ALPHA_VALUES \
    --clamp_features $CLAMP_FEATURES \
    --clamp_max_values $CLAMP_MAX_VALUES \
    --human_latents $OUTPUT_DIR/human_latents.pt \
    --beta 0.05 \
    --latent_indices $LATENT_INDICES

echo ""
echo "=============================================="
echo "Phase 3 Complete!"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/checkpoint"
echo ""
echo "Next: Run evaluation"
echo "  bash scripts/run_evaluation.sh"

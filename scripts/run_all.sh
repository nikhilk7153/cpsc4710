#!/bin/bash
# Master Script: Run Full Pipeline
# This runs the complete SARM feature control experiment
#
# Usage:
#   bash scripts/run_all.sh           # Run everything
#   bash scripts/run_all.sh --quick   # Quick test (fewer steps)

set -e

QUICK_MODE=false
if [ "$1" == "--quick" ]; then
    QUICK_MODE=true
    echo "Running in QUICK mode (reduced steps for testing)"
fi

echo "=============================================="
echo "SARM Feature Control - Full Pipeline"
echo "=============================================="
echo ""
echo "Hardware:  4x H100 80GB"
echo "RM Model:  Schrieffer/Llama-SARM-4B"
echo "Policy:    meta-llama/Llama-3.1-8B-Instruct"
echo "Data:      RewardBench-2 + RM-Bench (safety subsets)"
echo ""

if [ "$QUICK_MODE" = true ]; then
    echo "Estimated time (quick mode): ~30-45 min"
else
    echo "Estimated time (full): ~2-3 hours"
fi

echo ""
echo "Pipeline steps:"
echo "  1. Setup - Install deps, download models"
echo "  2. Data prep - Process SARM steering data"
echo "  3. Phase 2 - Identify safety features"
echo "  4. Phase 3 - PPO with feature controls"
echo "  5. Evaluation - Test steering effectiveness"
echo ""

START_TIME=$(date +%s)

# ============================================
# Step 1: Setup
# ============================================
echo ""
echo "========== [1/5] Setup =========="
bash scripts/setup.sh

# ============================================
# Step 2: Prepare Data
# ============================================
echo ""
echo "========== [2/5] Prepare Data =========="
python scripts/prepare_data.py

# ============================================
# Step 3: Phase 2 - Causal Probes
# ============================================
echo ""
echo "========== [3/5] Phase 2: Causal Probes =========="

if [ "$QUICK_MODE" = true ]; then
    # Quick version: just locate features, skip detailed probing
    export CUDA_VISIBLE_DEVICES=0
    
    echo "Quick mode: Running feature location only..."
    python sarm/src/1_steering_locate_features.py \
        --data_path sarm/steering/train/rewardbenchv2/safety.jsonl \
        --model_path Schrieffer/Llama-SARM-4B \
        --output_file outputs/probes/rb2_safety_scores.pt \
        --device cuda:0
    
    # Create selected features from scores
    mkdir -p outputs/probes
    python << 'PYEOF'
import torch
import json

scores = torch.load("outputs/probes/rb2_safety_scores.pt")
top_promoting = torch.argsort(scores, descending=True)[:10].tolist()
top_harming = torch.argsort(scores, descending=False)[:10].tolist()

selected = {
    "safety_promoting": top_promoting,
    "safety_harming": top_harming,
}
with open("outputs/probes/selected_features.json", "w") as f:
    json.dump(selected, f, indent=2)
print(f"Selected features: promoting={top_promoting[:3]}, harming={top_harming[:3]}")
PYEOF
else
    bash scripts/run_phase2_probes.sh
fi

# ============================================
# Step 4: Phase 3 - PPO Training (Optional in quick mode)
# ============================================
echo ""
echo "========== [4/5] Phase 3: PPO Training =========="

if [ "$QUICK_MODE" = true ]; then
    echo "Quick mode: Skipping full PPO training"
    echo "To run PPO: bash scripts/run_phase3_ppo.sh"
    
    # Just build human latents for evaluation
    mkdir -p outputs/ppo
    
    # Get features from Phase 2 (REQUIRED)
    if [ ! -f "outputs/probes/selected_features.json" ]; then
        echo "ERROR: outputs/probes/selected_features.json not found!"
        echo "Phase 2 must complete before this step."
        exit 1
    fi
    
    LATENT_INDICES=$(python -c "
import json
f = json.load(open('outputs/probes/selected_features.json'))
all_features = f['safety_promoting'][:5] + f['safety_harming'][:5]
print(','.join(map(str, all_features)))
")
    
    if [ ! -f "outputs/ppo/human_latents.pt" ]; then
        python build_human_latents.py \
            --demos_jsonl data/train/demos.jsonl \
            --out outputs/ppo/human_latents.pt \
            --latent_indices $LATENT_INDICES \
            --batch_size 8
    fi
else
    bash scripts/run_phase3_ppo.sh
fi

# ============================================
# Step 5: Evaluation
# ============================================
echo ""
echo "========== [5/5] Evaluation =========="
bash scripts/run_evaluation.sh

# ============================================
# Summary
# ============================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Output directories:"
echo "  - Data:       data/train/, data/test/"
echo "  - Probes:     outputs/probes/"
echo "  - PPO:        outputs/ppo/"
echo "  - Evaluation: outputs/eval/"
echo ""
echo "Key files:"
echo "  - Selected features:   outputs/probes/selected_features.json"
echo "  - Steering plots:      outputs/eval/*_dist.png"
echo "  - Evaluation report:   outputs/eval/evaluation_report.txt"
echo ""

if [ "$QUICK_MODE" = true ]; then
    echo "Quick mode completed. To run full PPO training:"
    echo "  bash scripts/run_phase3_ppo.sh"
fi

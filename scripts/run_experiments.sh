#!/bin/bash
# =============================================================================
# Run comprehensive experiments for project report
# Includes: Baseline, Full Method, and Ablations
# =============================================================================

set -e

# Configuration
POLICY_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SARM_MODEL="Schrieffer/Llama-SARM-4B"
PROMPTS="data/train/prompts.jsonl"
OUTPUT_BASE="outputs/experiments"
STEPS=100

# Load discovered features (REQUIRED - no fallback)
if [ ! -f "outputs/probes/selected_features.json" ]; then
    echo "ERROR: outputs/probes/selected_features.json not found!"
    echo "Run Phase 2 first: bash scripts/run_phase2_probes.sh"
    exit 1
fi

SAFETY_HARMING=$(python -c "
import json
f = json.load(open('outputs/probes/selected_features.json'))
print(','.join(map(str, f['safety_harming'][:3])))
")
echo "Using discovered safety-harming features: $SAFETY_HARMING"

mkdir -p "$OUTPUT_BASE"

# Common arguments
COMMON_ARGS="--policy_model $POLICY_MODEL \
    --sarm_model $SARM_MODEL \
    --prompts_jsonl $PROMPTS \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --total_steps $STEPS \
    --max_new_tokens 64 \
    --lora_r 16 \
    --lora_alpha 32"

echo "=============================================="
echo "Running Experiments for Project Report"
echo "=============================================="
echo ""

# =============================================================================
# Experiment 1: BASELINE (No feature control)
# =============================================================================
echo "[1/6] Running BASELINE (no feature control)..."
accelerate launch --config_file configs/accelerate_4gpu.yaml \
    ppo_feature_control_sarm.py \
    $COMMON_ARGS \
    --output_dir "$OUTPUT_BASE/baseline" \
    --experiment_name "baseline" \
    2>&1 | tee "$OUTPUT_BASE/baseline.log"

# =============================================================================
# Experiment 2: FULL METHOD (with feature penalties)
# =============================================================================
echo ""
echo "[2/6] Running FULL METHOD (with feature penalties)..."
accelerate launch --config_file configs/accelerate_4gpu.yaml \
    ppo_feature_control_sarm.py \
    $COMMON_ARGS \
    --output_dir "$OUTPUT_BASE/full_method" \
    --experiment_name "full_method" \
    --penalize_features "$SAFETY_HARMING" \
    --tau_values "3.0,3.0,3.0" \
    --alpha_values "0.1,0.1,0.1" \
    2>&1 | tee "$OUTPUT_BASE/full_method.log"

# =============================================================================
# Experiment 3: ABLATION - Higher alpha (0.2)
# =============================================================================
echo ""
echo "[3/6] Running ABLATION: alpha=0.2..."
accelerate launch --config_file configs/accelerate_4gpu.yaml \
    ppo_feature_control_sarm.py \
    $COMMON_ARGS \
    --output_dir "$OUTPUT_BASE/ablation_alpha_02" \
    --experiment_name "ablation_alpha_02" \
    --penalize_features "$SAFETY_HARMING" \
    --tau_values "3.0,3.0,3.0" \
    --alpha_values "0.2,0.2,0.2" \
    2>&1 | tee "$OUTPUT_BASE/ablation_alpha_02.log"

# =============================================================================
# Experiment 4: ABLATION - Lower alpha (0.05)
# =============================================================================
echo ""
echo "[4/6] Running ABLATION: alpha=0.05..."
accelerate launch --config_file configs/accelerate_4gpu.yaml \
    ppo_feature_control_sarm.py \
    $COMMON_ARGS \
    --output_dir "$OUTPUT_BASE/ablation_alpha_005" \
    --experiment_name "ablation_alpha_005" \
    --penalize_features "$SAFETY_HARMING" \
    --tau_values "3.0,3.0,3.0" \
    --alpha_values "0.05,0.05,0.05" \
    2>&1 | tee "$OUTPUT_BASE/ablation_alpha_005.log"

# =============================================================================
# Experiment 5: ABLATION - Higher tau (4.0)
# =============================================================================
echo ""
echo "[5/6] Running ABLATION: tau=4.0..."
accelerate launch --config_file configs/accelerate_4gpu.yaml \
    ppo_feature_control_sarm.py \
    $COMMON_ARGS \
    --output_dir "$OUTPUT_BASE/ablation_tau_4" \
    --experiment_name "ablation_tau_4" \
    --penalize_features "$SAFETY_HARMING" \
    --tau_values "4.0,4.0,4.0" \
    --alpha_values "0.1,0.1,0.1" \
    2>&1 | tee "$OUTPUT_BASE/ablation_tau_4.log"

# =============================================================================
# Experiment 6: ABLATION - Lower tau (2.0)
# =============================================================================
echo ""
echo "[6/6] Running ABLATION: tau=2.0..."
accelerate launch --config_file configs/accelerate_4gpu.yaml \
    ppo_feature_control_sarm.py \
    $COMMON_ARGS \
    --output_dir "$OUTPUT_BASE/ablation_tau_2" \
    --experiment_name "ablation_tau_2" \
    --penalize_features "$SAFETY_HARMING" \
    --tau_values "2.0,2.0,2.0" \
    --alpha_values "0.1,0.1,0.1" \
    2>&1 | tee "$OUTPUT_BASE/ablation_tau_2.log"

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_BASE/"
echo ""
echo "Experiments:"
echo "  1. baseline/           - No feature control"
echo "  2. full_method/        - Full method (α=0.1, τ=3.0)"
echo "  3. ablation_alpha_02/  - Higher penalty (α=0.2)"
echo "  4. ablation_alpha_005/ - Lower penalty (α=0.05)"
echo "  5. ablation_tau_4/     - Looser threshold (τ=4.0)"
echo "  6. ablation_tau_2/     - Stricter threshold (τ=2.0)"
echo ""
echo "To generate visualizations:"
echo "  python scripts/generate_experiment_figures.py"


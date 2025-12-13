#!/bin/bash
# Phase 2: Causal Feature Identification
# This script identifies safety-related features in SARM using RewardBench-2/RM-Bench data
#
# Run: bash scripts/run_phase2_probes.sh

set -e

echo "=============================================="
echo "Phase 2: Causal Feature Identification"
echo "=============================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3

SARM_MODEL="Schrieffer/Llama-SARM-4B"
OUTPUT_DIR="outputs/probes"
mkdir -p $OUTPUT_DIR

# ============================================
# Step 1: Locate safety-related features using SARM steering data
# ============================================
echo ""
echo "[Step 1/4] Locating safety-related features..."
echo "Using RewardBench-2 safety data from sarm/steering/"
echo "Formula: score_i = (chosen_activation_i - rejected_activation_i) / (sum + C)"
echo ""

# Use the original SARM steering data (already in correct format)
python sarm/src/1_steering_locate_features.py \
    --data_path sarm/steering/train/rewardbenchv2/safety.jsonl \
    --model_path $SARM_MODEL \
    --output_file $OUTPUT_DIR/rb2_safety_scores.pt \
    --device cuda:0

echo "RewardBench-2 feature scores saved"

# Also do RM-Bench for comparison
echo ""
echo "Now processing RM-Bench safety data..."
python sarm/src/1_steering_locate_features.py \
    --data_path sarm/steering/train/rm_bench/safety.jsonl \
    --model_path $SARM_MODEL \
    --output_file $OUTPUT_DIR/rmb_safety_scores.pt \
    --device cuda:0

echo "RM-Bench feature scores saved"

# ============================================
# Step 2: Analyze and select top features
# ============================================
echo ""
echo "[Step 2/4] Analyzing top features from both benchmarks..."
echo ""

python << 'EOF'
import torch
import json
from pathlib import Path

output_dir = Path("outputs/probes")

# Load scores from both benchmarks
rb2_scores = torch.load(output_dir / "rb2_safety_scores.pt")
rmb_scores = torch.load(output_dir / "rmb_safety_scores.pt")

# Average scores across benchmarks for robustness
avg_scores = (rb2_scores + rmb_scores) / 2

# Get top features
top_k = 20

# Safety-promoting: high score = feature activates more on chosen (safe) responses
top_promoting_idx = torch.argsort(avg_scores, descending=True)[:top_k]

# Safety-harming: low score = feature activates more on rejected (unsafe) responses
top_harming_idx = torch.argsort(avg_scores, descending=False)[:top_k]

print("=" * 70)
print("TOP 20 SAFETY-PROMOTING FEATURES")
print("(High score = activates more on safe/chosen responses)")
print("=" * 70)
for i, idx in enumerate(top_promoting_idx):
    rb2_s = rb2_scores[idx].item()
    rmb_s = rmb_scores[idx].item()
    avg_s = avg_scores[idx].item()
    print(f"  {i+1:2d}. Feature {idx.item():5d}: avg={avg_s:.4f} (RB2={rb2_s:.4f}, RMB={rmb_s:.4f})")

print("\n" + "=" * 70)
print("TOP 20 SAFETY-HARMING FEATURES")
print("(Low score = activates more on unsafe/rejected responses)")
print("=" * 70)
for i, idx in enumerate(top_harming_idx):
    rb2_s = rb2_scores[idx].item()
    rmb_s = rmb_scores[idx].item()
    avg_s = avg_scores[idx].item()
    print(f"  {i+1:2d}. Feature {idx.item():5d}: avg={avg_s:.4f} (RB2={rb2_s:.4f}, RMB={rmb_s:.4f})")

# Save selected features
selected = {
    "safety_promoting": top_promoting_idx[:10].tolist(),
    "safety_harming": top_harming_idx[:10].tolist(),
    "rb2_scores_path": str(output_dir / "rb2_safety_scores.pt"),
    "rmb_scores_path": str(output_dir / "rmb_safety_scores.pt"),
}

with open(output_dir / "selected_features.json", "w") as f:
    json.dump(selected, f, indent=2)

print(f"\nSelected features saved to {output_dir}/selected_features.json")

# Also save average scores
torch.save(avg_scores, output_dir / "avg_safety_scores.pt")
print(f"Average scores saved to {output_dir}/avg_safety_scores.pt")
EOF

# ============================================
# Step 3: Run causal intervention tests
# ============================================
echo ""
echo "[Step 3/4] Running causal intervention tests on top safety-harming features..."
echo ""

# Get top 3 safety-harming features
FEATURES=$(python -c "
import json
f = json.load(open('outputs/probes/selected_features.json'))
print(' '.join(map(str, f['safety_harming'][:3])))
")

for FEAT in $FEATURES; do
    echo "Testing causal intervention on feature $FEAT..."
    python causal_probe_sarm.py \
        --data_jsonl data/train/probes_safety.jsonl \
        --out_csv $OUTPUT_DIR/causal_probe_f${FEAT}.csv \
        --feature $FEAT \
        --set_value 0.0 \
        --batch_size 4
done

# ============================================
# Step 4: Summarize causal probe results
# ============================================
echo ""
echo "[Step 4/4] Summarizing causal probe results..."
echo ""

python << 'EOF'
import pandas as pd
import json
from pathlib import Path

output_dir = Path("outputs/probes")
selected = json.load(open(output_dir / "selected_features.json"))

print("=" * 70)
print("CAUSAL INTERVENTION SUMMARY")
print("Setting feature to 0.0 and measuring reward change")
print("=" * 70)

for feat in selected["safety_harming"][:3]:
    csv_path = output_dir / f"causal_probe_f{feat}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        
        mean_delta = df["delta"].mean()
        std_delta = df["delta"].std()
        pos_delta_pct = (df["delta"] > 0).mean() * 100
        
        print(f"\nFeature {feat}:")
        print(f"  Mean reward change (delta):  {mean_delta:.4f} ± {std_delta:.4f}")
        print(f"  Samples with positive delta: {pos_delta_pct:.1f}%")
        print(f"  Interpretation: Setting this feature to 0 {'INCREASES' if mean_delta > 0 else 'DECREASES'} reward")
        
        # Positive delta when zeroing a safety-harming feature means:
        # the feature was contributing negatively to reward, so zeroing it helps
        if mean_delta > 0:
            print(f"  ✓ Confirmed: Feature {feat} contributes to LOWER reward (safety-harming)")
        else:
            print(f"  ? Feature {feat} may not be purely safety-harming")

print("\n" + "=" * 70)
EOF

echo ""
echo "=============================================="
echo "Phase 2 Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  - RewardBench-2 scores: $OUTPUT_DIR/rb2_safety_scores.pt"
echo "  - RM-Bench scores:      $OUTPUT_DIR/rmb_safety_scores.pt"
echo "  - Average scores:       $OUTPUT_DIR/avg_safety_scores.pt"
echo "  - Selected features:    $OUTPUT_DIR/selected_features.json"
echo "  - Causal probe CSVs:    $OUTPUT_DIR/causal_probe_f*.csv"
echo ""
echo "Next: Build human latents and run Phase 3"
echo "  bash scripts/run_phase3_ppo.sh"

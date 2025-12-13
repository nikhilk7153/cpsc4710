#!/bin/bash
# Evaluation Script
# Tests steering effectiveness on RewardBench-2 and RM-Bench test sets
#
# Run: bash scripts/run_evaluation.sh

set -e

echo "=============================================="
echo "Evaluation: Steering Effectiveness"
echo "=============================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3
OUTPUT_DIR="outputs/eval"
mkdir -p $OUTPUT_DIR

SARM_MODEL="Schrieffer/Llama-SARM-4B"

# ============================================
# Step 1: Create steering config from Phase 2 features
# ============================================
echo ""
echo "[Step 1/4] Creating steering configuration..."
echo ""

python << 'EOF'
import json
from pathlib import Path

output_dir = Path("outputs/eval")
output_dir.mkdir(parents=True, exist_ok=True)

# Load features from Phase 2 (REQUIRED)
probes_dir = Path("outputs/probes")
features_file = probes_dir / "selected_features.json"
if not features_file.exists():
    print("ERROR: outputs/probes/selected_features.json not found!")
    print("Run Phase 2 first: bash scripts/run_phase2_probes.sh")
    sys.exit(1)

selected = json.load(open(features_file))
safety_harming = selected["safety_harming"][:5]
print(f"Using Phase 2 discovered features: {safety_harming}")

# Create steering config: suppress safety-harming features
# Format: {feature_id: [action, value]}
# "*" = multiply, "+" = add
steering_config = {}
for feat in safety_harming:
    steering_config[str(feat)] = ["*", 0.0]  # Zero out the feature

with open(output_dir / "steering_config.json", "w") as f:
    json.dump(steering_config, f, indent=2)

print(f"Steering config: suppress features {safety_harming}")
print(f"Saved to {output_dir}/steering_config.json")
EOF

# ============================================
# Step 2: Test on RewardBench-2 safety subset
# ============================================
echo ""
echo "[Step 2/4] Testing steering on RewardBench-2 safety data..."
echo ""

# Test on chosen (safe) responses - expect reward to DECREASE when we suppress safety features
# Wait, actually we're suppressing HARMING features, so reward should INCREASE
echo "Testing on safety_c (chosen/safe responses)..."
python sarm/src/2_steering_test.py \
    --data_path sarm/steering/test/rewardbenchv2/safety_c.jsonl \
    --model_path $SARM_MODEL \
    --steering_path $OUTPUT_DIR/steering_config.json \
    --output_path $OUTPUT_DIR/rb2_safety_chosen_results.json \
    --plot_path $OUTPUT_DIR/rb2_safety_chosen_dist.png \
    --num_example 400 \
    --device cuda:0

# Test on rejected (unsafe) responses - suppressing harming features should help more here
echo ""
echo "Testing on safety_j (rejected/unsafe responses)..."
python sarm/src/2_steering_test.py \
    --data_path sarm/steering/test/rewardbenchv2/safety_j.jsonl \
    --model_path $SARM_MODEL \
    --steering_path $OUTPUT_DIR/steering_config.json \
    --output_path $OUTPUT_DIR/rb2_safety_rejected_results.json \
    --plot_path $OUTPUT_DIR/rb2_safety_rejected_dist.png \
    --num_example 400 \
    --device cuda:0

# ============================================
# Step 3: Test on RM-Bench safety subset
# ============================================
echo ""
echo "[Step 3/4] Testing steering on RM-Bench safety data..."
echo ""

echo "Testing on RM-Bench safety_c..."
python sarm/src/2_steering_test.py \
    --data_path sarm/steering/test/rm_bench/safety_c.jsonl \
    --model_path $SARM_MODEL \
    --steering_path $OUTPUT_DIR/steering_config.json \
    --output_path $OUTPUT_DIR/rmb_safety_chosen_results.json \
    --plot_path $OUTPUT_DIR/rmb_safety_chosen_dist.png \
    --num_example 400 \
    --device cuda:0

echo ""
echo "Testing on RM-Bench safety_j..."
python sarm/src/2_steering_test.py \
    --data_path sarm/steering/test/rm_bench/safety_j.jsonl \
    --model_path $SARM_MODEL \
    --steering_path $OUTPUT_DIR/steering_config.json \
    --output_path $OUTPUT_DIR/rmb_safety_rejected_results.json \
    --plot_path $OUTPUT_DIR/rmb_safety_rejected_dist.png \
    --num_example 400 \
    --device cuda:0

# ============================================
# Step 4: Generate summary report
# ============================================
echo ""
echo "[Step 4/4] Generating evaluation report..."
echo ""

python << 'EOF'
import json
import statistics
from pathlib import Path

output_dir = Path("outputs/eval")

def analyze_results(path, name):
    """Analyze steering test results."""
    if not path.exists():
        return None
    
    data = json.load(open(path))
    
    before = [v["reward"] for v in data.values() if isinstance(v, dict) and "reward" in v]
    after = [v["reward_steered"] for v in data.values() if isinstance(v, dict) and "reward_steered" in v]
    
    if not before or not after:
        return None
    
    return {
        "name": name,
        "n_samples": len(before),
        "before_mean": statistics.mean(before),
        "before_std": statistics.stdev(before) if len(before) > 1 else 0,
        "after_mean": statistics.mean(after),
        "after_std": statistics.stdev(after) if len(after) > 1 else 0,
        "delta_mean": statistics.mean(after) - statistics.mean(before),
    }

# Collect results
results = []
results.append(analyze_results(output_dir / "rb2_safety_chosen_results.json", "RB2 Safety Chosen"))
results.append(analyze_results(output_dir / "rb2_safety_rejected_results.json", "RB2 Safety Rejected"))
results.append(analyze_results(output_dir / "rmb_safety_chosen_results.json", "RMB Safety Chosen"))
results.append(analyze_results(output_dir / "rmb_safety_rejected_results.json", "RMB Safety Rejected"))
results = [r for r in results if r is not None]

# Print report
report = []
report.append("=" * 70)
report.append("STEERING EVALUATION REPORT")
report.append("=" * 70)
report.append("")
report.append("Steering action: Suppress (multiply by 0) safety-harming features")
report.append("")

# Load which features were suppressed
config_path = output_dir / "steering_config.json"
if config_path.exists():
    config = json.load(open(config_path))
    report.append(f"Suppressed features: {list(config.keys())}")
    report.append("")

report.append("-" * 70)
report.append(f"{'Dataset':<25} {'N':>6} {'Before':>12} {'After':>12} {'Delta':>10}")
report.append("-" * 70)

for r in results:
    report.append(
        f"{r['name']:<25} {r['n_samples']:>6} "
        f"{r['before_mean']:>12.4f} {r['after_mean']:>12.4f} "
        f"{r['delta_mean']:>+10.4f}"
    )

report.append("-" * 70)
report.append("")

# Interpretation
report.append("INTERPRETATION:")
report.append("")

for r in results:
    delta = r["delta_mean"]
    if "Rejected" in r["name"]:
        # For rejected (unsafe) responses, positive delta = steering helped (increased reward)
        if delta > 0.1:
            report.append(f"✓ {r['name']}: Steering INCREASED reward by {delta:.4f}")
            report.append(f"  → Suppressing harmful features improved unsafe response scores")
        elif delta < -0.1:
            report.append(f"✗ {r['name']}: Steering DECREASED reward by {abs(delta):.4f}")
        else:
            report.append(f"○ {r['name']}: Minimal change ({delta:+.4f})")
    else:
        # For chosen (safe) responses, we expect minimal change
        if abs(delta) < 0.1:
            report.append(f"✓ {r['name']}: Minimal change ({delta:+.4f}) - good specificity")
        else:
            report.append(f"! {r['name']}: Unexpected change of {delta:+.4f}")

report.append("")
report.append("=" * 70)

# Print and save
report_text = "\n".join(report)
print(report_text)

with open(output_dir / "evaluation_report.txt", "w") as f:
    f.write(report_text)

print(f"\nReport saved to {output_dir}/evaluation_report.txt")
EOF

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  - Steering config:      $OUTPUT_DIR/steering_config.json"
echo "  - RB2 chosen results:   $OUTPUT_DIR/rb2_safety_chosen_results.json"
echo "  - RB2 rejected results: $OUTPUT_DIR/rb2_safety_rejected_results.json"
echo "  - RMB chosen results:   $OUTPUT_DIR/rmb_safety_chosen_results.json"
echo "  - RMB rejected results: $OUTPUT_DIR/rmb_safety_rejected_results.json"
echo "  - Distribution plots:   $OUTPUT_DIR/*_dist.png"
echo "  - Evaluation report:    $OUTPUT_DIR/evaluation_report.txt"

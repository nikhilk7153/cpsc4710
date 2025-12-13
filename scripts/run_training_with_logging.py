"""
Run PPO training with detailed logging for visualization.
Saves metrics to CSV for proper analysis.
"""
import subprocess
import json
import re
import sys
import csv
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("outputs/experiment_logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parse existing log
def parse_existing_log(log_path):
    """Parse training metrics from existing log file."""
    metrics = []
    
    with open(log_path) as f:
        for line in f:
            # Match pattern: reward=-0.102, rm=-0.102, pen=0.000, loss=0.033
            match = re.search(
                r'(\d+)/\d+.*reward=(-?[\d.]+).*rm=(-?[\d.]+).*pen=([\d.]+).*loss=([\d.]+)',
                line
            )
            if match:
                step = int(match.group(1))
                metrics.append({
                    'step': step,
                    'total_reward': float(match.group(2)),
                    'rm_score': float(match.group(3)),
                    'penalty': float(match.group(4)),
                    'loss': float(match.group(5)),
                })
    
    # Remove duplicates (keep last per step)
    seen = {}
    for m in metrics:
        seen[m['step']] = m
    return list(seen.values())


def main():
    # Parse the existing 100-step training log
    log_100 = Path("outputs/ppo_training.log")
    if log_100.exists():
        print("Parsing 100-step training log...")
        metrics_100 = parse_existing_log(log_100)
        
        # Save to CSV
        csv_path = OUTPUT_DIR / "training_100steps.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'total_reward', 'rm_score', 'penalty', 'loss'])
            writer.writeheader()
            writer.writerows(metrics_100)
        print(f"Saved {len(metrics_100)} datapoints to {csv_path}")
        
        # Print summary
        print("\nTraining Summary (100 steps):")
        print("-" * 50)
        for m in metrics_100:
            print(f"Step {m['step']:3d}: reward={m['total_reward']:+.3f}, "
                  f"rm={m['rm_score']:+.3f}, pen={m['penalty']:.3f}, loss={m['loss']:.3f}")
    
    # Also check for evaluation results
    eval_dir = Path("outputs/eval")
    if eval_dir.exists():
        print("\nEvaluation Results:")
        print("-" * 50)
        
        for result_file in sorted(eval_dir.glob("*_results.json")):
            with open(result_file) as f:
                data = json.load(f)
            
            name = result_file.stem.replace("_results", "")
            print(f"\n{name}:")
            print(f"  Samples: {data.get('num_samples', 'N/A')}")
            print(f"  Mean baseline: {data.get('mean_score_baseline', 'N/A'):.4f}")
            print(f"  Mean steered:  {data.get('mean_score_steered', 'N/A'):.4f}")
            print(f"  Delta:         {data.get('mean_delta', 'N/A'):.4f}")
    
    # Check for Phase 2 results
    probes_file = Path("outputs/probes/selected_features.json")
    if probes_file.exists():
        print("\nPhase 2 - Discovered Features:")
        print("-" * 50)
        with open(probes_file) as f:
            features = json.load(f)
        
        print("Safety-Promoting:", features['safety_promoting'][:5])
        print("Safety-Harming:  ", features['safety_harming'][:5])


if __name__ == "__main__":
    main()


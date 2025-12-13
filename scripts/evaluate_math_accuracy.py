"""
Evaluate math accuracy using the reference answers from RewardBench-2
"""
import json
import re
from pathlib import Path

def extract_final_answer(text):
    """Extract numerical answer from response"""
    # Look for boxed answers
    boxed = re.search(r'\\boxed{([^}]+)}', text)
    if boxed:
        return boxed.group(1).strip()
    
    # Look for "answer is X" patterns
    answer_patterns = [
        r'answer is[:\s]+([A-Da-d0-9.-]+)',
        r'correct answer is[:\s]+([A-Da-d0-9.-]+)',
        r'= ([0-9.-]+)$',
        r'([A-Da-d])\s*$',  # Multiple choice
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Last number in text
    numbers = re.findall(r'[-+]?[0-9]+\.?[0-9]*', text)
    if numbers:
        return numbers[-1]
    
    return None

def main():
    results_path = Path("outputs/experiments/qwen_1.5b_full_ft/results.json")
    if not results_path.exists():
        print(f"Results not found at {results_path}")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    print("="*70)
    print("MATH ACCURACY EVALUATION")
    print("="*70)
    
    # Load math eval pairs with reference answers
    eval_pairs = []
    with open("data/rewardbench2/math/eval_pairs.jsonl") as f:
        for line in f:
            eval_pairs.append(json.loads(line))
    
    # Create lookup by prompt
    reference_by_prompt = {}
    for pair in eval_pairs:
        reference_by_prompt[pair["prompt"][:100]] = pair["chosen"]  # First 100 chars as key
    
    print("\n--- BASELINE RESPONSES ---")
    for r in results["baseline_responses"][:5]:
        prompt_key = r["prompt"][:100]
        ref = reference_by_prompt.get(prompt_key, "N/A")
        print(f"\nQ: {r['prompt'][:80]}...")
        print(f"Model: {r['response'][:150]}...")
        print(f"Reference: {ref[:150]}..." if ref != "N/A" else "Reference: N/A")
        print(f"SARM Score: {r['score']:.4f}")
    
    print("\n" + "="*70)
    print("\n--- FINAL (TRAINED) RESPONSES ---")
    for r in results["final_responses"][:5]:
        prompt_key = r["prompt"][:100]
        ref = reference_by_prompt.get(prompt_key, "N/A")
        print(f"\nQ: {r['prompt'][:80]}...")
        print(f"Model: {r['response'][:150]}...")
        print(f"Reference: {ref[:150]}..." if ref != "N/A" else "Reference: N/A")
        print(f"SARM Score: {r['score']:.4f}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model: {results['model']}")
    print(f"Method: {results['method']} ({results['trainable_params']:,} params)")
    print(f"Dataset: {results['dataset']}")
    print(f"Steps: {results['steps']}")
    print(f"\nBaseline SARM: {results['baseline_score']:.4f}")
    print(f"Final SARM:    {results['final_score']:.4f}")
    print(f"Change:        {results['improvement']:+.4f}")
    
    # Training dynamics
    log = results['training_log']
    print(f"\nTraining Dynamics:")
    print(f"  First 10 steps avg reward:  {sum(l['reward'] for l in log[:10])/10:.4f}")
    print(f"  Last 10 steps avg reward:   {sum(l['reward'] for l in log[-10:])/10:.4f}")

if __name__ == "__main__":
    main()

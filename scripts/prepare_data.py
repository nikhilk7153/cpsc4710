#!/usr/bin/env python3
"""
prepare_data.py

Prepares data from the SARM steering datasets (RewardBench-2 and RM-Bench).
These are already included in sarm/steering/ directory.

This script:
1. Converts the existing SARM steering data to our format
2. Creates prompts file for PPO training
3. Creates demos file for human latent buffer

The SARM steering data uses the format:
  {"prompt": ..., "chosen": ..., "rejected": ..., "question_id": ..., "id": ...}

Output files:
  data/train/probes.jsonl       - For causal probing (chosen/rejected pairs)
  data/train/prompts.jsonl      - For PPO training (prompts only)
  data/train/demos.jsonl        - For human latent buffer (chosen responses)
  data/test/test_safety.jsonl   - For evaluation (safety subset)
  data/test/test_complement.jsonl - For evaluation (complement/non-safety)
"""

import json
from pathlib import Path
from collections import defaultdict


def load_jsonl(path: Path) -> list:
    """Load a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list, path: Path):
    """Save data to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved: {path} ({len(data)} examples)")


def main():
    print("=" * 60)
    print("Preparing Data from SARM Steering Datasets")
    print("=" * 60)
    
    sarm_dir = Path("sarm/steering")
    data_dir = Path("data")
    
    # Create directories
    (data_dir / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "test").mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # Load SARM steering data
    # ============================================
    print("\n[1/4] Loading SARM steering data...")
    
    # Training data: RewardBench-2 + RM-Bench safety sets
    rb2_safety = load_jsonl(sarm_dir / "train/rewardbenchv2/safety.jsonl")
    rb2_complement = load_jsonl(sarm_dir / "train/rewardbenchv2/safety_C.jsonl")
    rmb_safety = load_jsonl(sarm_dir / "train/rm_bench/safety.jsonl")
    rmb_complement = load_jsonl(sarm_dir / "train/rm_bench/safety_C.jsonl")
    
    print(f"  RewardBench-2 safety:     {len(rb2_safety)}")
    print(f"  RewardBench-2 complement: {len(rb2_complement)}")
    print(f"  RM-Bench safety:          {len(rmb_safety)}")
    print(f"  RM-Bench complement:      {len(rmb_complement)}")
    
    # Test data
    rb2_test_c = load_jsonl(sarm_dir / "test/rewardbenchv2/safety_c.jsonl")
    rb2_test_j = load_jsonl(sarm_dir / "test/rewardbenchv2/safety_j.jsonl")
    rb2_test_complement = load_jsonl(sarm_dir / "test/rewardbenchv2/safety_C_cj.jsonl")
    rmb_test_c = load_jsonl(sarm_dir / "test/rm_bench/safety_c.jsonl")
    rmb_test_j = load_jsonl(sarm_dir / "test/rm_bench/safety_j.jsonl")
    rmb_test_complement = load_jsonl(sarm_dir / "test/rm_bench/safety_C_cj.jsonl")
    
    print(f"  RB2 test chosen:          {len(rb2_test_c)}")
    print(f"  RB2 test rejected:        {len(rb2_test_j)}")
    print(f"  RB2 test complement:      {len(rb2_test_complement)}")
    print(f"  RMB test chosen:          {len(rmb_test_c)}")
    print(f"  RMB test rejected:        {len(rmb_test_j)}")
    print(f"  RMB test complement:      {len(rmb_test_complement)}")
    
    # ============================================
    # Create probe data for Phase 2
    # ============================================
    print("\n[2/4] Creating probe data for Phase 2 (causal testing)...")
    
    # Combine safety data from both benchmarks for probing
    probes_safety = []
    seen_prompts = set()
    
    for item in rb2_safety + rmb_safety:
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        
        if prompt and chosen and rejected and prompt not in seen_prompts:
            probes_safety.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source": "safety"
            })
            seen_prompts.add(prompt)
    
    # Also include complement data for contrast
    probes_complement = []
    for item in (rb2_complement + rmb_complement)[:2000]:  # Limit complement size
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        
        if prompt and chosen and rejected and prompt not in seen_prompts:
            probes_complement.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source": "complement"
            })
            seen_prompts.add(prompt)
    
    save_jsonl(probes_safety, data_dir / "train/probes_safety.jsonl")
    save_jsonl(probes_complement, data_dir / "train/probes_complement.jsonl")
    save_jsonl(probes_safety + probes_complement, data_dir / "train/probes.jsonl")
    
    # ============================================
    # Create PPO prompts and human demos
    # ============================================
    print("\n[3/4] Creating PPO prompts and human demos...")
    
    prompts = []
    demos = []
    seen_prompts = set()
    
    # Use all safety + complement data for PPO training prompts
    all_train = rb2_safety + rb2_complement + rmb_safety + rmb_complement
    
    for item in all_train:
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        
        if prompt and prompt not in seen_prompts:
            prompts.append({"prompt": prompt})
            seen_prompts.add(prompt)
            
            if chosen:
                demos.append({"prompt": prompt, "response": chosen})
    
    save_jsonl(prompts, data_dir / "train/prompts.jsonl")
    save_jsonl(demos, data_dir / "train/demos.jsonl")
    
    # ============================================
    # Create test data
    # ============================================
    print("\n[4/4] Creating test data...")
    
    # Test safety (chosen responses - steering should change these)
    test_safety_chosen = []
    for item in rb2_test_c + rmb_test_c:
        # Test files have different format: {"question": ..., "answer": ...}
        q = item.get("question", item.get("prompt", ""))
        a = item.get("answer", item.get("chosen", ""))
        if q and a:
            test_safety_chosen.append({"question": q, "answer": a, "type": "chosen"})
    
    # Test safety (rejected responses - steering should NOT change these much)
    test_safety_rejected = []
    for item in rb2_test_j + rmb_test_j:
        q = item.get("question", item.get("prompt", ""))
        a = item.get("answer", item.get("rejected", ""))
        if q and a:
            test_safety_rejected.append({"question": q, "answer": a, "type": "rejected"})
    
    # Test complement (steering should NOT change these)
    test_complement = []
    for item in (rb2_test_complement + rmb_test_complement)[:1000]:
        q = item.get("question", item.get("prompt", ""))
        a = item.get("answer", item.get("chosen", item.get("rejected", "")))
        if q and a:
            test_complement.append({"question": q, "answer": a, "type": "complement"})
    
    save_jsonl(test_safety_chosen, data_dir / "test/safety_chosen.jsonl")
    save_jsonl(test_safety_rejected, data_dir / "test/safety_rejected.jsonl")
    save_jsonl(test_complement, data_dir / "test/complement.jsonl")
    save_jsonl(test_safety_chosen + test_safety_rejected, data_dir / "test/safety_all.jsonl")
    
    # ============================================
    # Summary
    # ============================================
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    
    print("\nTraining data:")
    print(f"  data/train/probes_safety.jsonl     - {len(probes_safety)} pairs for Phase 2")
    print(f"  data/train/probes_complement.jsonl - {len(probes_complement)} pairs for Phase 2")
    print(f"  data/train/probes.jsonl            - {len(probes_safety) + len(probes_complement)} total probe pairs")
    print(f"  data/train/prompts.jsonl           - {len(prompts)} prompts for PPO")
    print(f"  data/train/demos.jsonl             - {len(demos)} demos for human latents")
    
    print("\nTest data:")
    print(f"  data/test/safety_chosen.jsonl      - {len(test_safety_chosen)} (expect steering to change)")
    print(f"  data/test/safety_rejected.jsonl    - {len(test_safety_rejected)} (expect minimal change)")
    print(f"  data/test/complement.jsonl         - {len(test_complement)} (expect minimal change)")
    
    print("\n" + "=" * 60)
    print("Ready for Phase 2!")
    print("Run: bash scripts/run_phase2_probes.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()

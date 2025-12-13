#!/usr/bin/env python3
"""
locate_features.py

Identifies safety-related features in SARM by comparing latent activations
on chosen (safe) vs rejected (unsafe) responses.

Uses sarm_wrapper.py which handles all model loading properly.
"""

import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sarm_wrapper import SARMRewardModel


def load_jsonl(path: str) -> list:
    """Load a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Locate safety features in SARM")
    parser.add_argument("--data_path", type=str, required=True, help="JSONL with prompt/chosen/rejected")
    parser.add_argument("--output_file", type=str, default="outputs/probes/safety_scores.pt")
    parser.add_argument("--model_name", type=str, default="Schrieffer/Llama-SARM-4B")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("=" * 60)
    print("Locating Safety Features in SARM")
    print("=" * 60)
    
    # Load SARM
    print(f"\nLoading SARM model: {args.model_name}")
    rm = SARMRewardModel(model_name=args.model_name, device=args.device)
    latent_size = rm.sae.latent_size
    print(f"Latent size: {latent_size}")
    
    # Load data
    print(f"\nLoading data: {args.data_path}")
    data = load_jsonl(args.data_path)
    print(f"Loaded {len(data)} examples")
    
    # Accumulate activations
    chosen_sum = torch.zeros(latent_size, device="cpu")
    rejected_sum = torch.zeros(latent_size, device="cpu")
    count = 0
    
    print("\nExtracting latent activations...")
    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data[i:i + args.batch_size]
        
        prompts = [ex.get("prompt", "") for ex in batch]
        chosen = [ex.get("chosen", "") for ex in batch]
        rejected = [ex.get("rejected", "") for ex in batch]
        
        # Skip if any are empty
        if not all(prompts) or not all(chosen) or not all(rejected):
            continue
        
        # Get latents for chosen responses
        z_chosen = rm.pooled_latents(prompts, chosen)
        chosen_sum += z_chosen.sum(dim=0).cpu().float()
        
        # Get latents for rejected responses
        z_rejected = rm.pooled_latents(prompts, rejected)
        rejected_sum += z_rejected.sum(dim=0).cpu().float()
        
        count += len(batch)
    
    print(f"\nProcessed {count} examples")
    
    # Compute feature scores
    # score_i = (chosen_i - rejected_i) / (chosen_i + rejected_i + C)
    # High positive score = feature activates more on chosen (safe) responses
    # Low negative score = feature activates more on rejected (unsafe) responses
    c = chosen_sum / count
    j = rejected_sum / count
    denominator = c + j
    C = denominator.mean() + 1e-9  # Stability constant
    scores = (c - j) / (denominator + C)
    
    # Save scores
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scores, output_path)
    print(f"\nSaved scores to {output_path}")
    
    # Print top features
    top_k = 10
    top_promoting = torch.argsort(scores, descending=True)[:top_k]
    top_harming = torch.argsort(scores, descending=False)[:top_k]
    
    print("\n" + "=" * 60)
    print(f"TOP {top_k} SAFETY-PROMOTING FEATURES")
    print("(activate more on chosen/safe responses)")
    print("=" * 60)
    for i, idx in enumerate(top_promoting):
        print(f"  {i+1:2d}. Feature {idx.item():5d}: score = {scores[idx].item():.4f}")
    
    print("\n" + "=" * 60)
    print(f"TOP {top_k} SAFETY-HARMING FEATURES")
    print("(activate more on rejected/unsafe responses)")
    print("=" * 60)
    for i, idx in enumerate(top_harming):
        print(f"  {i+1:2d}. Feature {idx.item():5d}: score = {scores[idx].item():.4f}")
    
    # Save selected features
    selected = {
        "safety_promoting": top_promoting.tolist(),
        "safety_harming": top_harming.tolist(),
        "scores_path": str(output_path),
    }
    selected_path = output_path.parent / "selected_features.json"
    with open(selected_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nSaved selected features to {selected_path}")


if __name__ == "__main__":
    main()


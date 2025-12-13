"""
Identify features the SARM way:
1. Compute differential activation on chosen vs rejected safety data
2. Find top features by value head weight (negative = safety-harming)
3. Interpret features with GPT-4

This replicates the SARM paper's methodology exactly.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI


def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def get_sequence_length(input_ids, pad_token_id):
    """Get position of last non-pad token."""
    seq_len = (input_ids == pad_token_id).int().argmax(-1) - 1
    seq_len = seq_len % input_ids.shape[-1]
    return seq_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Schrieffer/Llama-SARM-4B")
    parser.add_argument("--data_path", default="sarm/steering/train/rewardbenchv2/safety.jsonl")
    parser.add_argument("--output_dir", default="outputs/features")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top features to identify")
    parser.add_argument("--interpret", action="store_true", help="Use GPT-4 to interpret features")
    parser.add_argument("--openai_key", type=str, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SARM Feature Identification")
    print("="*60)
    
    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("[2/5] Loading SARM model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map=args.device,
    ).eval()
    
    # =========================================================================
    # METHOD 1: Value Head Weights (SARM Paper Method)
    # =========================================================================
    print("\n[3/5] Analyzing value head weights...")
    
    # Get the score layer weights
    score_weights = model.score.weight.squeeze().float().cpu()  # Shape: [65536]
    
    # Features with NEGATIVE weights = safety-harming (penalized by RM)
    # Features with POSITIVE weights = safety-promoting (rewarded by RM)
    
    safety_harming_by_weight = torch.argsort(score_weights)[:args.top_k].tolist()
    safety_promoting_by_weight = torch.argsort(score_weights, descending=True)[:args.top_k].tolist()
    
    print(f"\nBy Value Head Weights:")
    print(f"  Safety-harming (most negative weights): {safety_harming_by_weight}")
    print(f"  Safety-promoting (most positive weights): {safety_promoting_by_weight}")
    
    for i, feat in enumerate(safety_harming_by_weight[:5]):
        print(f"    Feature {feat}: weight = {score_weights[feat]:.6f}")
    
    # =========================================================================
    # METHOD 2: Differential Activation (Your Phase 2 Method)
    # =========================================================================
    print("\n[4/5] Computing differential activation on safety data...")
    
    # Load safety data
    data = load_jsonl(args.data_path)
    print(f"  Loaded {len(data)} preference pairs")
    
    # Hook to capture SAE features
    captured_features = []
    
    def hook_fn(module, input, output):
        pre_acts = output.detach() + model.sae.latent_bias
        if model.sarm_use_activation:
            features = model.sae.get_latents(pre_acts)
        else:
            features = pre_acts
        captured_features.append(features)
    
    hook = model.sae.encoder.register_forward_hook(hook_fn)
    
    try:
        sae_dim = model.config.sarm_param['sae_latent_size']
        chosen_sum = torch.zeros(sae_dim, device='cpu')
        rejected_sum = torch.zeros(sae_dim, device='cpu')
        
        for row in tqdm(data, desc="Processing pairs"):
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]
            
            # Process chosen
            captured_features.clear()
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            enc = tokenizer(text, truncation=True, max_length=2048, return_tensors="pt").to(args.device)
            
            with torch.no_grad():
                model(**enc)
            
            features_c = captured_features[0]
            seq_len = get_sequence_length(enc['input_ids'], tokenizer.pad_token_id)
            latent_c = features_c[0, seq_len].cpu()
            chosen_sum += latent_c
            
            # Process rejected
            captured_features.clear()
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            enc = tokenizer(text, truncation=True, max_length=2048, return_tensors="pt").to(args.device)
            
            with torch.no_grad():
                model(**enc)
            
            features_j = captured_features[0]
            seq_len = get_sequence_length(enc['input_ids'], tokenizer.pad_token_id)
            latent_j = features_j[0, seq_len].cpu()
            rejected_sum += latent_j
        
        # Compute differential score (SARM formula)
        denominator = chosen_sum + rejected_sum
        diff_score = (chosen_sum - rejected_sum) / (denominator + denominator.mean() + 1e-9)
        
        # Features where chosen >> rejected = safety-promoting
        # Features where rejected >> chosen = safety-harming
        safety_promoting_by_diff = torch.argsort(diff_score, descending=True)[:args.top_k].tolist()
        safety_harming_by_diff = torch.argsort(diff_score)[:args.top_k].tolist()
        
        print(f"\nBy Differential Activation:")
        print(f"  Safety-promoting (chosen >> rejected): {safety_promoting_by_diff}")
        print(f"  Safety-harming (rejected >> chosen): {safety_harming_by_diff}")
        
    finally:
        hook.remove()
    
    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n[5/5] Saving results...")
    
    results = {
        "method_1_value_head_weights": {
            "description": "Features identified by value head weight (SARM paper method)",
            "safety_harming": safety_harming_by_weight,
            "safety_promoting": safety_promoting_by_weight,
            "weights": {str(f): float(score_weights[f]) for f in safety_harming_by_weight + safety_promoting_by_weight}
        },
        "method_2_differential_activation": {
            "description": "Features identified by differential activation on safety data",
            "safety_harming": safety_harming_by_diff,
            "safety_promoting": safety_promoting_by_diff,
            "scores": {str(f): float(diff_score[f]) for f in safety_harming_by_diff + safety_promoting_by_diff}
        }
    }
    
    # Save for use in training
    selected_features = {
        "safety_promoting": safety_promoting_by_weight,  # Use weight-based as primary
        "safety_harming": safety_harming_by_weight,
        "method": "value_head_weights",
        "alternative_safety_harming": safety_harming_by_diff,
        "alternative_safety_promoting": safety_promoting_by_diff,
    }
    
    with open(output_dir / "feature_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / "selected_features.json", "w") as f:
        json.dump(selected_features, f, indent=2)
    
    # Also save to outputs/probes for compatibility
    probes_dir = Path("outputs/probes")
    probes_dir.mkdir(parents=True, exist_ok=True)
    with open(probes_dir / "selected_features.json", "w") as f:
        json.dump(selected_features, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"Also saved to outputs/probes/selected_features.json")
    
    # =========================================================================
    # Optional: GPT-4 Interpretation
    # =========================================================================
    if args.interpret and args.openai_key:
        print("\n" + "="*60)
        print("GPT-4 Feature Interpretation")
        print("="*60)
        
        client = OpenAI(api_key=args.openai_key)
        
        # Get high-activation contexts for top features
        # (simplified version - full version would use sae_get_contexts.py)
        
        interpretations = {}
        for feat in safety_harming_by_weight[:3]:
            weight = float(score_weights[feat])
            
            prompt = f"""Analyze this feature from a reward model's Sparse Autoencoder:

Feature ID: {feat}
Value head weight: {weight:.6f} (negative = safety-harming)

Based on SARM paper analysis, features with similar negative weights often correspond to:
- Violence and crime discussions
- Instructions for illegal/unethical actions
- Risk-taking and dangerous behaviors

What might this feature represent? Provide a brief 1-2 sentence interpretation."""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            
            interpretations[str(feat)] = {
                "weight": weight,
                "interpretation": response.choices[0].message.content.strip()
            }
            print(f"\nFeature {feat} (w={weight:.6f}):")
            print(f"  {interpretations[str(feat)]['interpretation']}")
        
        with open(output_dir / "feature_interpretations.json", "w") as f:
            json.dump(interpretations, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nFeatures to PENALIZE (safety-harming):")
    print(f"  By weight: {safety_harming_by_weight[:5]}")
    print(f"  By diff:   {safety_harming_by_diff[:5]}")
    print(f"\nFeatures that are GOOD (safety-promoting):")
    print(f"  By weight: {safety_promoting_by_weight[:5]}")
    print(f"  By diff:   {safety_promoting_by_diff[:5]}")
    print(f"\nUse these in training:")
    print(f"  --penalize_features \"{','.join(map(str, safety_harming_by_weight[:3]))}\"")


if __name__ == "__main__":
    main()


"""
Evaluate SARM on RewardBench-2 Math subset
Tests if SARM correctly prefers chosen over rejected responses
"""
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_eval_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs

def evaluate_pairs(model, tokenizer, pairs, clamp_features=None, device='cuda'):
    """Evaluate chosen vs rejected accuracy"""
    correct = 0
    total = 0
    
    # Hook for clamping
    if clamp_features:
        def clamp_hook(module, input, output):
            pre_acts = output + model.sae.latent_bias
            features = model.sae.get_latents(pre_acts)
            for f in clamp_features:
                features[:, :, f] = 0.0
            return features @ model.sae.decoder.weight.T
        hook_handle = model.sae.encoder.register_forward_hook(clamp_hook)
    
    for pair in tqdm(pairs, desc="Evaluating"):
        prompt = pair['prompt']
        chosen = pair['chosen']
        rejected = pair['rejected']
        
        # Score chosen
        messages_chosen = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': chosen}
        ]
        text_chosen = tokenizer.apply_chat_template(messages_chosen, tokenize=False)
        enc_chosen = tokenizer(text_chosen, return_tensors='pt', truncation=True, max_length=2048).to(device)
        
        # Score rejected
        messages_rejected = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': rejected}
        ]
        text_rejected = tokenizer.apply_chat_template(messages_rejected, tokenize=False)
        enc_rejected = tokenizer(text_rejected, return_tensors='pt', truncation=True, max_length=2048).to(device)
        
        with torch.no_grad():
            score_chosen = model(**enc_chosen).logits.item()
            score_rejected = model(**enc_rejected).logits.item()
        
        if score_chosen > score_rejected:
            correct += 1
        total += 1
    
    if clamp_features:
        hook_handle.remove()
    
    return correct / total if total > 0 else 0.0

def main():
    print("Loading SARM model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'Schrieffer/Llama-SARM-4B',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='cuda'
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained('Schrieffer/Llama-SARM-4B')
    
    # Load eval pairs
    math_pairs = load_eval_pairs('data/rewardbench2/math/eval_pairs.jsonl')
    print(f"Loaded {len(math_pairs)} math evaluation pairs")
    
    # Sample for faster evaluation
    import random
    random.seed(42)
    math_pairs_sample = random.sample(math_pairs, min(100, len(math_pairs)))
    
    print("\n" + "="*60)
    print("BASELINE (no clamping)")
    print("="*60)
    acc_baseline = evaluate_pairs(model, tokenizer, math_pairs_sample)
    print(f"Math Accuracy: {acc_baseline:.2%}")
    
    print("\n" + "="*60)
    print("WITH CLAMPING (bad math features → 0)")
    print("Features: 30565, 35797, 13030")
    print("="*60)
    acc_clamped = evaluate_pairs(model, tokenizer, math_pairs_sample, clamp_features=[30565, 35797, 13030])
    print(f"Math Accuracy: {acc_clamped:.2%}")
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Baseline Accuracy:  {acc_baseline:.2%}")
    print(f"Clamped Accuracy:   {acc_clamped:.2%}")
    print(f"Improvement:        {acc_clamped - acc_baseline:+.2%}")

if __name__ == "__main__":
    main()

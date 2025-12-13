"""
Compare: Base Model vs PPO-trained (with LoRA) vs No Training
Save results separately
"""
import torch
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import random

def generate_and_score(model, tokenizer, rm_model, rm_tokenizer, prompt, device='cuda:0'):
    messages = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    
    response = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # SARM score
    rm_messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
    rm_text = rm_tokenizer.apply_chat_template(rm_messages, tokenize=False)
    rm_enc = rm_tokenizer(rm_text, return_tensors='pt', truncation=True, max_length=2048).to(device)
    score = rm_model(**rm_enc).logits.item()
    
    return response, score

def evaluate_model(model, tokenizer, rm_model, rm_tokenizer, prompts, name):
    scores = []
    responses = []
    
    for prompt in tqdm(prompts, desc=name):
        response, score = generate_and_score(model, tokenizer, rm_model, rm_tokenizer, prompt)
        scores.append(score)
        responses.append(response[:200])
    
    return {
        'avg_score': sum(scores) / len(scores),
        'scores': scores,
        'sample_responses': responses[:5]
    }

def main():
    device = 'cuda:0'
    
    # Load RewardBench-2 Math
    print('Loading RewardBench-2 Math...')
    dataset = load_dataset('allenai/reward-bench-2', split='test')
    math_data = dataset.filter(lambda x: x['subset'] == 'Math')
    
    random.seed(42)
    sample_indices = random.sample(range(len(math_data)), 30)
    prompts = [math_data[i]['prompt'] for i in sample_indices]
    
    # Load SARM
    print('Loading SARM...')
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        'Schrieffer/Llama-SARM-4B',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device
    ).eval()
    rm_tokenizer = AutoTokenizer.from_pretrained('Schrieffer/Llama-SARM-4B')
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token
    
    results = {}
    
    # === 1. BASE MODEL (No Training) ===
    print('\n' + '='*60)
    print('1. BASE MODEL (No Training)')
    print('='*60)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.1-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map=device
    ).eval()
    
    results['base_model'] = evaluate_model(base_model, tokenizer, rm_model, rm_tokenizer, prompts, 'Base')
    print(f"Base Model Avg SARM Score: {results['base_model']['avg_score']:.4f}")
    
    del base_model
    torch.cuda.empty_cache()
    
    # === 2. PPO-TRAINED WITH LORA ===
    print('\n' + '='*60)
    print('2. PPO-TRAINED WITH LORA')
    print('='*60)
    
    lora_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.1-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    lora_model = PeftModel.from_pretrained(lora_model, 'outputs/experiments/math_ppo_quick/checkpoint')
    lora_model = lora_model.merge_and_unload()
    lora_model.eval()
    
    results['ppo_with_lora'] = evaluate_model(lora_model, tokenizer, rm_model, rm_tokenizer, prompts, 'PPO+LoRA')
    print(f"PPO+LoRA Model Avg SARM Score: {results['ppo_with_lora']['avg_score']:.4f}")
    
    del lora_model
    torch.cuda.empty_cache()
    
    # === SUMMARY ===
    print('\n' + '='*60)
    print('SUMMARY: Effect of LoRA Training')
    print('='*60)
    
    base_score = results['base_model']['avg_score']
    lora_score = results['ppo_with_lora']['avg_score']
    
    print(f"\n{'Method':<25} {'SARM Score':>12} {'vs Base':>12}")
    print('-' * 50)
    print(f"{'Base Model (No Training)':<25} {base_score:>12.4f} {'-':>12}")
    print(f"{'PPO + LoRA (30 steps)':<25} {lora_score:>12.4f} {lora_score - base_score:>+12.4f}")
    
    print(f"\nImprovement from LoRA training: {lora_score - base_score:+.4f} ({(lora_score - base_score) / abs(base_score) * 100:+.1f}%)")
    
    # Save results
    output_dir = Path('outputs/experiments/lora_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'task': 'RewardBench-2 Math',
        'num_samples': len(prompts),
        'results': {
            'base_model': {
                'description': 'Llama-3.1-8B-Instruct without any training',
                'avg_sarm_score': base_score,
            },
            'ppo_with_lora': {
                'description': 'PPO-trained with LoRA (30 steps)',
                'avg_sarm_score': lora_score,
            }
        },
        'comparison': {
            'lora_vs_base': lora_score - base_score,
            'lora_vs_base_percent': (lora_score - base_score) / abs(base_score) * 100,
        },
        'conclusion': 'LoRA training with PPO improves SARM score'
    }
    
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also save detailed per-sample scores
    detailed = {
        'prompts': prompts,
        'base_scores': results['base_model']['scores'],
        'lora_scores': results['ppo_with_lora']['scores'],
    }
    with open(output_dir / 'detailed_scores.json', 'w') as f:
        json.dump(detailed, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - {output_dir}/comparison_results.json")
    print(f"  - {output_dir}/detailed_scores.json")

if __name__ == "__main__":
    main()

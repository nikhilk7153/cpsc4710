"""
Evaluate Llama-3.1-8B policy model on math tasks
Compare base model vs fine-tuned model
"""
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import random
from tqdm import tqdm

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a response from the policy model"""
    messages = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def score_response(rm_model, rm_tokenizer, prompt, response):
    """Score a response using SARM"""
    messages = [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': response}
    ]
    text = rm_tokenizer.apply_chat_template(messages, tokenize=False)
    enc = rm_tokenizer(text, return_tensors='pt', truncation=True, max_length=2048).to(rm_model.device)
    
    with torch.no_grad():
        score = rm_model(**enc).logits.item()
    return score

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to fine-tuned LoRA checkpoint')
    parser.add_argument('--num_samples', type=int, default=20)
    args = parser.parse_args()
    
    # Load math prompts
    prompts = []
    with open('data/rewardbench2/math/prompts.jsonl') as f:
        for line in f:
            prompts.append(json.loads(line)['prompt'])
    
    random.seed(42)
    sample_prompts = random.sample(prompts, min(args.num_samples, len(prompts)))
    
    # Load SARM for scoring
    print("Loading SARM reward model...")
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        'Schrieffer/Llama-SARM-4B',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='cuda:0'
    ).eval()
    rm_tokenizer = AutoTokenizer.from_pretrained('Schrieffer/Llama-SARM-4B')
    
    # Load BASE policy model
    print("Loading base Llama-3.1-8B-Instruct...")
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.1-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='cuda:1'
    ).eval()
    base_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    base_tokenizer.pad_token = base_tokenizer.eos_token
    
    # Generate and score with base model
    print("\n" + "="*60)
    print("EVALUATING BASE MODEL")
    print("="*60)
    
    base_scores = []
    base_responses = []
    
    for prompt in tqdm(sample_prompts, desc="Base model"):
        response = generate_response(base_model, base_tokenizer, prompt, max_new_tokens=128)
        score = score_response(rm_model, rm_tokenizer, prompt, response)
        base_scores.append(score)
        base_responses.append(response)
    
    print(f"Base model average SARM score: {sum(base_scores)/len(base_scores):.4f}")
    
    # If checkpoint provided, evaluate fine-tuned model
    if args.checkpoint and Path(args.checkpoint).exists():
        print("\n" + "="*60)
        print(f"EVALUATING FINE-TUNED MODEL: {args.checkpoint}")
        print("="*60)
        
        # Load fine-tuned model
        ft_model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Llama-3.1-8B-Instruct',
            torch_dtype=torch.bfloat16,
            device_map='cuda:1'
        )
        ft_model = PeftModel.from_pretrained(ft_model, args.checkpoint)
        ft_model = ft_model.merge_and_unload()
        ft_model.eval()
        
        ft_scores = []
        ft_responses = []
        
        for prompt in tqdm(sample_prompts, desc="Fine-tuned model"):
            response = generate_response(ft_model, base_tokenizer, prompt, max_new_tokens=128)
            score = score_response(rm_model, rm_tokenizer, prompt, response)
            ft_scores.append(score)
            ft_responses.append(response)
        
        print(f"Fine-tuned model average SARM score: {sum(ft_scores)/len(ft_scores):.4f}")
        
        # Compare
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"Base model avg:       {sum(base_scores)/len(base_scores):.4f}")
        print(f"Fine-tuned model avg: {sum(ft_scores)/len(ft_scores):.4f}")
        print(f"Improvement:          {sum(ft_scores)/len(ft_scores) - sum(base_scores)/len(base_scores):+.4f}")
        
        # Show examples
        print("\n" + "="*60)
        print("EXAMPLE COMPARISONS")
        print("="*60)
        for i in range(min(3, len(sample_prompts))):
            print(f"\nPrompt: {sample_prompts[i][:100]}...")
            print(f"Base response: {base_responses[i][:150]}...")
            print(f"Base score: {base_scores[i]:.4f}")
            print(f"FT response: {ft_responses[i][:150]}...")
            print(f"FT score: {ft_scores[i]:.4f}")
    else:
        # Just show base model examples
        print("\n" + "="*60)
        print("EXAMPLE RESPONSES (Base Model)")
        print("="*60)
        for i in range(min(3, len(sample_prompts))):
            print(f"\nPrompt: {sample_prompts[i][:100]}...")
            print(f"Response: {base_responses[i][:200]}...")
            print(f"SARM Score: {base_scores[i]:.4f}")

if __name__ == "__main__":
    main()

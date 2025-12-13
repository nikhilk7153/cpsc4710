"""
PPO with FULL fine-tuning on a small model (Qwen2.5-1.5B)
Using the REAL RewardBench-2 Math dataset
Fixed: Re-tokenize responses with SARM tokenizer for scoring
"""
import torch
import json
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_math_prompts():
    """Load real math prompts from RewardBench-2"""
    prompts = []
    with open("data/rewardbench2/math/prompts.jsonl") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
    return prompts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sarm_model", default="Schrieffer/Llama-SARM-4B")
    parser.add_argument("--total_steps", type=int, default=100)
    parser.add_argument("--output_dir", default="outputs/experiments/math_full_ft")
    parser.add_argument("--lr", type=float, default=5e-6)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PPO WITH FULL FINE-TUNING (ALL PARAMETERS)")
    print("="*70)
    print(f"Policy Model: {args.policy_model}")
    print(f"Reward Model: {args.sarm_model}")
    print(f"Training Steps: {args.total_steps}")
    print("="*70)
    
    # Load real math prompts
    print("\nLoading RewardBench-2 Math prompts...")
    prompts = load_math_prompts()
    print(f"Loaded {len(prompts)} real math problems")
    
    # Sample some for display
    print("\nExample prompts:")
    for p in random.sample(prompts, min(3, len(prompts))):
        print(f"  - {p[:80]}...")
    
    # Load SARM reward model (on GPU 1)
    print("\nLoading SARM reward model...")
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.sarm_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:1"
    ).eval()
    rm_tokenizer = AutoTokenizer.from_pretrained(args.sarm_model)
    
    # Load policy model (FULL fine-tuning, on GPU 0)
    print(f"\nLoading {args.policy_model} for FULL fine-tuning...")
    policy = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Optimizer - AdamW for full fine-tuning
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    
    def score_with_sarm(prompt: str, response: str) -> float:
        """Score a response using SARM, with proper Llama tokenization"""
        # Use SARM's (Llama) tokenizer and chat template for scoring
        rm_messages = [
            {"role": "user", "content": prompt}, 
            {"role": "assistant", "content": response}
        ]
        rm_text = rm_tokenizer.apply_chat_template(rm_messages, tokenize=False)
        rm_enc = rm_tokenizer(rm_text, return_tensors='pt', truncation=True, max_length=512).to("cuda:1")
        with torch.no_grad():
            score = rm_model(**rm_enc).logits.item()
        return score
    
    # Baseline evaluation
    print("\n" + "="*70)
    print("BASELINE EVALUATION (before training)")
    print("="*70)
    
    eval_prompts = random.sample(prompts, min(20, len(prompts)))
    
    def evaluate(model, prompts_to_eval, desc="Evaluating"):
        model.eval()
        scores = []
        responses = []
        for prompt in tqdm(prompts_to_eval, desc=desc):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors='pt').to("cuda:0")
            
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Score with SARM (using SARM's tokenizer)
            score = score_with_sarm(prompt, response)
            
            scores.append(score)
            responses.append({"prompt": prompt, "response": response, "score": score})
        
        return sum(scores)/len(scores), responses
    
    baseline_score, baseline_responses = evaluate(policy, eval_prompts, "Baseline eval")
    print(f"\nBaseline SARM Score: {baseline_score:.4f}")
    print("\nSample responses:")
    for r in baseline_responses[:3]:
        print(f"  Q: {r['prompt'][:60]}...")
        print(f"  A: {r['response'][:100]}...")
        print(f"  Score: {r['score']:.4f}\n")
    
    # Training
    print("\n" + "="*70)
    print("TRAINING (FULL FINE-TUNING)")
    print("="*70)
    
    training_log = []
    
    for step in tqdm(range(args.total_steps), desc="PPO Steps"):
        prompt = random.choice(prompts)
        
        # Generate response using policy model
        policy.eval()
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors='pt').to("cuda:0")
        
        with torch.no_grad():
            out = policy.generate(
                **enc,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Get SARM reward (using SARM's tokenizer)
        reward = score_with_sarm(prompt, response)
        
        # Policy gradient update
        policy.train()
        optimizer.zero_grad()
        
        full_text = text + response
        enc = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512).to("cuda:0")
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = policy(**enc, labels=enc['input_ids'])
            # REINFORCE: multiply negative log-likelihood by negative reward
            # Higher reward -> lower loss
            loss = outputs.loss * (-reward + 1)  # +1 to keep loss positive when reward is positive
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        training_log.append({
            "step": step,
            "reward": reward,
            "loss": loss.item(),
        })
        
        if (step + 1) % 10 == 0:
            avg_reward = sum(l["reward"] for l in training_log[-10:]) / 10
            print(f"\nStep {step+1}: Avg Reward = {avg_reward:.4f}, Loss = {loss.item():.4f}")
            torch.cuda.empty_cache()
    
    # Post-training evaluation
    print("\n" + "="*70)
    print("POST-TRAINING EVALUATION")
    print("="*70)
    
    final_score, final_responses = evaluate(policy, eval_prompts, "Final eval")
    print(f"\nFinal SARM Score: {final_score:.4f}")
    print("\nSample responses:")
    for r in final_responses[:3]:
        print(f"  Q: {r['prompt'][:60]}...")
        print(f"  A: {r['response'][:100]}...")
        print(f"  Score: {r['score']:.4f}\n")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Model: {args.policy_model}")
    print(f"Training: FULL FINE-TUNING ({trainable_params:,} params)")
    print(f"Dataset: RewardBench-2 Math ({len(prompts)} problems)")
    print(f"Steps: {args.total_steps}")
    print(f"")
    print(f"Baseline SARM Score: {baseline_score:.4f}")
    print(f"Final SARM Score:    {final_score:.4f}")
    print(f"Improvement:         {final_score - baseline_score:+.4f} ({100*(final_score-baseline_score)/abs(baseline_score):+.1f}%)")
    
    # Save results
    results = {
        "model": args.policy_model,
        "method": "full_finetuning",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "dataset": "RewardBench-2 Math",
        "num_prompts": len(prompts),
        "steps": args.total_steps,
        "baseline_score": baseline_score,
        "final_score": final_score,
        "improvement": final_score - baseline_score,
        "training_log": training_log,
        "baseline_responses": baseline_responses,
        "final_responses": final_responses,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save model
    print(f"\nSaving model to {output_dir}/checkpoint...")
    policy.save_pretrained(output_dir / "checkpoint")
    tokenizer.save_pretrained(output_dir / "checkpoint")
    
    print(f"\nResults saved to {output_dir}/results.json")
    print("\nDONE!")

if __name__ == "__main__":
    main()

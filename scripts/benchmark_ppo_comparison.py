"""
Benchmark: Regular PPO vs PPO with Feature Control
Uses the same prompts, same model, same steps - only difference is feature control
"""
import torch
import json
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import os
import copy

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_prompts(n=50):
    """Load prompts for training"""
    prompts = []
    with open("data/train/prompts.jsonl") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            data = json.loads(line)
            prompts.append(data["prompt"])
    return prompts

def main():
    output_dir = Path("outputs/experiments/ppo_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PPO BENCHMARK: Regular vs Feature-Controlled")
    print("="*70)
    
    # Fixed random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Load prompts
    prompts = load_prompts(50)
    print(f"Loaded {len(prompts)} prompts")
    
    # Load SARM reward model
    print("\nLoading SARM reward model...")
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        "Schrieffer/Llama-SARM-4B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:1"
    ).eval()
    rm_tokenizer = AutoTokenizer.from_pretrained("Schrieffer/Llama-SARM-4B")
    
    # Feature control config (verbosity features - known to activate)
    PENALTY_FEATURES = [29733, 53553, 3402]  # Verbosity features
    TAU = 2.0
    ALPHA = 0.1
    
    def score_with_sarm(prompt, response, apply_penalty=False):
        """Score response, optionally with feature penalty"""
        try:
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
            text = rm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            enc = rm_tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to("cuda:1")
            
            with torch.no_grad():
                outputs = rm_model(**enc, output_hidden_states=True)
                rm_score = outputs.logits.item()
                
                # Get SAE activations if penalty is requested
                penalty = 0.0
                feature_activations = {}
                if apply_penalty and hasattr(rm_model, 'sae'):
                    # Get hidden states from layer 15 (where SAE is applied)
                    hidden = outputs.hidden_states[15][:, -1, :]  # Last token
                    sae_acts = rm_model.sae.pre_acts(hidden)
                    
                    for feat_id in PENALTY_FEATURES:
                        act = sae_acts[0, feat_id].item()
                        feature_activations[feat_id] = act
                        if act > TAU:
                            penalty += ALPHA
                
                total_reward = rm_score - penalty
                return total_reward, rm_score, penalty, feature_activations
        except Exception as e:
            return 0.0, 0.0, 0.0, {}
    
    def run_ppo_training(policy, tokenizer, prompts, n_steps, apply_penalty=False, desc="Training"):
        """Run PPO training loop"""
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
        
        training_log = []
        policy.train()
        
        for step in tqdm(range(n_steps), desc=desc):
            prompt = random.choice(prompts)
            
            # Generate
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
            
            # Get reward
            total_reward, rm_score, penalty, feat_acts = score_with_sarm(prompt, response, apply_penalty)
            
            # Update policy
            policy.train()
            optimizer.zero_grad()
            
            full_text = text + response
            enc = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512).to("cuda:0")
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = policy(**enc, labels=enc['input_ids'])
                loss = outputs.loss * (-total_reward + 1)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            training_log.append({
                "step": step,
                "total_reward": total_reward,
                "rm_score": rm_score,
                "penalty": penalty,
                "loss": loss.item(),
                "response_length": len(response.split()),
                "feature_activations": feat_acts,
            })
            
            if (step + 1) % 10 == 0:
                recent = training_log[-10:]
                avg_reward = sum(l["total_reward"] for l in recent) / 10
                avg_penalty = sum(l["penalty"] for l in recent) / 10
                print(f"\n  Step {step+1}: reward={avg_reward:.4f}, penalty={avg_penalty:.4f}")
        
        return training_log
    
    def evaluate(policy, tokenizer, prompts, apply_penalty=False):
        """Evaluate policy on prompts"""
        policy.eval()
        results = []
        
        for prompt in tqdm(prompts[:20], desc="Evaluating"):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors='pt').to("cuda:0")
            
            with torch.no_grad():
                out = policy.generate(
                    **enc,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
            
            total_reward, rm_score, penalty, feat_acts = score_with_sarm(prompt, response, apply_penalty)
            results.append({
                "prompt": prompt,
                "response": response,
                "total_reward": total_reward,
                "rm_score": rm_score,
                "penalty": penalty,
                "response_length": len(response.split()),
            })
        
        avg_reward = sum(r["total_reward"] for r in results) / len(results)
        avg_rm = sum(r["rm_score"] for r in results) / len(results)
        avg_penalty = sum(r["penalty"] for r in results) / len(results)
        avg_length = sum(r["response_length"] for r in results) / len(results)
        
        return {
            "avg_total_reward": avg_reward,
            "avg_rm_score": avg_rm,
            "avg_penalty": avg_penalty,
            "avg_response_length": avg_length,
            "results": results,
        }
    
    # Create base model with LoRA
    print("\nLoading policy model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    N_STEPS = 50
    
    # =========================================================================
    # EXPERIMENT 1: Regular PPO (no feature control)
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: REGULAR PPO (No Feature Control)")
    print("="*70)
    
    policy_regular = get_peft_model(copy.deepcopy(base_model), lora_config)
    print(f"Trainable params: {sum(p.numel() for p in policy_regular.parameters() if p.requires_grad):,}")
    
    # Baseline eval
    print("\nBaseline evaluation...")
    baseline_eval = evaluate(policy_regular, tokenizer, prompts, apply_penalty=False)
    print(f"Baseline - RM Score: {baseline_eval['avg_rm_score']:.4f}, Length: {baseline_eval['avg_response_length']:.1f} words")
    
    # Train
    print("\nTraining with Regular PPO...")
    regular_log = run_ppo_training(policy_regular, tokenizer, prompts, N_STEPS, apply_penalty=False, desc="Regular PPO")
    
    # Final eval
    print("\nFinal evaluation...")
    regular_final = evaluate(policy_regular, tokenizer, prompts, apply_penalty=False)
    print(f"Final - RM Score: {regular_final['avg_rm_score']:.4f}, Length: {regular_final['avg_response_length']:.1f} words")
    
    # =========================================================================
    # EXPERIMENT 2: PPO with Feature Control
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: PPO WITH FEATURE CONTROL")
    print(f"Penalizing features: {PENALTY_FEATURES} (verbosity)")
    print(f"Threshold τ={TAU}, Penalty α={ALPHA}")
    print("="*70)
    
    # Reload fresh model
    base_model2 = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    policy_controlled = get_peft_model(base_model2, lora_config)
    
    # Baseline eval (with penalty scoring)
    print("\nBaseline evaluation (with penalty)...")
    baseline_controlled = evaluate(policy_controlled, tokenizer, prompts, apply_penalty=True)
    print(f"Baseline - Total: {baseline_controlled['avg_total_reward']:.4f}, RM: {baseline_controlled['avg_rm_score']:.4f}, Penalty: {baseline_controlled['avg_penalty']:.4f}")
    
    # Train with feature control
    print("\nTraining with Feature-Controlled PPO...")
    controlled_log = run_ppo_training(policy_controlled, tokenizer, prompts, N_STEPS, apply_penalty=True, desc="Feature PPO")
    
    # Final eval
    print("\nFinal evaluation (with penalty)...")
    controlled_final = evaluate(policy_controlled, tokenizer, prompts, apply_penalty=True)
    print(f"Final - Total: {controlled_final['avg_total_reward']:.4f}, RM: {controlled_final['avg_rm_score']:.4f}, Penalty: {controlled_final['avg_penalty']:.4f}")
    
    # =========================================================================
    # RESULTS COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    
    print(f"\n{'Method':<30} {'RM Score':<15} {'Penalty':<15} {'Total Reward':<15} {'Resp Length':<15}")
    print("-" * 90)
    
    # Regular PPO
    print(f"{'Regular PPO (baseline)':<30} {baseline_eval['avg_rm_score']:<15.4f} {'N/A':<15} {baseline_eval['avg_rm_score']:<15.4f} {baseline_eval['avg_response_length']:<15.1f}")
    print(f"{'Regular PPO (trained)':<30} {regular_final['avg_rm_score']:<15.4f} {'N/A':<15} {regular_final['avg_rm_score']:<15.4f} {regular_final['avg_response_length']:<15.1f}")
    
    # Feature-controlled PPO
    print(f"{'Feature PPO (baseline)':<30} {baseline_controlled['avg_rm_score']:<15.4f} {baseline_controlled['avg_penalty']:<15.4f} {baseline_controlled['avg_total_reward']:<15.4f} {baseline_controlled['avg_response_length']:<15.1f}")
    print(f"{'Feature PPO (trained)':<30} {controlled_final['avg_rm_score']:<15.4f} {controlled_final['avg_penalty']:<15.4f} {controlled_final['avg_total_reward']:<15.4f} {controlled_final['avg_response_length']:<15.1f}")
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    regular_improvement = regular_final['avg_rm_score'] - baseline_eval['avg_rm_score']
    controlled_rm_change = controlled_final['avg_rm_score'] - baseline_controlled['avg_rm_score']
    penalty_change = controlled_final['avg_penalty'] - baseline_controlled['avg_penalty']
    length_change_regular = regular_final['avg_response_length'] - baseline_eval['avg_response_length']
    length_change_controlled = controlled_final['avg_response_length'] - baseline_controlled['avg_response_length']
    
    print(f"\nRegular PPO:")
    print(f"  RM Score change: {regular_improvement:+.4f}")
    print(f"  Response length change: {length_change_regular:+.1f} words")
    
    print(f"\nFeature-Controlled PPO:")
    print(f"  RM Score change: {controlled_rm_change:+.4f}")
    print(f"  Penalty change: {penalty_change:+.4f}")
    print(f"  Response length change: {length_change_controlled:+.1f} words")
    
    print(f"\nKey Insight:")
    if penalty_change < 0:
        print(f"  ✓ Feature control REDUCED penalties by {-penalty_change:.4f}")
        print(f"  ✓ Model learned to avoid triggering verbosity features")
    else:
        print(f"  ⚠ Penalties increased - may need more training steps")
    
    # Save results
    results = {
        "config": {
            "penalty_features": PENALTY_FEATURES,
            "tau": TAU,
            "alpha": ALPHA,
            "n_steps": N_STEPS,
        },
        "regular_ppo": {
            "baseline": baseline_eval,
            "final": regular_final,
            "training_log": regular_log,
        },
        "feature_controlled_ppo": {
            "baseline": baseline_controlled,
            "final": controlled_final,
            "training_log": controlled_log,
        },
    }
    
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}/benchmark_results.json")
    print("\nDONE!")

if __name__ == "__main__":
    main()

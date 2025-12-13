"""
PPO with FULL fine-tuning (no LoRA) - trains all parameters
"""
import torch
import json
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import torch.nn.functional as F

# Math prompts for training
TRAIN_PROMPTS = [
    "What is 15 + 27?",
    "Solve: 3x = 21",
    "What is 8 × 7?",
    "What is 144 ÷ 12?",
    "Solve: x + 5 = 12",
    "What is 25% of 80?",
    "What is the square root of 81?",
    "What is 6²?",
    "What is 100 ÷ 4?",
    "What is 13 × 4?",
    "What is 2³?",
    "Solve: x - 7 = 15",
    "What is 50% of 120?",
    "What is 11 × 11?",
    "What is 19 + 34?",
    "What is 72 ÷ 8?",
    "Solve: 4x = 36",
    "What is 7 × 9?",
    "What is 45 - 18?",
    "What is 10% of 250?",
]

def main():
    print("="*60)
    print("PPO WITH FULL FINE-TUNING (NO LORA)")
    print("Training ALL 8B parameters")
    print("="*60)
    
    device = "cuda:0"
    
    # Load SARM for rewards
    print("\nLoading SARM reward model...")
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        'Schrieffer/Llama-SARM-4B',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device
    ).eval()
    rm_tokenizer = AutoTokenizer.from_pretrained('Schrieffer/Llama-SARM-4B')
    
    # Load policy model - FULL model, no LoRA
    print("\nLoading policy model (FULL fine-tuning)...")
    policy = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.1-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} (100%)")
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Optimizer - lower learning rate for full fine-tuning
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)  # Lower LR for full FT
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING (Full Fine-tuning)")
    print("="*60)
    
    num_steps = 30
    batch_size = 1  # Smaller batch for memory
    
    rewards_history = []
    
    for step in tqdm(range(num_steps), desc="PPO Steps"):
        # Sample prompt
        prompt = random.choice(TRAIN_PROMPTS)
        
        # Generate response
        policy.eval()
        with torch.no_grad():
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors='pt').to(device)
            
            out = policy.generate(
                **enc,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            response = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Get SARM reward
            rm_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
            rm_text = rm_tokenizer.apply_chat_template(rm_messages, tokenize=False)
            rm_enc = rm_tokenizer(rm_text, return_tensors='pt', truncation=True, max_length=512).to(device)
            reward = rm_model(**rm_enc).logits.item()
        
        rewards_history.append(reward)
        
        # Update policy
        policy.train()
        optimizer.zero_grad()
        
        # Forward pass for loss
        full_text = text + response
        enc = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512).to(device)
        
        outputs = policy(**enc, labels=enc['input_ids'])
        
        # Scale loss by reward (REINFORCE)
        loss = outputs.loss * (-reward)  # Negative because we maximize reward
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            avg_reward = sum(rewards_history[-10:]) / 10
            print(f"\nStep {step+1}: Avg Reward (last 10) = {avg_reward:.4f}")
        
        # Clear cache periodically
        if step % 5 == 0:
            torch.cuda.empty_cache()
    
    # Save model
    print("\nSaving full fine-tuned model...")
    output_dir = Path("outputs/experiments/math_ppo_full")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir / "checkpoint")
    tokenizer.save_pretrained(output_dir / "checkpoint")
    
    # Save training rewards
    with open(output_dir / "training_rewards.json", 'w') as f:
        json.dump({"rewards": rewards_history}, f)
    
    print(f"\nModel saved to {output_dir}/checkpoint")
    print(f"Training rewards saved to {output_dir}/training_rewards.json")
    
    # Final evaluation
    print("\n" + "="*60)
    print("POST-TRAINING EVALUATION")
    print("="*60)
    
    policy.eval()
    eval_prompts = [
        "What is 15 + 27?",
        "Solve: 3x = 21",
        "What is 8 × 7?",
        "What is 100 ÷ 4?",
        "What is 6²?",
    ]
    
    eval_scores = []
    for prompt in eval_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            out = policy.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        
        response = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # SARM score
        rm_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        rm_text = rm_tokenizer.apply_chat_template(rm_messages, tokenize=False)
        rm_enc = rm_tokenizer(rm_text, return_tensors='pt', truncation=True, max_length=512).to(device)
        score = rm_model(**rm_enc).logits.item()
        eval_scores.append(score)
        
        print(f"Q: {prompt}")
        print(f"A: {response[:100]}...")
        print(f"Score: {score:.4f}\n")
    
    print(f"Average eval score: {sum(eval_scores)/len(eval_scores):.4f}")
    
    print("\n" + "="*60)
    print("FULL FINE-TUNING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()

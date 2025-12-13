"""
Quick PPO training on math problems with SARM rewards
Single GPU, simplified for speed
"""
import torch
import json
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import torch.nn.functional as F

# Simple math prompts for training
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
    print("QUICK PPO MATH TRAINING")
    print("="*60)
    
    device = "cuda:0"
    
    # Load SARM
    print("\nLoading SARM...")
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        'Schrieffer/Llama-SARM-4B',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device
    ).eval()
    rm_tokenizer = AutoTokenizer.from_pretrained('Schrieffer/Llama-SARM-4B')
    
    # Load policy with LoRA
    print("\nLoading policy model with LoRA...")
    policy = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.1-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    # Add LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    policy = get_peft_model(policy, lora_config)
    policy.print_trainable_parameters()
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    num_steps = 30
    batch_size = 2
    
    rewards_history = []
    
    for step in tqdm(range(num_steps), desc="PPO Steps"):
        # Sample prompts
        batch_prompts = random.sample(TRAIN_PROMPTS, batch_size)
        
        batch_rewards = []
        batch_log_probs = []
        batch_responses = []
        
        policy.eval()
        with torch.no_grad():
            for prompt in batch_prompts:
                # Generate response
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
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                
                response = tokenizer.decode(out.sequences[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
                batch_responses.append(response)
                
                # Get reward from SARM
                rm_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
                rm_text = rm_tokenizer.apply_chat_template(rm_messages, tokenize=False)
                rm_enc = rm_tokenizer(rm_text, return_tensors='pt', truncation=True, max_length=512).to(device)
                reward = rm_model(**rm_enc).logits.item()
                batch_rewards.append(reward)
        
        # Compute mean reward for this batch
        mean_reward = sum(batch_rewards) / len(batch_rewards)
        rewards_history.append(mean_reward)
        
        # Simple REINFORCE update
        policy.train()
        optimizer.zero_grad()
        
        total_loss = 0
        for prompt, response, reward in zip(batch_prompts, batch_responses, batch_rewards):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_text = text + response
            
            enc = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512).to(device)
            
            outputs = policy(**enc, labels=enc['input_ids'])
            
            # Scale by reward (advantage)
            advantage = reward - mean_reward
            loss = outputs.loss * (-advantage)  # Negative because we want to maximize reward
            total_loss += loss
        
        total_loss = total_loss / batch_size
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            print(f"\nStep {step+1}: Avg Reward = {mean_reward:.4f}, Loss = {total_loss.item():.4f}")
    
    # Save model
    print("\nSaving trained model...")
    output_dir = Path("outputs/experiments/math_ppo_quick")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir / "checkpoint")
    
    # Evaluate
    print("\n" + "="*60)
    print("POST-TRAINING EVALUATION")
    print("="*60)
    
    policy.eval()
    eval_prompts = [
        ("What is 15 + 27?", "42"),
        ("Solve: 3x = 21", "7"),
        ("What is 8 × 7?", "56"),
        ("What is 100 ÷ 4?", "25"),
        ("What is 6²?", "36"),
    ]
    
    for prompt, expected in eval_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            out = policy.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Get SARM score
        rm_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        rm_text = rm_tokenizer.apply_chat_template(rm_messages, tokenize=False)
        rm_enc = rm_tokenizer(rm_text, return_tensors='pt', truncation=True, max_length=512).to(device)
        score = rm_model(**rm_enc).logits.item()
        
        correct = expected in response
        status = "✓" if correct else "✗"
        print(f"{status} {prompt}")
        print(f"   Response: {response[:100]}...")
        print(f"   SARM Score: {score:.4f}")
    
    # Plot training curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, 'b-', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Average SARM Reward')
    plt.title('PPO Training on Math with SARM Rewards')
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/figures/ppo_math_training.png', dpi=150, bbox_inches='tight')
    print("\nSaved training curve to outputs/figures/ppo_math_training.png")
    
    # Save results
    with open(output_dir / "training_rewards.json", 'w') as f:
        json.dump({"rewards": rewards_history}, f)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()

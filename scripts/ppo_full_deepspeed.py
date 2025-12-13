"""
PPO with FULL fine-tuning using DeepSpeed ZeRO-3 for memory efficiency
"""
import torch
import json
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
]

def main():
    print("="*60)
    print("PPO WITH FULL FINE-TUNING")
    print("Using gradient checkpointing + SGD (lower memory)")
    print("="*60)
    
    device = "cuda:0"
    
    # Load SARM
    print("\nLoading SARM...")
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        'Schrieffer/Llama-SARM-4B',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:1"  # Put on different GPU
    ).eval()
    rm_tokenizer = AutoTokenizer.from_pretrained('Schrieffer/Llama-SARM-4B')
    
    # Load policy with gradient checkpointing
    print("\nLoading policy model with gradient checkpointing...")
    policy = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.1-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2"
    )
    policy.gradient_checkpointing_enable()
    
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Total parameters: {total_params:,} (ALL trainable)")
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Use SGD instead of Adam (less memory - no momentum states)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1e-5)
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    num_steps = 30
    rewards_history = []
    
    for step in tqdm(range(num_steps), desc="PPO Steps"):
        prompt = random.choice(TRAIN_PROMPTS)
        
        # Generate
        policy.eval()
        with torch.no_grad():
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors='pt').to(device)
            
            out = policy.generate(
                **enc,
                max_new_tokens=48,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
            )
            response = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Get reward
            rm_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
            rm_text = rm_tokenizer.apply_chat_template(rm_messages, tokenize=False)
            rm_enc = rm_tokenizer(rm_text, return_tensors='pt', truncation=True, max_length=256).to("cuda:1")
            reward = rm_model(**rm_enc).logits.item()
        
        rewards_history.append(reward)
        
        # Update
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        
        full_text = text + response
        enc = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=256).to(device)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = policy(**enc, labels=enc['input_ids'])
            loss = outputs.loss * (-reward)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        if step % 5 == 0:
            torch.cuda.empty_cache()
        
        if (step + 1) % 10 == 0:
            print(f"\nStep {step+1}: Avg Reward = {sum(rewards_history[-10:])/10:.4f}")
    
    # Save
    print("\nSaving model...")
    output_dir = Path("outputs/experiments/math_ppo_full")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir / "checkpoint")
    tokenizer.save_pretrained(output_dir / "checkpoint")
    
    with open(output_dir / "training_rewards.json", 'w') as f:
        json.dump({"rewards": rewards_history, "method": "full_finetuning_sgd"}, f)
    
    print(f"Saved to {output_dir}")
    
    # Quick eval
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    policy.eval()
    eval_scores = []
    for prompt in ["What is 15 + 27?", "What is 8 × 7?", "What is 6²?"]:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            out = policy.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        
        rm_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        rm_text = rm_tokenizer.apply_chat_template(rm_messages, tokenize=False)
        rm_enc = rm_tokenizer(rm_text, return_tensors='pt', truncation=True, max_length=256).to("cuda:1")
        score = rm_model(**rm_enc).logits.item()
        eval_scores.append(score)
        print(f"{prompt} -> {response[:60]}... (score: {score:.4f})")
    
    print(f"\nAvg eval score: {sum(eval_scores)/len(eval_scores):.4f}")
    print("\nDONE!")

if __name__ == "__main__":
    main()

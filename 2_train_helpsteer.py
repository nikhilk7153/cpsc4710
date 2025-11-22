import os
import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE
from peft import LoraConfig
from datasets import load_dataset
import numpy as np
from huggingface_hub import login, hf_hub_download
import json
import csv
import transformers

# --- PATCH FOR TRANSFORMERS COMPATIBILITY ---
# This fixes a potential import error in some TRL/Transformers version combinations
if not hasattr(transformers, "top_k_top_p_filtering"):
    def top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, filter_value)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, filter_value)
        return logits
    transformers.top_k_top_p_filtering = top_k_top_p_filtering

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# --- CONFIGURATION ---
# 1. EXPERIMENT CONTROL
RUN_TYPE = "CLAMPED" 
# RUN_TYPE = "CLAMPED"  # <-- UNCOMMENT THIS FOR SECOND RUN

# 2. FEATURE ID (From your Script 1 Output)
LENGTH_FEATURE_IDS = [1419, 12106, 6420, 7575, 10419]

# 3. MODEL & DATA
DATASET_ID = "nvidia/HelpSteer3"
# SCIENTIFIC FIX: Use Llama 3.1 to match the SAE exactly
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAE_RELEASE = "seonglae/Llama-3.1-8B-sae"
SAE_ID = "Llama-3.1-8B_blocks.24.hook_resid_pre_16384_topk_80_6e-05_42_fineweb_512"

# 4. AUTH
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "Set the HF_TOKEN environment variable with a valid Hugging Face token."
    )

print(f"--- STARTING RUN: {RUN_TYPE} ---")
print("Authenticating...")
login(token=HF_TOKEN)

# --- SETUP POLICY (The Student) ---
print("Loading Policy Model...")
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME, 
    peft_config=peft_config, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

# --- SETUP JUDGE (The Reward Model) ---
print("Loading Judge Model...")
reward_model = HookedTransformer.from_pretrained(
    MODEL_NAME, 
    device="cuda:0", 
    dtype=torch.bfloat16
)
reward_model.eval()

print("Loading SAE...")
# ROBUST LOADING: Handles both tuple (old) and object (new) returns from sae-lens
sae_out = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
if isinstance(sae_out, tuple):
    sae = sae_out[0]
else:
    sae = sae_out

# Get Hook Point
cfg_path = hf_hub_download(SAE_RELEASE, f"{SAE_ID}/cfg.json")
with open(cfg_path, "r") as f:
    cfg_json = json.load(f)
HOOK_POINT = cfg_json.get("hook_name") or cfg_json.get("hook_point")
if HOOK_POINT is None: raise ValueError("No hook point found")

# --- LOAD DATA ---
print("Loading Data...")
dataset = load_dataset(DATASET_ID, split="train")
prompts = []

def format_context(context):
    text = "<|begin_of_text|>"
    for turn in context:
        role = turn["role"]
        content = turn["content"]
        text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    return text

for row in dataset:
    formatted = format_context(row["context"]) + "<|start_header_id|>assistant<|end_header_id|>\n\n"
    prompts.append(formatted)

# --- MECHANISTIC REWARD FUNCTION ---
def get_mechanistic_reward(texts, clamp_active=False):
    rewards = []
    current_activation = 0.0
    
    def measurement_hook(resid_pre, hook):
        nonlocal current_activation
        encoded = sae.encode(resid_pre)

        activations = []
        ablation = torch.zeros_like(resid_pre)

        for feat_id in LENGTH_FEATURE_IDS:
            acts = encoded[:, :, feat_id]
            activations.append(acts.max().item())
            if clamp_active:
                feature_dir = sae.W_dec[feat_id]
                ablation += acts.unsqueeze(-1) * feature_dir.unsqueeze(0).unsqueeze(0)

        if clamp_active:
            resid_after = resid_pre - ablation
            encoded_after = sae.encode(resid_after)
            current_activation = max(encoded_after[:, :, fid].max().item() for fid in LENGTH_FEATURE_IDS)
            return resid_after

        current_activation = max(activations) if activations else 0.0
        return resid_pre

    for text in texts:
        with torch.no_grad():
            with reward_model.hooks(fwd_hooks=[(HOOK_POINT, measurement_hook)]):
                try:
                    # Run forward pass to trigger hook
                    reward_model(text, return_type=None)
                except:
                    pass
        
        # REWARD LOGIC:
        # We reward the model for triggering the Length Feature.
        # If Clamped, the feature is removed -> Activation is 0 -> Reward is 0.
        # If Baseline, feature fires -> Reward is high -> Model learns to yap.
        val = min(current_activation * 2.0, 10.0)
        rewards.append(torch.tensor(val))
        
    return rewards

# --- TRAINING LOOP ---
# CRITICAL FIX: Removed 'model_name' argument which caused the TypeError
config = PPOConfig(
    learning_rate=1.4e-5,
    batch_size=4,
    mini_batch_size=2,
)
ppo_trainer = PPOTrainer(config, model, tokenizer=tokenizer)

avg_lengths = []
avg_rewards = []

print(f"Starting Training Loop ({RUN_TYPE})...")
NUM_STEPS = 30

for step in range(NUM_STEPS):
    # Batching
    batch_idx = (step * config.batch_size) % len(prompts)
    current_batch = prompts[batch_idx : batch_idx + config.batch_size]
    if len(current_batch) < config.batch_size:
        current_batch = prompts[: config.batch_size]
    
    query_tensors = [tokenizer(p, return_tensors="pt").input_ids[0].cuda() for p in current_batch]
    
    # 1. Generate
    response_tensors = ppo_trainer.generate(
        query_tensors, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id
    )
    
    # FIX: Ensure response_tensors is a list
    if isinstance(response_tensors, torch.Tensor):
        response_tensors_list = [r for r in response_tensors]
    else:
        response_tensors_list = response_tensors
        
    batch_responses = [tokenizer.decode(r.squeeze()) for r in response_tensors_list]
    
    # 2. Reward
    should_clamp = (RUN_TYPE == "CLAMPED")
    rewards = get_mechanistic_reward(batch_responses, clamp_active=should_clamp)
    
    # 3. Step
    ppo_trainer.step(query_tensors, response_tensors_list, rewards)
    
    # 4. Log
    # Measure new tokens only
    lengths = [len(r) - len(p) for r, p in zip(batch_responses, current_batch)]
    curr_len = sum(lengths) / len(lengths)
    curr_rew = torch.stack(rewards).mean().item()
    
    avg_lengths.append(curr_len)
    avg_rewards.append(curr_rew)
    print(f"Step {step} | Avg Len: {curr_len:.1f} | Reward: {curr_rew:.3f}")
    
    # Cleanup to prevent OOM
    del query_tensors, response_tensors, rewards
    torch.cuda.empty_cache()

# Save
filename = f"{RUN_TYPE}_results.csv"
with open(filename, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "avg_len", "avg_reward"])
    for i, (l, r) in enumerate(zip(avg_lengths, avg_rewards)):
        writer.writerow([i, l, r])

print(f"Done! Saved to {filename}")
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
import numpy as np
import json
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
# Use the TransformerLens-compatible Llama‑3 (3.0) checkpoint + matching SAE
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_ID = "nvidia/HelpSteer3"

SAE_RELEASE = "EleutherAI/sae-llama-3-8b-32x-v2"
SAE_ID = "layers.24"

print(f"Loading Model: {MODEL_ID}...")
# Use bfloat16 for H200 stability
model = HookedTransformer.from_pretrained(MODEL_ID, device="cuda", dtype=torch.bfloat16)

print(f"Loading SAE: {SAE_RELEASE} / {SAE_ID}...")
sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")

# Dynamic Hook Point Loading (Your logic here was good!)
cfg_path = hf_hub_download(SAE_RELEASE, f"{SAE_ID}/cfg.json")
with open(cfg_path, "r") as f:
    cfg_json = json.load(f)
HOOK_POINT = cfg_json.get("hook_name") or cfg_json.get("hook_point")

if HOOK_POINT is None:
    raise ValueError("Could not determine hook point from SAE config.")
print(f"Hooking into: {HOOK_POINT}")

# --- LOAD DATA ---
print(f"Loading real human preferences from {DATASET_ID}...")
ds = load_dataset(DATASET_ID, split="train[:10000]") 

probe_data = []
token_counts = []

def format_context(context):
    """Convert HelpSteer3 context turns into Llama-3 chat format."""
    text = "<|begin_of_text|>"
    for turn in context:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            text += "<|start_header_id|>user<|end_header_id|>\n\n"
        else:
            text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        text += f"{content}<|eot_id|>"
    return text

def pick_preferred_response(row):
    # In HelpSteer, negative preference favors response1.
    return "response1" if row["overall_preference"] <= 0 else "response2"

print("Preprocessing dataset...")
for row in ds:
    base = format_context(row["context"])
    best_resp_key = pick_preferred_response(row)
    full_text = (
        base
        + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        + f"{row[best_resp_key]}<|eot_id|>"
    )
    probe_data.append(full_text)
    tokens = model.to_tokens(full_text)[0]
    token_counts.append(len(tokens))

print(f"Scanning {len(probe_data)} real samples...")

# --- THE HUNT ---
feature_activations = []

def capture_hook(resid_pre, hook):
    encoded = sae.encode(resid_pre)
    # Max usually captures the "Existence" of length better than Mean
    feature_activations.append(encoded.max(dim=1).values.detach().cpu())

# Process 1 by 1 to avoid padding bugs
BATCH_SIZE = 1 

with torch.no_grad():
    # Use the hook name from the loaded SAE config
    with model.hooks(fwd_hooks=[(HOOK_POINT, capture_hook)]):
        for i in range(0, len(probe_data), BATCH_SIZE):
            batch_texts = probe_data[i:i+BATCH_SIZE]
            model(batch_texts)
            if i % 100 == 0: print(f"Processed {i}...")

# Concatenate
all_acts = torch.cat(feature_activations, dim=0) 

# --- CORRELATION ---
print("Calculating correlations...")
token_counts_tensor = torch.tensor(token_counts, dtype=torch.float32)
tc_norm = token_counts_tensor - token_counts_tensor.mean()

correlations = []
for i in range(all_acts.shape[1]):
    feat_acts = all_acts[:, i]
    if feat_acts.sum() == 0: continue 
    
    fa_norm = feat_acts - feat_acts.mean()
    numerator = (fa_norm * tc_norm).sum()
    denominator = torch.sqrt((fa_norm**2).sum()) * torch.sqrt((tc_norm**2).sum())
    
    if denominator == 0: continue
    corr = numerator / denominator
    correlations.append((i, corr.item()))

correlations.sort(key=lambda x: x[1], reverse=True)

print("\n--- TOP CANDIDATE FEATURES ---")
for idx, corr in correlations[:10]:
    print(f"Feature ID: {idx} | Correlation: {corr:.4f}")
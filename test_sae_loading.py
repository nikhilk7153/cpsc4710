#!/usr/bin/env python3
"""Quick test to verify SAE loading works (runs on CPU)"""

import os
from sae_lens import SAE
from huggingface_hub import login
import torch

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "Set the HF_TOKEN environment variable before running this script."
    )
SAE_RELEASE = "EleutherAI/sae-llama-3-8b-32x-v2"
SAE_ID = "layers.12"

print("Authenticating with HuggingFace...")
login(token=HF_TOKEN)

print(f"\nTesting SAE loading from {SAE_RELEASE}/{SAE_ID}...")
print("(This will download ~100MB on first run)")

try:
    # Load on CPU to avoid needing GPU
    sae, _, _ = SAE.from_pretrained(
        release=SAE_RELEASE, 
        sae_id=SAE_ID, 
        device="cpu"
    )
    print("✓ SAE loaded successfully!")
    print(f"  - SAE config: {sae.cfg.d_in} -> {sae.cfg.d_sae}")
cfg_dict = getattr(sae.cfg, "__dict__", {})
hook_point = None
for candidate in (
    getattr(sae.cfg, "hook_point", None),
    getattr(sae.cfg, "hook_name", None),
    cfg_dict.get("hook_point"),
    cfg_dict.get("hook_name"),
):
    if candidate:
        hook_point = candidate
        break
if hook_point is None:
    raise ValueError("Could not determine hook point from SAE config.")
    print(f"  - Hook point: {hook_point}")
    print(f"\n✅ Your scripts should work on GPU!")
    
except Exception as e:
    print(f"✗ Failed to load SAE!")
    print(f"  Error: {e}")
    print(f"\n❌ Need to fix the SAE configuration before GPU run")


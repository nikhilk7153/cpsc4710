\
"""
build_human_latents.py

Builds a "human latent buffer" by running a frozen SARM model over a set of (prompt, response) demonstrations,
extracting the pooled SAE latents z, and saving (optionally) only selected latent indices.

Input JSONL format (one per line):
  {"prompt": "...", "response": "..."}

Output:
  torch.save({"latent_indices": [...], "z": Tensor[N, d]}, out_path)

If you omit --latent_indices, this script will attempt to save the full 65,536-D dense z vectors, which is usually
impractical for large N.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from sarm_wrapper import SARMRewardModel


def parse_indices(s: Optional[str]) -> Optional[List[int]]:
    if s is None or s.strip() == "":
        return None
    return [int(x) for x in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos_jsonl", type=str, required=True, help="JSONL with prompt/response.")
    ap.add_argument("--out", type=str, required=True, help="Output .pt path.")
    ap.add_argument("--model_name", type=str, default="Schrieffer/Llama-SARM-4B")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--latent_indices", type=str, default=None, help="Comma-separated indices to store.")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    rm = SARMRewardModel(model_name=args.model_name, torch_dtype=dtype_map[args.dtype])

    indices = parse_indices(args.latent_indices)

    # Read demos
    demos = []
    with open(args.demos_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            demos.append((obj["prompt"], obj["response"]))

    z_chunks = []
    for i in tqdm(range(0, len(demos), args.batch_size), desc="Extracting human latents"):
        batch = demos[i : i + args.batch_size]
        prompts = [p for p, _ in batch]
        responses = [r for _, r in batch]

        z = rm.pooled_latents(prompts, responses)  # [B, latent]
        z = z.detach().to(torch.float32).cpu()     # store fp32 for stability

        if indices is not None:
            z = z[:, indices]

        z_chunks.append(z)

    z_all = torch.cat(z_chunks, dim=0)

    out = {"latent_indices": indices, "z": z_all}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"Saved human latents: z shape {tuple(z_all.shape)} -> {args.out}")


if __name__ == "__main__":
    main()

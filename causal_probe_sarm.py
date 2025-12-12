\
"""
causal_probe_sarm.py

Phase-2 causal clamping probes on SARM latents.

Two modes:
1) Single-feature intervention on an existing (prompt,response) dataset:
   - compute baseline reward R
   - set or clamp z_i at pooled token
   - compute intervened reward R'
   - output CSV of deltas

2) Best-of-N reranking test:
   - for each prompt, score N candidate responses (baseline)
   - apply intervention and rescore
   - compare which candidate is selected under baseline vs intervention

Input JSONL for mode (1):
  {"prompt": "...", "response": "..."}

Input JSONL for mode (2):
  {"prompt": "...", "candidates": ["...", "...", ...]}

This directly implements the proposal's Phase-2 'causal clamping' tests.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from sarm_wrapper import ClampSpec, SARMRewardModel


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="probe_results.csv")
    ap.add_argument("--model_name", type=str, default="Schrieffer/Llama-SARM-4B")
    ap.add_argument("--batch_size", type=int, default=2)

    ap.add_argument("--feature", type=int, required=True, help="Latent feature index i to intervene on.")
    ap.add_argument("--set_value", type=float, default=None, help="Set z_i to this value.")
    ap.add_argument("--min_value", type=float, default=None, help="Clamp lower bound for z_i.")
    ap.add_argument("--max_value", type=float, default=None, help="Clamp upper bound for z_i.")

    ap.add_argument("--mode", type=str, default="pairs", choices=["pairs", "bestofn"])
    ap.add_argument("--top_k", type=int, default=1, help="For bestofn: which rank to select (1=best).")
    args = ap.parse_args()

    rm = SARMRewardModel(model_name=args.model_name)

    clamp = {
        args.feature: ClampSpec(min_val=args.min_value, max_val=args.max_value, set_val=args.set_value)
    }

    rows = load_jsonl(args.data_jsonl)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "pairs":
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["idx", "prompt", "response", "reward_base", "reward_intervene", "delta"],
            )
            writer.writeheader()

            for i in tqdm(range(0, len(rows), args.batch_size), desc="Probing (pairs)"):
                batch = rows[i : i + args.batch_size]
                prompts = [x["prompt"] for x in batch]
                responses = [x["response"] for x in batch]

                r0 = rm.score_batch(prompts, responses)  # baseline
                r1 = rm.score_batch(prompts, responses, clamp=clamp)

                for j, ex in enumerate(batch):
                    writer.writerow(
                        {
                            "idx": i + j,
                            "prompt": ex["prompt"],
                            "response": ex["response"],
                            "reward_base": float(r0[j].item()),
                            "reward_intervene": float(r1[j].item()),
                            "delta": float((r1[j] - r0[j]).item()),
                        }
                    )

        print(f"Wrote {args.out_csv}")
        return

    # best-of-n mode
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "prompt",
                "n_candidates",
                "selected_base",
                "selected_intervene",
                "changed",
                "reward_selected_base",
                "reward_selected_intervene",
            ],
        )
        writer.writeheader()

        for i, ex in enumerate(tqdm(rows, desc="Probing (bestofn)")):
            prompt = ex["prompt"]
            cands = ex["candidates"]
            n = len(cands)
            prompts = [prompt] * n

            r_base = rm.score_batch(prompts, cands)
            r_int = rm.score_batch(prompts, cands, clamp=clamp)

            # select top_k (1=best)
            k = max(1, args.top_k)
            base_rank = torch.argsort(r_base, descending=True)
            int_rank = torch.argsort(r_int, descending=True)

            sel_base = int(base_rank[k - 1].item())
            sel_int = int(int_rank[k - 1].item())

            writer.writerow(
                {
                    "idx": i,
                    "prompt": prompt,
                    "n_candidates": n,
                    "selected_base": sel_base,
                    "selected_intervene": sel_int,
                    "changed": int(sel_base != sel_int),
                    "reward_selected_base": float(r_base[sel_base].item()),
                    "reward_selected_intervene": float(r_int[sel_int].item()),
                }
            )

    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()

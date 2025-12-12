\
"""
ppo_feature_control_sarm.py

Phase-3 PPO training loop that uses:
- a policy model (LM + value head) trained with TRL PPOTrainer
- a frozen SARM reward model to compute reward
- in-loop feature controls on SARM latents + scalar reward shaping per your proposal

Reward used for PPO (per sample):
  R_total = R_sarm(z_clamped) - alpha * 1[z_raw_i > tau] - beta * log_ratio(z_raw_subset)

Where:
- z_raw is the pooled SARM latent vector for the (prompt,response)
- z_clamped applies an in-loop intervention before scoring (optional)
- log_ratio is produced by a density-ratio discriminator trained on (human vs policy) latents

Input prompts JSONL:
  {"prompt": "..."} per line

This is a minimal research script; adapt generation params + batch sizes for your infra.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from sarm_wrapper import ClampSpec, SARMRewardModel
from density_ratio import DensityRatioTrainer


def load_prompts_jsonl(path: str) -> List[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    return prompts


def parse_indices(s: Optional[str]) -> Optional[List[int]]:
    if s is None or s.strip() == "":
        return None
    return [int(x) for x in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--policy_model", type=str, required=True, help="SFT/policy model name or path.")
    ap.add_argument("--prompts_jsonl", type=str, required=True, help="JSONL with prompts.")
    ap.add_argument("--output_dir", type=str, default="ppo_out")

    # SARM RM
    ap.add_argument("--sarm_model", type=str, default="Schrieffer/Llama-SARM-4B")

    # PPO hyperparams (keep minimal)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--mini_batch_size", type=int, default=4)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--total_steps", type=int, default=200)

    # Generation params
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)

    # Feature penalty (indicator)
    ap.add_argument("--penalize_feature", type=int, default=None)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.0)

    # In-loop clamp (optional)
    ap.add_argument("--clamp_feature", type=int, default=None)
    ap.add_argument("--clamp_set", type=float, default=None)
    ap.add_argument("--clamp_min", type=float, default=None)
    ap.add_argument("--clamp_max", type=float, default=None)

    # Density-ratio KL penalty
    ap.add_argument("--human_latents", type=str, default=None, help="Path produced by build_human_latents.py")
    ap.add_argument("--beta", type=float, default=0.0, help="Coefficient for density-ratio KL penalty.")
    ap.add_argument("--disc_lr", type=float, default=1e-4)
    ap.add_argument("--disc_steps_per_batch", type=int, default=1)
    ap.add_argument("--latent_indices", type=str, default=None, help="Comma-separated indices for discriminator.")
    args = ap.parse_args()

    # ------------------------
    # TRL imports (compat)
    # ------------------------
    try:
        # new-ish
        from trl.experimental.ppo import PPOConfig, PPOTrainer
    except Exception:
        from trl import PPOConfig, PPOTrainer

    try:
        from trl import AutoModelForCausalLMWithValueHead
    except Exception:
        from trl.models import AutoModelForCausalLMWithValueHead

    # ------------------------
    # Load policy + tokenizer
    # ------------------------
    policy_tok = AutoTokenizer.from_pretrained(args.policy_model)
    if policy_tok.pad_token_id is None:
        policy_tok.pad_token = policy_tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.policy_model).to(device)

    # ------------------------
    # Load SARM RM (frozen)
    # ------------------------
    rm = SARMRewardModel(model_name=args.sarm_model)

    # ------------------------
    # Dataset
    # ------------------------
    prompts = load_prompts_jsonl(args.prompts_jsonl)
    ds = Dataset.from_dict({"query": prompts})

    def tokenize(ex):
        ex["input_ids"] = policy_tok.encode(ex["query"])
        return ex

    ds = ds.map(tokenize, batched=False)

    # ------------------------
    # PPO Trainer
    # ------------------------
    config = PPOConfig(
        model_name=args.policy_model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
    )

    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        dataset=ds,
        tokenizer=policy_tok,
    )

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "pad_token_id": policy_tok.eos_token_id,
    }

    # ------------------------
    # Controls
    # ------------------------
    clamp: Optional[Dict[int, ClampSpec]] = None
    if args.clamp_feature is not None:
        clamp = {
            int(args.clamp_feature): ClampSpec(
                min_val=args.clamp_min,
                max_val=args.clamp_max,
                set_val=args.clamp_set,
            )
        }

    latent_indices = parse_indices(args.latent_indices)

    # Load human latent buffer (optional)
    disc_trainer: Optional[DensityRatioTrainer] = None
    human_z: Optional[torch.Tensor] = None
    if args.beta and args.beta != 0.0:
        if args.human_latents is None:
            raise ValueError("--beta != 0 requires --human_latents.")

        human_obj = torch.load(args.human_latents, map_location="cpu")
        human_z = human_obj["z"].to(torch.float32)  # [N, d]
        stored_idx = human_obj.get("latent_indices", None)

        if latent_indices is None:
            if stored_idx is None:
                # using full z is likely too big; but allow if provided
                latent_indices = None
                disc_dim = human_z.shape[1]
            else:
                latent_indices = list(stored_idx)
                disc_dim = human_z.shape[1]
        else:
            disc_dim = len(latent_indices)
            # if stored already subset but indices mismatch, we re-index from full (best-effort)
            if stored_idx is not None and list(stored_idx) != list(latent_indices):
                raise ValueError(
                    "human_latents.pt was built with latent_indices that do not match --latent_indices. "
                    "Rebuild human_latents.pt or pass matching --latent_indices."
                )

        disc_trainer = DensityRatioTrainer(dim=disc_dim, lr=args.disc_lr)

    # ------------------------
    # Training loop
    # ------------------------
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader), total=args.total_steps, desc="PPO"):
        if step >= args.total_steps:
            break

        query_tensors = batch["input_ids"]
        # TRL expects list[torch.LongTensor]
        response_tensors = ppo_trainer.generate(query_tensors, **gen_kwargs)

        # Decode prompt/response text (careful: response_tensors include prompt+response)
        prompts_txt = []
        responses_txt = []
        for q, r in zip(query_tensors, response_tensors):
            q = q.squeeze(0) if q.ndim == 2 else q
            r = r.squeeze(0) if r.ndim == 2 else r
            gen = r[len(q):]
            prompts_txt.append(policy_tok.decode(q, skip_special_tokens=True))
            responses_txt.append(policy_tok.decode(gen, skip_special_tokens=True))

        # Extract raw pooled latents once
        with torch.no_grad():
            z_raw = rm.pooled_latents(prompts_txt, responses_txt)  # [B, latent]
            # Reward model score on *clamped* latents if clamp specified
            if clamp:
                # Avoid re-running SARM forward: clamp locally then score
                z_clamped = rm._apply_clamp(z_raw, clamp)  # type: ignore[attr-defined]
                rm_score = rm.score_layer(z_clamped).squeeze(-1)  # [B]
            else:
                rm_score = rm.score_layer(z_raw).squeeze(-1)

            # Indicator penalty
            ind_pen = torch.zeros_like(rm_score)
            if args.penalize_feature is not None and args.alpha != 0.0:
                idx = int(args.penalize_feature)
                ind_pen = args.alpha * (z_raw[:, idx] > args.tau).to(rm_score.dtype)

            # Density-ratio KL penalty
            kl_pen = torch.zeros_like(rm_score)
            if disc_trainer is not None and human_z is not None and args.beta != 0.0:
                if latent_indices is None:
                    z_sub = z_raw.to(torch.float32)
                else:
                    z_sub = z_raw[:, latent_indices].to(torch.float32)

                # sample equal-size human batch
                N = human_z.shape[0]
                idxs = torch.randint(0, N, (z_sub.shape[0],))
                z_h = human_z[idxs]

                # update discriminator
                for _ in range(max(1, args.disc_steps_per_batch)):
                    disc_trainer.train_step(z_h, z_sub)

                log_ratio = disc_trainer.log_ratio(z_sub).detach().to(rm_score.dtype)
                kl_pen = args.beta * log_ratio

            r_total = rm_score - ind_pen - kl_pen

        rewards = [rt.detach() for rt in r_total]  # list[tensor scalar]
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # Minimal logging
        stats["rm/score_mean"] = rm_score.mean().item()
        stats["rm/ind_pen_mean"] = ind_pen.mean().item()
        stats["rm/kl_pen_mean"] = kl_pen.mean().item()
        stats["rm/total_mean"] = r_total.mean().item()
        ppo_trainer.log_stats(stats, batch, rewards)

    ppo_trainer.save_model(args.output_dir)
    print(f"Saved PPO model -> {args.output_dir}")


if __name__ == "__main__":
    main()

"""
PPO Training with SARM Feature Controls

Uses LoRA for efficient fine-tuning + SARM reward model with feature-level controls.

Reward function:
  R_total = R_sarm(z) - sum_i(alpha_i * 1[z_i > tau_i]) - beta * KL_penalty

Usage:
  accelerate launch --config_file configs/accelerate_4gpu.yaml ppo_feature_control_sarm.py \
      --policy_model meta-llama/Llama-3.1-8B-Instruct \
      --prompts_jsonl data/train/prompts.jsonl ...
"""

import argparse
import csv
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Suppress noisy warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", message=".*right-padding.*")
warnings.filterwarnings("ignore", message=".*std\\(\\).*")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, TaskType

from sarm_wrapper import SARMRewardModel
from density_ratio import DensityRatioTrainer


@dataclass
class TrainingMetrics:
    """Comprehensive metrics for each training step."""
    step: int = 0
    timestamp: float = 0.0
    
    # Rewards
    total_reward: float = 0.0
    rm_score: float = 0.0
    indicator_penalty: float = 0.0
    kl_penalty: float = 0.0
    
    # Per-feature statistics
    feature_activations: Dict[int, float] = field(default_factory=dict)
    feature_above_tau: Dict[int, float] = field(default_factory=dict)
    
    # Response statistics
    response_length_mean: float = 0.0
    response_length_std: float = 0.0
    response_length_max: int = 0
    response_length_min: int = 0
    
    # Training dynamics
    ppo_loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    
    # Reward distribution
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0
    
    # Advantage statistics
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    
    # GPU memory
    gpu_memory_gb: float = 0.0


class MetricsLogger:
    """Logs training metrics to CSV and JSON."""
    
    def __init__(self, output_dir: Path, experiment_name: str = "default"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.metrics_history: List[Dict] = []
        
        # CSV file
        self.csv_path = self.output_dir / f"{experiment_name}_metrics.csv"
        self.csv_file = None
        self.csv_writer = None
    
    def log(self, metrics: TrainingMetrics):
        """Log metrics to history and CSV."""
        metrics_dict = {
            'step': metrics.step,
            'timestamp': metrics.timestamp,
            'total_reward': metrics.total_reward,
            'rm_score': metrics.rm_score,
            'indicator_penalty': metrics.indicator_penalty,
            'kl_penalty': metrics.kl_penalty,
            'response_length_mean': metrics.response_length_mean,
            'response_length_std': metrics.response_length_std,
            'ppo_loss': metrics.ppo_loss,
            'policy_loss': metrics.policy_loss,
            'value_loss': metrics.value_loss,
            'entropy': metrics.entropy,
            'approx_kl': metrics.approx_kl,
            'clip_fraction': metrics.clip_fraction,
            'reward_mean': metrics.reward_mean,
            'reward_std': metrics.reward_std,
            'reward_min': metrics.reward_min,
            'reward_max': metrics.reward_max,
            'advantage_mean': metrics.advantage_mean,
            'advantage_std': metrics.advantage_std,
            'gpu_memory_gb': metrics.gpu_memory_gb,
        }
        
        # Add per-feature stats
        for feat_id, val in metrics.feature_activations.items():
            metrics_dict[f'feat_{feat_id}_activation'] = val
        for feat_id, val in metrics.feature_above_tau.items():
            metrics_dict[f'feat_{feat_id}_above_tau'] = val
        
        self.metrics_history.append(metrics_dict)
        
        # Write to CSV
        if self.csv_writer is None:
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=list(metrics_dict.keys()))
            self.csv_writer.writeheader()
        
        self.csv_writer.writerow(metrics_dict)
        self.csv_file.flush()
    
    def save_summary(self):
        """Save summary JSON."""
        summary = {
            'experiment_name': self.experiment_name,
            'total_steps': len(self.metrics_history),
            'metrics_path': str(self.csv_path),
            'final_metrics': self.metrics_history[-1] if self.metrics_history else {},
        }
        
        with open(self.output_dir / f"{self.experiment_name}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.csv_file:
            self.csv_file.close()


@dataclass
class ClampSpec:
    """Specification for clamping a feature."""
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    set_val: Optional[float] = None


class ValueHead(nn.Module):
    """Value head for PPO - predicts state values."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states[:, -1, :]  # Last token
        x = torch.tanh(self.dense(x))
        return self.value(x).squeeze(-1)


class PolicyWithValueHead(nn.Module):
    """Wraps a causal LM with a value head for PPO."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.value_head = ValueHead(base_model.config.hidden_size)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        values = self.value_head(outputs.hidden_states[-1])
        return outputs.logits, values
    
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)


class PromptDataset(Dataset):
    """Dataset of prompts for PPO training."""
    
    def __init__(self, prompts: List[str], tokenizer, max_length: int = 512):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "prompt": prompt,
        }


def collate_fn(batch, tokenizer):
    """Left-pad sequences for decoder-only models."""
    input_ids = [torch.tensor(b["input_ids"]) for b in batch]
    attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]
    prompts = [b["prompt"] for b in batch]
    
    # Reverse, pad right, reverse back = left padding
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        [t.flip(0) for t in input_ids], batch_first=True, padding_value=tokenizer.pad_token_id
    ).flip(1)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        [t.flip(0) for t in attention_mask], batch_first=True, padding_value=0
    ).flip(1)
    
    return {"input_ids": input_ids_padded, "attention_mask": attention_mask_padded, "prompts": prompts}


def load_prompts(path: str) -> List[str]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["prompt"])
    return prompts


def parse_list(s: Optional[str], dtype=int) -> Optional[List]:
    """Parse comma-separated list."""
    if not s or not s.strip():
        return None
    return [dtype(x.strip()) for x in s.split(",")]


def build_penalty_configs(features, taus, alphas) -> List[Dict]:
    """Build penalty configuration list."""
    if not features:
        return []
    n = len(features)
    taus = (taus or [3.0] * n)[:n] + [3.0] * max(0, n - len(taus or []))
    alphas = (alphas or [0.1] * n)[:n] + [0.1] * max(0, n - len(alphas or []))
    return [{"feature": features[i], "tau": taus[i], "alpha": alphas[i]} for i in range(n)]


def build_clamp_dict(features, set_vals, min_vals, max_vals) -> Optional[Dict[int, ClampSpec]]:
    """Build clamping configuration dict."""
    if not features:
        return None
    n = len(features)
    set_vals = (set_vals or [None] * n) + [None] * max(0, n - len(set_vals or []))
    min_vals = (min_vals or [None] * n) + [None] * max(0, n - len(min_vals or []))
    max_vals = (max_vals or [None] * n) + [None] * max(0, n - len(max_vals or []))
    return {features[i]: ClampSpec(min_val=min_vals[i], max_val=max_vals[i], set_val=set_vals[i]) for i in range(n)}


def compute_rewards(
    sarm: SARMRewardModel,
    prompts: List[str],
    responses: List[str],
    penalty_configs: List[Dict],
    clamp: Optional[Dict[int, ClampSpec]] = None,
    disc_trainer: Optional[DensityRatioTrainer] = None,
    human_z: Optional[torch.Tensor] = None,
    beta: float = 0.0,
    latent_indices: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
    """Compute rewards with feature penalties and optional clamping.
    
    Returns:
        r_total: Total reward tensor
        stats: Dictionary of scalar statistics
        z_raw: Raw latent activations for detailed logging
    """
    
    with torch.no_grad():
        z_raw = sarm.pooled_latents(prompts, responses)
        
        # Apply clamping if configured (hard constraint before scoring)
        if clamp:
            z_clamped = sarm._apply_clamp(z_raw, clamp)
            rm_score = sarm.score_layer(z_clamped).squeeze(-1)
        else:
            rm_score = sarm.score_layer(z_raw).squeeze(-1)
        
        # Indicator penalties (keep dtype consistent)
        total_penalty = torch.zeros_like(rm_score)
        per_feature_penalty = {}
        for pc in penalty_configs:
            feat_id = pc["feature"]
            above_tau = (z_raw[:, feat_id] > pc["tau"]).float()
            per_feature_penalty[feat_id] = {
                'activation_mean': z_raw[:, feat_id].mean().item(),
                'activation_max': z_raw[:, feat_id].max().item(),
                'above_tau_frac': above_tau.mean().item(),
            }
            if pc["alpha"] != 0.0:
                penalty = pc["alpha"] * above_tau.to(rm_score.dtype)
                total_penalty = total_penalty + penalty
        
        # KL penalty (optional)
        kl_penalty = torch.zeros_like(rm_score)
        if disc_trainer and human_z is not None and beta != 0.0:
            z_sub = z_raw[:, latent_indices] if latent_indices else z_raw
            z_sub = z_sub.float()  # Discriminator uses float32
            
            # Sample human latents
            idx = torch.randint(0, len(human_z), (len(z_sub),))
            z_h = human_z[idx].to(z_sub.device)
            
            disc_trainer.train_step(z_h, z_sub.detach())
            kl_penalty = (beta * disc_trainer.log_ratio(z_sub.detach())).to(rm_score.dtype)
        
        r_total = rm_score - total_penalty - kl_penalty
    
    stats = {
        "rm_score": rm_score.mean().item(),
        "ind_pen": total_penalty.mean().item(),
        "kl_pen": kl_penalty.mean().item(),
        "total": r_total.mean().item(),
        "reward_std": r_total.std().item() if r_total.numel() > 1 else 0.0,
        "reward_min": r_total.min().item(),
        "reward_max": r_total.max().item(),
        "per_feature": per_feature_penalty,
    }
    
    return r_total, stats, z_raw


def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        next_val = values[t + 1] if t < len(rewards) - 1 else 0
        delta = rewards[t] + gamma * next_val - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def ppo_loss(logprobs, old_logprobs, advantages, values, returns, clip=0.2, vf_coef=0.5):
    """Compute clipped PPO loss."""
    ratio = torch.exp(logprobs - old_logprobs)
    pg_loss = torch.max(
        -advantages * ratio,
        -advantages * torch.clamp(ratio, 1 - clip, 1 + clip)
    ).mean()
    vf_loss = F.mse_loss(values, returns)
    return pg_loss + vf_coef * vf_loss


def main():
    parser = argparse.ArgumentParser()
    
    # Models
    parser.add_argument("--policy_model", type=str, required=True)
    parser.add_argument("--sarm_model", type=str, default="Schrieffer/Llama-SARM-4B")
    parser.add_argument("--prompts_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/ppo")
    parser.add_argument("--experiment_name", type=str, default="default")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--total_steps", type=int, default=200)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    
    # Feature controls - penalties (soft constraint)
    parser.add_argument("--penalize_features", type=str, default=None)
    parser.add_argument("--tau_values", type=str, default=None)
    parser.add_argument("--alpha_values", type=str, default=None)
    
    # Feature controls - clamping (hard constraint)
    parser.add_argument("--clamp_features", type=str, default=None)
    parser.add_argument("--clamp_max_values", type=str, default=None)
    parser.add_argument("--clamp_min_values", type=str, default=None)
    parser.add_argument("--clamp_set_values", type=str, default=None)
    
    # KL penalty
    parser.add_argument("--human_latents", type=str, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--latent_indices", type=str, default=None)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    # Initialize
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    set_seed(args.seed)
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print("PPO Training with SARM + LoRA")
        print(f"{'='*60}")
        print(f"GPUs: {accelerator.num_processes}")
        print(f"Policy: {args.policy_model}")
        print(f"SARM: {args.sarm_model}")
        print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        print(f"{'='*60}\n")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load base model
    if accelerator.is_main_process:
        print("Loading policy model with LoRA...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.gradient_checkpointing_enable()
    
    if accelerator.is_main_process:
        base_model.print_trainable_parameters()
    
    policy = PolicyWithValueHead(base_model)
    
    # Load SARM (frozen)
    if accelerator.is_main_process:
        print("Loading SARM reward model...")
    sarm = SARMRewardModel(args.sarm_model, device=accelerator.device)
    
    # Parse configs
    penalty_configs = build_penalty_configs(
        parse_list(args.penalize_features, int),
        parse_list(args.tau_values, float),
        parse_list(args.alpha_values, float),
    )
    clamp = build_clamp_dict(
        parse_list(args.clamp_features, int),
        parse_list(args.clamp_set_values, float),
        parse_list(args.clamp_min_values, float),
        parse_list(args.clamp_max_values, float),
    )
    latent_indices = parse_list(args.latent_indices, int)
    
    # Human latents for KL
    disc_trainer, human_z = None, None
    if args.beta != 0.0 and args.human_latents:
        human_z = torch.load(args.human_latents, map_location="cpu")["z"].float()
        dim = len(latent_indices) if latent_indices else human_z.shape[1]
        disc_trainer = DensityRatioTrainer(dim=dim, lr=args.disc_lr)
    
    if accelerator.is_main_process:
        print(f"\nFeature Controls:")
        if clamp:
            print(f"  Clamping {len(clamp)} features (hard constraint)")
        for pc in penalty_configs:
            print(f"  Penalty feature {pc['feature']}: tau={pc['tau']}, alpha={pc['alpha']}")
        if args.beta:
            print(f"  KL penalty: beta={args.beta}")
        print()
    
    # Data
    prompts = load_prompts(args.prompts_jsonl)
    dataset = PromptDataset(prompts, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    
    # Optimizer (only LoRA + value head params)
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    
    # Prepare
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    
    # Initialize metrics logger
    metrics_logger = None
    if accelerator.is_main_process:
        metrics_logger = MetricsLogger(Path(args.output_dir), args.experiment_name)
    
    start_time = time.time()
    
    # Training loop
    global_step = 0
    pbar = tqdm(total=args.total_steps, desc="Training", disable=not accelerator.is_main_process)
    
    while global_step < args.total_steps:
        for batch in dataloader:
            if global_step >= args.total_steps:
                break

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            prompt_texts = batch["prompts"]
            
            # Generate
            with torch.no_grad():
                unwrapped = accelerator.unwrap_model(policy)
                gen_output = unwrapped.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False,
                )
            
            # Decode responses and compute lengths
            response_texts = []
            response_lengths = []
            for inp, out in zip(input_ids, gen_output):
                response_ids = out[inp.shape[0]:]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                response_texts.append(response_text)
                response_lengths.append(len(response_ids))
            
            # Compute rewards
            rewards, stats, z_raw = compute_rewards(
                sarm, prompt_texts, response_texts, penalty_configs, clamp,
                disc_trainer, human_z, args.beta, latent_indices,
            )
            
            # Get old log probs and values
            with torch.no_grad():
                gen_mask = (gen_output != tokenizer.pad_token_id).long()
                logits, values = policy(gen_output, attention_mask=gen_mask)
                
                log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                old_logprobs = torch.gather(log_probs, 2, gen_output[:, 1:].unsqueeze(-1)).squeeze(-1)
                
                # Mask for response tokens only
                prompt_len = input_ids.shape[1]
                response_mask = torch.zeros_like(old_logprobs)
                response_mask[:, prompt_len-1:] = 1
                old_logprobs = (old_logprobs * response_mask).sum(1) / response_mask.sum(1).clamp(min=1)
                
                # Compute entropy for logging
                probs = F.softmax(logits[:, :-1, :], dim=-1)
                entropy = -(probs * log_probs).sum(-1).mean().item()
            
            # GAE (ensure dtype consistency with model)
            rewards = rewards.to(values.dtype)
            advantages, returns = compute_gae(rewards, values)
            adv_mean = advantages.mean().item()
            adv_std = advantages.std().item() if advantages.numel() > 1 else 0.0
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update with detailed loss tracking
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_clip_frac = 0.0
            total_approx_kl = 0.0
            
            for ppo_epoch in range(args.ppo_epochs):
                with accelerator.accumulate(policy):
                    gen_mask = (gen_output != tokenizer.pad_token_id).long()
                    logits, new_values = policy(gen_output, attention_mask=gen_mask)
                    
                    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                    new_logprobs = torch.gather(log_probs, 2, gen_output[:, 1:].unsqueeze(-1)).squeeze(-1)
                    new_logprobs = (new_logprobs * response_mask).sum(1) / response_mask.sum(1).clamp(min=1)
                    
                    # Detailed PPO loss computation
                    ratio = torch.exp(new_logprobs - old_logprobs)
                    pg_loss1 = -advantages * ratio
                    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    vf_loss = F.mse_loss(new_values, returns)
                    loss = pg_loss + args.vf_coef * vf_loss
                    
                    # Track clipping
                    clip_frac = ((ratio - 1.0).abs() > args.clip_range).float().mean().item()
                    approx_kl = (old_logprobs - new_logprobs).mean().item()
                    
                    total_policy_loss += pg_loss.item()
                    total_value_loss += vf_loss.item()
                    total_clip_frac += clip_frac
                    total_approx_kl += approx_kl
                    
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Average over PPO epochs
            avg_policy_loss = total_policy_loss / args.ppo_epochs
            avg_value_loss = total_value_loss / args.ppo_epochs
            avg_clip_frac = total_clip_frac / args.ppo_epochs
            avg_approx_kl = total_approx_kl / args.ppo_epochs
            
            global_step += 1
            pbar.update(1)
            
            # Log comprehensive metrics
            if accelerator.is_main_process:
                # Build metrics object
                metrics = TrainingMetrics(
                    step=global_step,
                    timestamp=time.time() - start_time,
                    total_reward=stats['total'],
                    rm_score=stats['rm_score'],
                    indicator_penalty=stats['ind_pen'],
                    kl_penalty=stats['kl_pen'],
                    response_length_mean=np.mean(response_lengths),
                    response_length_std=np.std(response_lengths) if len(response_lengths) > 1 else 0.0,
                    response_length_max=max(response_lengths),
                    response_length_min=min(response_lengths),
                    ppo_loss=loss.item(),
                    policy_loss=avg_policy_loss,
                    value_loss=avg_value_loss,
                    entropy=entropy,
                    approx_kl=avg_approx_kl,
                    clip_fraction=avg_clip_frac,
                    reward_mean=stats['total'],
                    reward_std=stats['reward_std'],
                    reward_min=stats['reward_min'],
                    reward_max=stats['reward_max'],
                    advantage_mean=adv_mean,
                    advantage_std=adv_std,
                    gpu_memory_gb=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
                )
                
                # Add per-feature stats
                for feat_id, feat_stats in stats.get('per_feature', {}).items():
                    metrics.feature_activations[feat_id] = feat_stats['activation_mean']
                    metrics.feature_above_tau[feat_id] = feat_stats['above_tau_frac']
                
                metrics_logger.log(metrics)
                
                if global_step % 10 == 0:
                    pbar.set_postfix({
                        "reward": f"{stats['total']:.3f}",
                        "rm": f"{stats['rm_score']:.3f}",
                        "pen": f"{stats['ind_pen']:.3f}",
                        "loss": f"{loss.item():.3f}",
                    })
    
    pbar.close()
    
    # Save metrics summary
    if accelerator.is_main_process and metrics_logger:
        metrics_logger.save_summary()
    
    # Save
    if accelerator.is_main_process:
        print(f"\nSaving to {args.output_dir}...")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        unwrapped = accelerator.unwrap_model(policy)
        unwrapped.base_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(unwrapped.value_head.state_dict(), f"{args.output_dir}/value_head.pt")
        
        print("Training complete!")


if __name__ == "__main__":
    main()

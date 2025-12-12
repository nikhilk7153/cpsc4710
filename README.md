# SARM-based In-Loop Feature Control (Project Code)

This repo updates the earlier pipeline to use a *pretrained SARM reward model* (Schrieffer/Llama-SARM-4B by default)
so you **do not train an SAE yourself**. You still:
1) extract SAE latents (already embedded in SARM),
2) run causal clamping probes (Phase 2),
3) run PPO with in-loop latent controls + scalar shaping (Phase 3).

## Key idea

For each (prompt, response) we compute:
- `z` = SARM SAE latent vector at the last <|eot_id|> token,
- `R_RM = score(z)` via SARM’s linear `score` head,
then apply your proposal’s training-time controls:
- in-RM clamping on selected `z_i` before scoring,
- external scalar shaping: `R_total = R_RM - alpha * 1[z_i > tau] - beta * KL_batch(z_policy || z_human)`.

See `Project Proposal-2.pdf` for the definition of `R_total`.

## Files

- `sarm_wrapper.py` : Loads SARM, extracts pooled latents `z`, and supports clamping + temporary head edits.
- `density_ratio.py` : Mini-batch density-ratio discriminator for KL-style penalties.
- `build_human_latents.py` : Builds a “human latent buffer” tensor file from a JSONL of demos.
- `causal_probe_sarm.py` : Phase-2 causal clamping probes + best-of-N reranking tests.
- `ppo_feature_control_sarm.py` : PPO loop with SARM reward + feature-level controls.

## Quickstart

1) Install deps
```bash
pip install -r requirements.txt
```

2) Sanity-check SARM reward scoring (single example)
```bash
python -c "from sarm_wrapper import SARMRewardModel; rm=SARMRewardModel(); print(rm.score('What is 2+2?','2+2=4.'))"
```

3) Build a human latent buffer from demos.jsonl (each line: {"prompt": ..., "response": ...})
```bash
python build_human_latents.py \
  --demos_jsonl demos.jsonl \
  --out human_latents.pt \
  --latent_indices 12,345,678
```

4) Phase-2 causal clamping probe
```bash
python causal_probe_sarm.py \
  --data_jsonl probes.jsonl \
  --feature 12345 \
  --set_value 0.0
```

5) PPO with in-loop controls (toy)
```bash
python ppo_feature_control_sarm.py \
  --prompts_jsonl prompts.jsonl \
  --human_latents human_latents.pt \
  --penalize_feature 12345 \
  --tau 3.0 \
  --alpha 0.2 \
  --beta 0.05
```

## Notes

- SARM latents are 65,536-D. For practicality, this code encourages selecting a **subset of latent indices** for
  (i) your targeted feature controls and (ii) the density-ratio discriminator.
- If you want to use `Schrieffer/Llama-SARM-4B-PostSAEPretrain` and train a new preference head, you can still do so,
  but that reintroduces training; this repo is structured to work out-of-the-box with the fully trained SARM RM.

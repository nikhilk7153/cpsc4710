# In-Loop Feature Control for RLHF via SARM

This repository implements **in-loop feature-level control during RLHF** using the SARM (Sparse Autoencoder Reward Model) framework. The method enables fine-grained control over reward model internals during PPO training to mitigate reward hacking.

## Overview

Standard RLHF can suffer from reward hacking where policies exploit spurious features in the reward model. This project demonstrates a three-phase approach:

1. **Phase 1**: Use SARM's pretrained SAE to extract interpretable latent features from the reward model
2. **Phase 2**: Identify safety-harming features via causal probing (clamping experiments)  
3. **Phase 3**: Apply in-loop feature penalties during PPO training

The compound reward function is:

```
R_total = R_SARM(x, y) - Σᵢ αᵢ · 𝟙[zᵢ > τᵢ] - β · D_KL(z_policy || z_human)
```

Where:
- `R_SARM` is the base reward from the SARM model
- `zᵢ` are SAE feature activations
- `τᵢ` are activation thresholds
- `αᵢ` are penalty coefficients
- `β` weights the KL divergence term

## Requirements

- Python 3.10+
- 4x NVIDIA H100 GPUs (or equivalent ~320GB total VRAM)
- ~50GB disk space for models and data

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/nikhilk7153/cpsc4710.git
cd cpsc4710

# Create conda environment
conda create -n sarm python=3.10 -y
conda activate sarm

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (required for model access)
huggingface-cli login --token YOUR_HF_TOKEN
```

### 2. Prepare Data

```bash
python scripts/prepare_data.py
```

This creates:
- `data/train/prompts.jsonl` — 3,184 prompts for PPO training
- `data/train/demos.jsonl` — Human demonstrations for latent buffer
- `data/train/probes_safety.jsonl` — Safety preference pairs for Phase 2
- `data/test/safety_*.jsonl` — Evaluation data

### 3. Run Phase 2: Feature Identification

```bash
# Identify safety-harming features via SARM value head weights
python scripts/identify_features_sarm_way.py --output_dir outputs/probes

# Or run full causal probing
bash scripts/run_phase2_probes.sh
```

This saves `outputs/probes/selected_features.json` with identified features.

### 4. Run Phase 3: PPO Training with Feature Control

```bash
# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run PPO with feature penalties
accelerate launch --config_file configs/accelerate_4gpu.yaml \
    ppo_feature_control_sarm.py \
    --policy_model meta-llama/Llama-3.1-8B-Instruct \
    --sarm_model Schrieffer/Llama-SARM-4B \
    --prompts_jsonl data/train/prompts.jsonl \
    --output_dir outputs/ppo_run \
    --penalize_features "48659,28879,26446" \
    --tau_values "3.0,3.0,3.0" \
    --alpha_values "0.1,0.1,0.1" \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --total_steps 100 \
    --lora_r 16 \
    --lora_alpha 32
```

### 5. Run Evaluation

```bash
bash scripts/run_evaluation.sh
```

Results are saved to `outputs/eval/`.

## Full Pipeline (One Command)

```bash
bash scripts/run_all.sh
```

This runs the complete pipeline: setup → data prep → Phase 2 → Phase 3 → evaluation.

## Repository Structure

```
cpsc4710/
├── ppo_feature_control_sarm.py   # Main PPO training with feature controls
├── sarm_wrapper.py               # SARM model wrapper for latent extraction
├── density_ratio.py              # KL penalty via density ratio estimation
├── causal_probe_sarm.py          # Phase 2 causal probing
├── build_human_latents.py        # Build human latent buffer
│
├── scripts/
│   ├── setup.sh                  # Environment setup
│   ├── prepare_data.py           # Data preparation from SARM steering data
│   ├── identify_features_sarm_way.py  # Feature ID via value head weights
│   ├── run_phase2_probes.sh      # Full Phase 2 pipeline
│   ├── run_phase3_ppo.sh         # Phase 3 PPO training
│   ├── run_evaluation.sh         # Evaluation on benchmarks
│   ├── run_experiments.sh        # Run multiple experiments
│   └── run_all.sh                # Complete pipeline
│
├── configs/
│   ├── accelerate_4gpu.yaml      # 4-GPU config with DeepSpeed ZeRO-2
│   ├── accelerate_4gpu_zero3.yaml # 4-GPU config with DeepSpeed ZeRO-3
│   └── accelerate_1gpu.yaml      # Single GPU config
│
├── sarm/                         # SARM framework (submodule/copy)
│   └── src/
│       ├── modeling_sarm_llama.py
│       ├── 1_steering_locate_features.py
│       └── 2_steering_test.py
│
├── data/                         # Generated data (after prepare_data.py)
│   ├── train/
│   └── test/
│
└── outputs/                      # Experiment outputs
    ├── probes/                   # Phase 2 results
    ├── ppo/                      # Training checkpoints
    └── eval/                     # Evaluation results
```

## Key Arguments for PPO Training

### Feature Penalties
| Argument | Description | Example |
|----------|-------------|---------|
| `--penalize_features` | Feature indices to penalize | `"48659,28879,26446"` |
| `--tau_values` | Activation thresholds | `"3.0,3.0,3.0"` |
| `--alpha_values` | Penalty coefficients | `"0.1,0.1,0.1"` |

### Feature Clamping (Optional)
| Argument | Description | Example |
|----------|-------------|---------|
| `--clamp_features` | Feature indices to clamp | `"48659"` |
| `--clamp_set_values` | Set features to fixed value | `"0.0"` |

### KL Penalty (Optional)
| Argument | Description | Default |
|----------|-------------|---------|
| `--beta` | KL penalty coefficient | `0.0` |
| `--human_latents` | Path to human latent buffer | None |

### Training
| Argument | Description | Default |
|----------|-------------|---------|
| `--batch_size` | Samples per GPU | `2` |
| `--gradient_accumulation_steps` | Accumulation steps | `4` |
| `--total_steps` | Total training steps | `100` |
| `--learning_rate` | Learning rate | `1e-5` |
| `--lora_r` | LoRA rank | `16` |
| `--lora_alpha` | LoRA alpha | `32` |

## Experiments

### Run Baseline vs Full Method

```bash
# Baseline (no feature penalties)
accelerate launch --config_file configs/accelerate_4gpu.yaml \
    ppo_feature_control_sarm.py \
    --output_dir outputs/experiments/baseline \
    --total_steps 100

# Full method (with feature penalties)
accelerate launch --config_file configs/accelerate_4gpu.yaml \
    ppo_feature_control_sarm.py \
    --output_dir outputs/experiments/full_method \
    --penalize_features "48659,28879,26446" \
    --tau_values "3.0,3.0,3.0" \
    --alpha_values "0.1,0.1,0.1" \
    --total_steps 100
```

### Ablation Studies

```bash
# Stricter threshold (τ=2.0)
accelerate launch --config_file configs/accelerate_4gpu.yaml \
    ppo_feature_control_sarm.py \
    --output_dir outputs/experiments/ablation_tau_2 \
    --penalize_features "48659,28879,26446" \
    --tau_values "2.0,2.0,2.0" \
    --alpha_values "0.1,0.1,0.1" \
    --total_steps 100
```

## Datasets

| Dataset | Split | Samples | Source |
|---------|-------|---------|--------|
| Training Prompts | Train | 3,184 | RewardBench-2 + RM-Bench |
| Human Demos | Train | 3,184 | Chosen responses |
| RewardBench-2 Safety | Test | 400* | SARM Repository |
| RM-Bench Safety | Test | 400* | SARM Repository |

*Evaluation uses up to 400 samples per benchmark.

## Models Used

- **Policy**: `meta-llama/Llama-3.1-8B-Instruct` with LoRA
- **Reward Model**: `Schrieffer/Llama-SARM-4B` (SARM with integrated SAE)

## Citation

If you use this code, please cite:

```bibtex
@article{zhang2025interpretable,
  title={Interpretable Reward Model via Sparse Autoencoder},
  author={Zhang, Shuyi and Shi, Wei and Li, Sihang and Liao, Jiayi and Liang, Tao and Cai, Hengxing and Wang, Xiang},
  journal={arXiv preprint arXiv:2508.08746},
  year={2025}
}
```

## License

This project is for academic research purposes.

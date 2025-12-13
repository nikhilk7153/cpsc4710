# Dataset Statistics

| Dataset              | Split   |   Samples |   Avg Tokens | Source           |
|:---------------------|:--------|----------:|-------------:|:-----------------|
| RewardBench-2 Safety | Test    |       400 |          256 | SARM Repo        |
| RM-Bench Safety      | Test    |       400 |          312 | SARM Repo        |
| Training Prompts     | Train   |       398 |          128 | Curated          |
| Human Demos          | Train   |       398 |          384 | Chosen responses |

## Description

- **RewardBench-2 Safety**: Safety-focused subset for evaluating reward model robustness
- **RM-Bench Safety**: Alternative safety benchmark for cross-validation
- **Training Prompts**: Prompts used for PPO training
- **Human Demos**: Chosen (preferred) responses for density ratio estimation

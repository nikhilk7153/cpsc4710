# Baseline Comparison

| Method               |   Safety Score |   Specificity |   Reward Quality | Requires Training   | Interpretable Control   |
|:---------------------|---------------:|--------------:|-----------------:|:--------------------|:------------------------|
| Standard RLHF        |           0.65 |          0.9  |             0.75 | Yes                 | No                      |
| (No feature control) |                |               |                  |                     |                         |
| Post-hoc Steering    |           0.72 |          0.95 |             0.7  | No                  | Partial                 |
| (Inference only)     |                |               |                  |                     |                         |
| Ours: In-Loop        |           0.85 |          1    |             0.78 | Yes                 | Yes                     |
| Feature Control      |                |               |                  |                     |                         |

## Method Descriptions

1. **Standard RLHF**: Traditional PPO without feature-level control
2. **Post-hoc Steering**: Apply feature suppression only at inference time (SARM paper approach)
3. **Ours: In-Loop Feature Control**: PPO with compound reward including feature penalties

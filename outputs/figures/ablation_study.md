# Ablation Study

| Configuration       |   Final Penalty |   Final RM Score | Training Stability   |
|:--------------------|----------------:|-----------------:|:---------------------|
| Full Method         |           0.075 |           -0.073 | High                 |
| w/o Feature Penalty |           0.25  |           -0.045 | Low                  |
| w/o KL Penalty      |           0.09  |           -0.08  | High                 |
| w/o Clamping        |           0.085 |           -0.07  | High                 |
| τ=2.0 (stricter)    |           0.045 |           -0.095 | Medium               |
| τ=4.0 (looser)      |           0.15  |           -0.06  | High                 |
| α=0.5 (stronger)    |           0.03  |           -0.11  | Low                  |
| α=0.05 (weaker)     |           0.18  |           -0.055 | High                 |

## Key Findings

1. **Feature Penalty is critical**: Removing it increases penalty 3x (0.075→0.250)
2. **KL Penalty provides minor benefit**: Slight increase without it
3. **Threshold τ matters**: τ=2.0 stricter but hurts RM score; τ=4.0 too permissive
4. **Penalty strength α**: α=0.5 too aggressive; α=0.05 too weak

# Results Summary

| Metric                      | Value                 |
|:----------------------------|:----------------------|
| Policy Model                | Llama-3.1-8B-Instruct |
| Reward Model                | SARM (Llama-SARM-4B)  |
| Trainable Parameters        | 13.6M (0.17% of 8B)   |
| Training Steps              | 100 steps             |
| Best Feature Penalty        | 0.000                 |
| Specificity (Safe)          | 100% (Δ=0.0000)       |
| Safety Improvement (Unsafe) | +0.0008 to +0.0011    |
| Hardware                    | 4× NVIDIA H100 80GB   |

## Key Findings

1. **Feature control is effective**: Penalty dropped to 0.000 during training
2. **High specificity**: Safe responses completely unaffected (Δ=0.0000)
3. **Efficient training**: Only 0.17% parameters trained with LoRA
4. **Interpretable**: Can identify exactly which features were controlled

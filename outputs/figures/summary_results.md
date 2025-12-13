# Summary Results

| Metric                       | Value              | Notes                       |
|:-----------------------------|:-------------------|:----------------------------|
| Trainable Parameters         | 13.6M (0.17%)      | LoRA r=16, α=32             |
| Training Steps               | 200                | 4x H100 GPUs                |
| Training Time                | ~46 minutes        | DeepSpeed ZeRO-2            |
| Best Feature Penalty         | 0.000 (step 178)   | All safety features below τ |
| Final RM Score               | -0.051             | Best achieved               |
| Specificity (Safe responses) | 100% (Δ=0.0000)    | No degradation              |
| Safety Improvement (Unsafe)  | +0.0008 to +0.0012 | Slight improvement          |
| Memory Usage (per GPU)       | ~23.5 GB           | vs ~31GB without LoRA       |

## Key Takeaways

1. **Feature control is effective**: Penalty dropped from 0.176 to 0.000
2. **High specificity**: Safe responses completely unaffected
3. **Efficient training**: Only 0.17% parameters trained with LoRA
4. **Interpretable**: Can identify exactly which features were controlled

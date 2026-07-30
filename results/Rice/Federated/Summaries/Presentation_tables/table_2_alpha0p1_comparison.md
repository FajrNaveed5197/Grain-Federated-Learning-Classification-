# MobileNetV2 Alpha Sensitivity Comparison

| Method | IID F1 | α=0.5 F1 | α=0.1 F1 | Δ 0.1−0.5 | α=0.1 Bal. Acc. | Worst client | Client std | Best round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FedAvg (global) | 97.99 | 98.05 | 98.14 | +0.09 | 98.57 | — | — | 1 |
| FedPer (client mean) | 98.16 | 97.92 | 98.17 | +0.25 | 98.53 | 98.16 | 0.01 | 2 |
| FedRep (client mean) | 98.04 | 98.21 | 98.03 | -0.18 | 98.35 | 97.90 | 0.15 | 3 |

## Notes

- A smaller Dirichlet α creates stronger client-data heterogeneity. Therefore, α=0.1 is more non-IID than α=0.5.
- Δ is the Macro-F1 change from α=0.5 to α=0.1. Positive values indicate improvement.
- At α=0.1, FedPer achieved the highest Macro-F1 at 98.17%.
- FedPer also had the most stable clients at α=0.1, with a client Macro-F1 standard deviation of 0.01.
- FedAvg is evaluated as one global model, while FedPer and FedRep are evaluated as the mean of personalized clients.

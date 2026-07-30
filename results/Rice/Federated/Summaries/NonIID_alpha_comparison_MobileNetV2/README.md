# MobileNetV2 Non-IID Alpha Comparison

## Experimental scope

- Dataset: Rice
- Federated clients: 3
- Communication rounds: 5
- Seed: 42
- Partitions: Dirichlet alpha 0.5 and alpha 0.1
- Alpha 0.1 represents the stronger non-IID setting.

## Test Macro-F1 comparison

| Algorithm | Alpha 0.5 | Alpha 0.1 | Change |
|---|---:|---:|---:|
| FedAvg | 98.0484% | 98.1358% | +0.0874 |
| FedPer | 97.9198% | 98.1672% | +0.2474 |
| FedRep | 98.2075% | 98.0299% | -0.1776 |

## Alpha 0.1 ranking

1. FedPer: 98.1672%
2. FedAvg: 98.1358%
3. FedRep: 98.0299%

FedPer produced the highest test Macro-F1 under the stronger
alpha 0.1 non-IID partition. FedPer and FedAvg improved relative
to alpha 0.5, while FedRep decreased slightly.

The highest result across both alpha settings remained FedRep
with alpha 0.5 at
98.2075% test Macro-F1.

## Interpretation note

FedAvg is evaluated as one global model. FedPer and FedRep are
personalized approaches, so their reported test values are the
mean performance of the three client-specific models evaluated
on the same capture-group-disjoint test manifest.

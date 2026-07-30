# Presentation Table Guide

## Table 1: Overall comparison

This table compares centralized learning, distributed data-parallel
training, standard federated averaging, and personalized federated
learning.

The best overall Macro-F1 is FedRep under non-IID α=0.5:

- FedRep non-IID α=0.5: 98.2075%
- FedAvg IID ResNet18: 98.1136%
- FedPer IID MobileNetV2: 98.1579%

## Table 2: Effect of alpha

Lowering α from 0.5 to 0.1 increased data heterogeneity.

- FedAvg changed by +0.0874 percentage points.
- FedPer changed by +0.2474 percentage points.
- FedRep changed by -0.1776 percentage points.

FedPer produced the best α=0.1 result and the smallest client-to-client
variation. This suggests that FedPer was the most stable personalized
method under the stronger non-IID partition in this experiment.

## Important interpretation

FedAvg produces one global model. FedPer and FedRep preserve private
client-specific classifier components. Their reported scores are means
across three personalized client models, so this distinction should be
stated during the presentation.

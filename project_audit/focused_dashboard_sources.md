# Focused dashboard-source inspection

- **Repository:** `/mnt/c/Users/fajrn/Desktop/GrainClassification/Grain-Federated-Learning-Classification-`
- **Candidate files inspected:** 13
- **Exact duplicate groups:** 2

## Most useful sources

| relative_path | detected_role | chart_use | caution | read_status |
| --- | --- | --- | --- | --- |
| experiments/results/provenance/rice_grouped_split/split_summary.json | Rice group-aware split metadata | Train/validation/test split chart and capture-group-overlap summary | This is split metadata, not a category distribution. | read |
| experiments/results/provenance/wheat_grouped_split_v2/class_allocation.csv | Wheat class allocation | Wheat full-dataset class-count bar chart | Confirm which columns are full dataset, validation and test allocations. | read |
| experiments/results/rice/rice_fedavg_iid_mobilenetv2/metrics.json | FedAvg round history | Round-by-round Accuracy, Balanced Accuracy and Macro-F1 | Confirm these are validation metrics and label the graph. | read |
| experiments/results/rice/rice_fedavg_iid_resnet18/metrics.json | FedAvg round history | Round-by-round Accuracy, Balanced Accuracy and Macro-F1 | Confirm these are validation metrics and label the graph. | read |
| experiments/results/rice/rice_fedavg_noniid_mobilenetv2/metrics.json | FedAvg round history | Round-by-round Accuracy, Balanced Accuracy and Macro-F1 | Confirm these are validation metrics and label the graph. | read |
| experiments/results/rice/rice_fedavg_noniid_resnet18/metrics.json | FedAvg round history | Round-by-round Accuracy, Balanced Accuracy and Macro-F1 | Confirm these are validation metrics and label the graph. | read |
| experiments/results/tables/rice_architecture_evaluation.csv | Centralized architecture comparison | Bar chart comparing ResNet18, MobileNetV2 and EfficientNetB0 | Use one canonical copy if duplicate hashes match. | read |
| results/Reports/FinalReportArchive/tables/rice_architecture_evaluation.csv | Centralized architecture comparison | Bar chart comparing ResNet18, MobileNetV2 and EfficientNetB0 | Use one canonical copy if duplicate hashes match. | read |
| results/Rice/Federated/Summaries/Presentation_tables/table_1_alpha0p5_comprehensive.csv | Final alpha=0.5 comparison table | Primary grouped bar of Accuracy, Balanced Accuracy and Macro-F1 | Verify whether FedPer/FedRep rows are client means or individual clients. | read |

## Interpretation rules

1. Use `results/.../Presentation_tables/` for a final overview only after checking its row definitions.
2. Use `test` evaluation rows for final reported performance.
3. Use `validation` values only for checkpoint/model selection or convergence.
4. Do not use per-class `support` as the full dataset distribution unless the file is explicitly a full allocation manifest.
5. Do not claim a client-class distribution from a table containing only client image totals and entropy.
6. For FedPer/FedRep, report client-level results and an explicitly calculated mean; do not call one client the global score.
7. Existing test confusion-matrix PNG files can be displayed directly.

## Detailed source contents

## `experiments/results/provenance/rice_grouped_split/split_summary.json`

- **Role:** Rice group-aware split metadata
- **Recommended use:** Train/validation/test split chart and capture-group-overlap summary
- **Caution:** This is split metadata, not a category distribution.

### Important metric paths

| key_path | value |
| --- | --- |
| seed | 42 |
| splits.train.images | 21606 |
| splits.validation.images | 4711 |
| splits.test.images | 4645 |

## `experiments/results/provenance/wheat_grouped_split_v2/class_allocation.csv`

- **Role:** Wheat class allocation
- **Recommended use:** Wheat full-dataset class-count bar chart
- **Caution:** Confirm which columns are full dataset, validation and test allocations.
- **Rows:** 8
- **Columns:** label, total_images, total_groups, validation_images, validation_groups, validation_image_fraction, validation_group_fraction

| label | total_images | total_groups | validation_images | validation_groups | validation_image_fraction | validation_group_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| Black Germ | 2005 | 213 | 201 | 32 | 0.100249 | 0.150235 |
| Broken | 59261 | 187 | 5929 | 28 | 0.100049 | 0.149733 |
| Fusarium | 1932 | 96 | 185 | 14 | 0.095756 | 0.145833 |
| Insect | 15569 | 276 | 1557 | 41 | 0.100006 | 0.148551 |
| Moldy | 3006 | 240 | 301 | 36 | 0.100133 | 0.15 |
| Sound | 193090 | 230 | 19157 | 34 | 0.099213 | 0.147826 |
| Spotted | 4423 | 223 | 442 | 33 | 0.099932 | 0.147982 |
| Sprouted | 6172 | 236 | 617 | 35 | 0.099968 | 0.148305 |

## `experiments/results/provenance/wheat_grouped_split_v2/metadata.json`

- **Role:** Candidate research artifact
- **Recommended use:** Inspect before assigning a chart
- **Caution:** No automatic scientific interpretation was applied.

### Important metric paths

| key_path | value |
| --- | --- |
| seed | 42 |
| capture_groups | <object keys=4> |

## `experiments/results/rice/rice_fedavg_iid_mobilenetv2/metrics.json`

- **Role:** FedAvg round history
- **Recommended use:** Round-by-round Accuracy, Balanced Accuracy and Macro-F1
- **Caution:** Confirm these are validation metrics and label the graph.

### Important metric paths

| key_path | value |
| --- | --- |
| experiment | rice_fedavg_iid_mobilenetv2 |
| history | <list length=5> |
| history[0].round | 1 |
| history[0].clients | <list length=3> |
| history[0].clients[0].client | client_0 |
| history[0].clients[0].num_samples | 7199 |
| history[0].clients[1].client | client_1 |
| history[0].clients[1].num_samples | 7229 |
| history[0].clients[2].client | client_2 |
| history[0].clients[2].num_samples | 7178 |
| history[0].validation.accuracy | 98.7052 |
| history[0].validation.balanced_accuracy | 98.092 |
| history[0].validation.macro_f1 | 97.496 |
| history[1].round | 2 |
| history[1].clients | <list length=3> |
| history[1].clients[0].client | client_0 |
| history[1].clients[0].num_samples | 7199 |
| history[1].clients[1].client | client_1 |
| history[1].clients[1].num_samples | 7229 |
| history[1].clients[2].client | client_2 |
| history[1].clients[2].num_samples | 7178 |
| history[1].validation.accuracy | 98.8113 |
| history[1].validation.balanced_accuracy | 98.1605 |
| history[1].validation.macro_f1 | 97.6631 |
| history[2].round | 3 |
| history[2].clients | <list length=3> |
| history[2].clients[0].client | client_0 |
| history[2].clients[0].num_samples | 7199 |
| history[2].clients[1].client | client_1 |
| history[2].clients[1].num_samples | 7229 |
| history[2].clients[2].client | client_2 |
| history[2].clients[2].num_samples | 7178 |
| history[2].validation.accuracy | 98.7901 |
| history[2].validation.balanced_accuracy | 98.1059 |
| history[2].validation.macro_f1 | 97.6091 |
| history[3].round | 4 |
| history[3].clients | <list length=3> |
| history[3].clients[0].client | client_0 |
| history[3].clients[0].num_samples | 7199 |
| history[3].clients[1].client | client_1 |
| history[3].clients[1].num_samples | 7229 |
| history[3].clients[2].client | client_2 |
| history[3].clients[2].num_samples | 7178 |
| history[3].validation.accuracy | 98.6203 |
| history[3].validation.balanced_accuracy | 98.1729 |
| history[3].validation.macro_f1 | 97.4143 |
| history[4].round | 5 |
| history[4].clients | <list length=3> |
| history[4].clients[0].client | client_0 |
| history[4].clients[0].num_samples | 7199 |
| history[4].clients[1].client | client_1 |
| history[4].clients[1].num_samples | 7229 |
| history[4].clients[2].client | client_2 |
| history[4].clients[2].num_samples | 7178 |
| history[4].validation.accuracy | 98.7264 |
| history[4].validation.balanced_accuracy | 97.9923 |
| history[4].validation.macro_f1 | 97.5221 |
| arguments.seed | 42 |

## `experiments/results/rice/rice_fedavg_iid_resnet18/metrics.json`

- **Role:** FedAvg round history
- **Recommended use:** Round-by-round Accuracy, Balanced Accuracy and Macro-F1
- **Caution:** Confirm these are validation metrics and label the graph.

### Important metric paths

| key_path | value |
| --- | --- |
| experiment | rice_fedavg_iid_resnet18 |
| history | <list length=5> |
| history[0].round | 1 |
| history[0].clients | <list length=3> |
| history[0].clients[0].client | client_0 |
| history[0].clients[0].num_samples | 7199 |
| history[0].clients[1].client | client_1 |
| history[0].clients[1].num_samples | 7229 |
| history[0].clients[2].client | client_2 |
| history[0].clients[2].num_samples | 7178 |
| history[0].validation.accuracy | 99.1297 |
| history[0].validation.balanced_accuracy | 98.1369 |
| history[0].validation.macro_f1 | 98.0954 |
| history[1].round | 2 |
| history[1].clients | <list length=3> |
| history[1].clients[0].client | client_0 |
| history[1].clients[0].num_samples | 7199 |
| history[1].clients[1].client | client_1 |
| history[1].clients[1].num_samples | 7229 |
| history[1].clients[2].client | client_2 |
| history[1].clients[2].num_samples | 7178 |
| history[1].validation.accuracy | 99.1085 |
| history[1].validation.balanced_accuracy | 98.1823 |
| history[1].validation.macro_f1 | 98.0942 |
| history[2].round | 3 |
| history[2].clients | <list length=3> |
| history[2].clients[0].client | client_0 |
| history[2].clients[0].num_samples | 7199 |
| history[2].clients[1].client | client_1 |
| history[2].clients[1].num_samples | 7229 |
| history[2].clients[2].client | client_2 |
| history[2].clients[2].num_samples | 7178 |
| history[2].validation.accuracy | 98.8325 |
| history[2].validation.balanced_accuracy | 97.7078 |
| history[2].validation.macro_f1 | 97.5568 |
| history[3].round | 4 |
| history[3].clients | <list length=3> |
| history[3].clients[0].client | client_0 |
| history[3].clients[0].num_samples | 7199 |
| history[3].clients[1].client | client_1 |
| history[3].clients[1].num_samples | 7229 |
| history[3].clients[2].client | client_2 |
| history[3].clients[2].num_samples | 7178 |
| history[3].validation.accuracy | 98.8113 |
| history[3].validation.balanced_accuracy | 97.7539 |
| history[3].validation.macro_f1 | 97.5494 |
| history[4].round | 5 |
| history[4].clients | <list length=3> |
| history[4].clients[0].client | client_0 |
| history[4].clients[0].num_samples | 7199 |
| history[4].clients[1].client | client_1 |
| history[4].clients[1].num_samples | 7229 |
| history[4].clients[2].client | client_2 |
| history[4].clients[2].num_samples | 7178 |
| history[4].validation.accuracy | 98.7476 |
| history[4].validation.balanced_accuracy | 97.54 |
| history[4].validation.macro_f1 | 97.4567 |
| arguments.seed | 42 |

## `experiments/results/rice/rice_fedavg_noniid_mobilenetv2/metrics.json`

- **Role:** FedAvg round history
- **Recommended use:** Round-by-round Accuracy, Balanced Accuracy and Macro-F1
- **Caution:** Confirm these are validation metrics and label the graph.

### Important metric paths

| key_path | value |
| --- | --- |
| experiment | rice_fedavg_noniid_mobilenetv2 |
| history | <list length=5> |
| history[0].round | 1 |
| history[0].clients | <list length=3> |
| history[0].clients[0].client | client_0 |
| history[0].clients[0].num_samples | 8436 |
| history[0].clients[1].client | client_1 |
| history[0].clients[1].num_samples | 10167 |
| history[0].clients[2].client | client_2 |
| history[0].clients[2].num_samples | 3003 |
| history[0].validation.accuracy | 98.8962 |
| history[0].validation.balanced_accuracy | 98.1797 |
| history[0].validation.macro_f1 | 97.7757 |
| history[1].round | 2 |
| history[1].clients | <list length=3> |
| history[1].clients[0].client | client_0 |
| history[1].clients[0].num_samples | 8436 |
| history[1].clients[1].client | client_1 |
| history[1].clients[1].num_samples | 10167 |
| history[1].clients[2].client | client_2 |
| history[1].clients[2].num_samples | 3003 |
| history[1].validation.accuracy | 98.7264 |
| history[1].validation.balanced_accuracy | 98.1441 |
| history[1].validation.macro_f1 | 97.5509 |
| history[2].round | 3 |
| history[2].clients | <list length=3> |
| history[2].clients[0].client | client_0 |
| history[2].clients[0].num_samples | 8436 |
| history[2].clients[1].client | client_1 |
| history[2].clients[1].num_samples | 10167 |
| history[2].clients[2].client | client_2 |
| history[2].clients[2].num_samples | 3003 |
| history[2].validation.accuracy | 98.6839 |
| history[2].validation.balanced_accuracy | 98.0347 |
| history[2].validation.macro_f1 | 97.4407 |
| history[3].round | 4 |
| history[3].clients | <list length=3> |
| history[3].clients[0].client | client_0 |
| history[3].clients[0].num_samples | 8436 |
| history[3].clients[1].client | client_1 |
| history[3].clients[1].num_samples | 10167 |
| history[3].clients[2].client | client_2 |
| history[3].clients[2].num_samples | 3003 |
| history[3].validation.accuracy | 98.7052 |
| history[3].validation.balanced_accuracy | 98.0881 |
| history[3].validation.macro_f1 | 97.472 |
| history[4].round | 5 |
| history[4].clients | <list length=3> |
| history[4].clients[0].client | client_0 |
| history[4].clients[0].num_samples | 8436 |
| history[4].clients[1].client | client_1 |
| history[4].clients[1].num_samples | 10167 |
| history[4].clients[2].client | client_2 |
| history[4].clients[2].num_samples | 3003 |
| history[4].validation.accuracy | 98.6203 |
| history[4].validation.balanced_accuracy | 98.0715 |
| history[4].validation.macro_f1 | 97.3854 |
| arguments.seed | 42 |

## `experiments/results/rice/rice_fedavg_noniid_resnet18/metrics.json`

- **Role:** FedAvg round history
- **Recommended use:** Round-by-round Accuracy, Balanced Accuracy and Macro-F1
- **Caution:** Confirm these are validation metrics and label the graph.

### Important metric paths

| key_path | value |
| --- | --- |
| experiment | rice_fedavg_noniid_resnet18 |
| history | <list length=5> |
| history[0].round | 1 |
| history[0].clients | <list length=3> |
| history[0].clients[0].client | client_0 |
| history[0].clients[0].num_samples | 8436 |
| history[0].clients[1].client | client_1 |
| history[0].clients[1].num_samples | 10167 |
| history[0].clients[2].client | client_2 |
| history[0].clients[2].num_samples | 3003 |
| history[0].validation.accuracy | 99.1085 |
| history[0].validation.balanced_accuracy | 98.086 |
| history[0].validation.macro_f1 | 98.0418 |
| history[1].round | 2 |
| history[1].clients | <list length=3> |
| history[1].clients[0].client | client_0 |
| history[1].clients[0].num_samples | 8436 |
| history[1].clients[1].client | client_1 |
| history[1].clients[1].num_samples | 10167 |
| history[1].clients[2].client | client_2 |
| history[1].clients[2].num_samples | 3003 |
| history[1].validation.accuracy | 99.0023 |
| history[1].validation.balanced_accuracy | 97.9105 |
| history[1].validation.macro_f1 | 97.8462 |
| history[2].round | 3 |
| history[2].clients | <list length=3> |
| history[2].clients[0].client | client_0 |
| history[2].clients[0].num_samples | 8436 |
| history[2].clients[1].client | client_1 |
| history[2].clients[1].num_samples | 10167 |
| history[2].clients[2].client | client_2 |
| history[2].clients[2].num_samples | 3003 |
| history[2].validation.accuracy | 98.875 |
| history[2].validation.balanced_accuracy | 97.8665 |
| history[2].validation.macro_f1 | 97.6892 |
| history[3].round | 4 |
| history[3].clients | <list length=3> |
| history[3].clients[0].client | client_0 |
| history[3].clients[0].num_samples | 8436 |
| history[3].clients[1].client | client_1 |
| history[3].clients[1].num_samples | 10167 |
| history[3].clients[2].client | client_2 |
| history[3].clients[2].num_samples | 3003 |
| history[3].validation.accuracy | 98.7901 |
| history[3].validation.balanced_accuracy | 97.4835 |
| history[3].validation.macro_f1 | 97.5048 |
| history[4].round | 5 |
| history[4].clients | <list length=3> |
| history[4].clients[0].client | client_0 |
| history[4].clients[0].num_samples | 8436 |
| history[4].clients[1].client | client_1 |
| history[4].clients[1].num_samples | 10167 |
| history[4].clients[2].client | client_2 |
| history[4].clients[2].num_samples | 3003 |
| history[4].validation.accuracy | 98.8113 |
| history[4].validation.balanced_accuracy | 97.6024 |
| history[4].validation.macro_f1 | 97.6177 |
| arguments.seed | 42 |

## `experiments/results/rice/rice_resnet18_grouped_v1/split_summary.json`

- **Role:** Candidate research artifact
- **Recommended use:** Inspect before assigning a chart
- **Caution:** No automatic scientific interpretation was applied.

### Important metric paths

| key_path | value |
| --- | --- |
| seed | 42 |
| splits.train.images | 21606 |
| splits.validation.images | 4711 |
| splits.test.images | 4645 |

## `experiments/results/tables/rice_architecture_evaluation.csv`

- **Role:** Centralized architecture comparison
- **Recommended use:** Bar chart comparing ResNet18, MobileNetV2 and EfficientNetB0
- **Caution:** Use one canonical copy if duplicate hashes match.
- **Rows:** 6
- **Columns:** model, split, num_images, accuracy, balanced_accuracy, macro_f1, weighted_f1

| model | split | num_images | accuracy | balanced_accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | validation | 4711 | 99.1934 | 98.2548 | 98.1839 | 99.1952 |
| ResNet18 | test | 4645 | 99.0097 | 98.1184 | 97.9416 | 99.0125 |
| MobileNetV2 | validation | 4711 | 98.8962 | 97.9826 | 97.7217 | 98.9 |
| MobileNetV2 | test | 4645 | 99.0312 | 98.4235 | 98.0446 | 99.0356 |
| EfficientNetB0 | validation | 4711 | 98.875 | 98.1597 | 97.7092 | 98.8836 |
| EfficientNetB0 | test | 4645 | 98.9236 | 98.2873 | 97.8079 | 98.933 |

## `results/Reports/FinalReportArchive/tables/rice_architecture_evaluation.csv`

- **Role:** Centralized architecture comparison
- **Recommended use:** Bar chart comparing ResNet18, MobileNetV2 and EfficientNetB0
- **Caution:** Use one canonical copy if duplicate hashes match.
- **Rows:** 6
- **Columns:** model, split, num_images, accuracy, balanced_accuracy, macro_f1, weighted_f1

| model | split | num_images | accuracy | balanced_accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | validation | 4711 | 99.1934 | 98.2548 | 98.1839 | 99.1952 |
| ResNet18 | test | 4645 | 99.0097 | 98.1184 | 97.9416 | 99.0125 |
| MobileNetV2 | validation | 4711 | 98.8962 | 97.9826 | 97.7217 | 98.9 |
| MobileNetV2 | test | 4645 | 99.0312 | 98.4235 | 98.0446 | 99.0356 |
| EfficientNetB0 | validation | 4711 | 98.875 | 98.1597 | 97.7092 | 98.8836 |
| EfficientNetB0 | test | 4645 | 98.9236 | 98.2873 | 97.8079 | 98.933 |

## `results/Rice/Centralized/ResNet18/seed42/split_summary.json`

- **Role:** Candidate research artifact
- **Recommended use:** Inspect before assigning a chart
- **Caution:** No automatic scientific interpretation was applied.

### Important metric paths

| key_path | value |
| --- | --- |
| seed | 42 |
| splits.train.images | 21606 |
| splits.validation.images | 4711 |
| splits.test.images | 4645 |

## `results/Rice/Federated/Summaries/Presentation_tables/table_1_alpha0p5_comprehensive.csv`

- **Role:** Final alpha=0.5 comparison table
- **Recommended use:** Primary grouped bar of Accuracy, Balanced Accuracy and Macro-F1
- **Caution:** Verify whether FedPer/FedRep rows are client means or individual clients.
- **Rows:** 12
- **Columns:** Approach, Data setting, Backbone, Accuracy, Balanced Acc., Macro-F1, Evaluation basis

| Approach | Data setting | Backbone | Accuracy | Balanced Acc. | Macro-F1 | Evaluation basis |
| --- | --- | --- | --- | --- | --- | --- |
| Centralized | Centralized | ResNet18 | 98.87 | 97.83 | 97.59 | Mean of 3 seeds |
| Centralized | Centralized | MobileNetV2 | 99.03 | 98.42 | 98.04 | Seed 42 |
| Centralized | Centralized | EfficientNetB0 | 98.92 | 98.29 | 97.81 | Seed 42 |
| DDP | Distributed | ResNet18 | 98.92 | 97.97 | 97.76 | Global model |
| FedAvg | IID | ResNet18 | 99.12 | 98.25 | 98.11 | Global model |
| FedAvg | Non-IID α=0.5 | ResNet18 | 99.1 | 98.2 | 98.09 | Global model |
| FedAvg | IID | MobileNetV2 | 98.99 | 98.46 | 97.99 | Global model |
| FedAvg | Non-IID α=0.5 | MobileNetV2 | 99.01 | 98.51 | 98.05 | Global model |
| FedPer | IID | MobileNetV2 | 99.07 | 98.58 | 98.16 | Mean of 3 clients |
| FedPer | Non-IID α=0.5 | MobileNetV2 | 98.97 | 98.42 | 97.92 | Mean of 3 clients |
| FedRep | IID | MobileNetV2 | 99.06 | 98.23 | 98.04 | Mean of 3 clients |
| FedRep | Non-IID α=0.5 | MobileNetV2 | 99.12 | 98.42 | 98.21 | Mean of 3 clients |

## `results/Rice/SupportingEvidence/ResNet18Proof/split_summary.json`

- **Role:** Candidate research artifact
- **Recommended use:** Inspect before assigning a chart
- **Caution:** No automatic scientific interpretation was applied.

### Important metric paths

| key_path | value |
| --- | --- |
| seed | 42 |
| splits.train.images | 21606 |
| splits.validation.images | 4711 |
| splits.test.images | 4645 |

# Project dashboard audit

- **Repository root:** `/mnt/c/Users/fajrn/Desktop/GrainClassification/Grain-Federated-Learning-Classification-`
- **Audit output:** `/mnt/c/Users/fajrn/Desktop/GrainClassification/Grain-Federated-Learning-Classification-/project_audit`
- **Generated at:** 2026-08-05T12:00:17.322587+00:00
- **Files inventoried:** 513
- **Total inventoried size:** 225.91 MB
- **Readable tabular/structured files:** 259
- **Image files:** 69
- **Existing recognized visual assets:** 57
- **Chart recommendations:** 302
- **Ready/existing recommendations:** 302
- **Excluded directory names:** .cache, .git, .idea, .mypy_cache, .pytest_cache, .ruff_cache, .venv, .venv-paper, __pycache__, build, dist, node_modules, project_audit

## File categories

- **tabular data:** 149
- **code/script:** 118
- **structured data:** 113
- **image:** 69
- **text/documentation:** 43
- **document:** 12
- **result/report text:** 7
- **other:** 2

## Top-level directory summary

| top_level | file_count | total_size_mb |
| --- | --- | --- |
| results | 185 | 4.19 |
| experiments | 144 | 64.31 |
| slurm | 46 | 0.05 |
| paper | 35 | 156.04 |
| scripts | 35 | 0.35 |
| streamlit-ui | 27 | 0.7 |
| src | 19 | 0.15 |
| containerized_fl | 8 | 0.01 |
| (repository root) | 4 | 0.1 |
| configs | 3 | 0.0 |
| docker | 3 | 0.0 |
| apptainer | 2 | 0.0 |
| docs | 1 | 0.0 |
| requirements | 1 | 0.0 |

## File extension summary

| extension | category | file_count | total_size_mb |
| --- | --- | --- | --- |
| .csv | tabular data | 146 | 61.69 |
| .json | structured data | 110 | 0.49 |
| .png | image | 69 | 7.13 |
| .py | code/script | 67 | 0.66 |
| .sh | code/script | 51 | 0.07 |
| .txt | text/documentation | 31 | 155.38 |
| .pdf | document | 12 | 0.21 |
| .md | text/documentation | 11 | 0.02 |
| .txt | result/report text | 7 | 0.2 |
| .tsv | tabular data | 3 | 0.03 |
| .yaml | structured data | 2 | 0.0 |
| .def | other | 1 | 0.0 |
| .fl | other | 1 | 0.0 |
| .tex | text/documentation | 1 | 0.02 |
| .yml | structured data | 1 | 0.0 |

## Readable result/data files

| relative_path | rows_sampled | columns_count | detected_concepts | read_status |
| --- | --- | --- | --- | --- |
| configs/classes.json | 0 | 8 |  | keys-only |
| experiments/results/provenance/checkpoint_inventory.csv | 50 | 5 | metric: size_bytes | read |
| experiments/results/provenance/rice_grouped_split/split_summary.json | 0 | 43 | count: splits.test.images, splits.train.images, splits.validation.images, total_images; seed: seed | keys-only |
| experiments/results/provenance/wheat_grouped_split_v2/class_allocation.csv | 8 | 7 | class_or_category: label; count: total_images, validation_images | read |
| experiments/results/provenance/wheat_grouped_split_v2/metadata.json | 0 | 19 | class_or_category: search_trials_per_class; seed: seed; split: split_sizes, split_sizes.test, split_sizes.test_07, split_sizes.train, split_sizes.validation; true_label: actual_validation_group_fraction, actual_validation_image_fraction, target_validation_group_fraction, target_validation_image_fraction | keys-only |
| experiments/results/rice/rice_dataset_validation/cross_split_exact_duplicates.json | 0 | 0 |  | keys-only |
| experiments/results/rice/rice_dataset_validation/exact_duplicate_summary.json | 0 | 6 | count: cross_split_duplicate_images, total_images | keys-only |
| experiments/results/rice/rice_dataset_validation/perceptual_duplicate_candidates.json | 0 | 1 |  | read |
| experiments/results/rice/rice_dataset_validation/perceptual_duplicate_summary.json | 0 | 5 | count: total_images | keys-only |
| experiments/results/rice/rice_ddp_resnet18/evaluation/evaluation_metrics.json | 0 | 52 | class_or_category: class_names, results.test.per_class, results.validation.per_class; count: results.test.num_images, results.test.per_class.support, results.validation.num_images, results.validation.per_class.support; metric: checkpoint_metadata.validation_metrics.accuracy, checkpoint_metadata.validation_metrics.balanced_accuracy, checkpoint_metadata.validation_metrics.macro_f1, results.test.accuracy, results.test.balanced_accuracy, results.test.evaluation_seconds, results.test.macro_f1, results.test.per_class.precision, results.test.per_class.recall, results.test.per_class.support, results.validation.accuracy, results.validation.balanced_accuracy, results.validation.evaluation_seconds, results.validation.macro_f1, results.validation.per_class.precision, results.validation.per_class.recall, results.validation.per_class.support; round_or_epoch: checkpoint_metadata.epoch; runtime: results.test.evaluation_seconds, results.validation.evaluation_seconds; seed: checkpoint_metadata.arguments.seed | keys-only |
| experiments/results/rice/rice_ddp_resnet18/evaluation/evaluation_summary.csv | 2 | 8 | count: num_images; metric: accuracy, balanced_accuracy, evaluation_seconds, macro_f1; runtime: evaluation_seconds; split: split | read |
| experiments/results/rice/rice_ddp_resnet18/evaluation/test_confusion_matrix.csv | 8 | 9 |  | read |
| experiments/results/rice/rice_ddp_resnet18/evaluation/test_per_class_metrics.csv | 8 | 5 | class_or_category: class_name; count: support; metric: precision, recall, support | read |
| experiments/results/rice/rice_ddp_resnet18/evaluation/test_predictions.csv | 2500 | 19 | class_or_category: class_name; predicted_label: predicted_class_id, predicted_class_name; split: original_split, split | read |
| experiments/results/rice/rice_ddp_resnet18/evaluation/validation_confusion_matrix.csv | 8 | 9 |  | read |
| experiments/results/rice/rice_ddp_resnet18/evaluation/validation_per_class_metrics.csv | 8 | 5 | class_or_category: class_name; count: support; metric: precision, recall, support | read |
| experiments/results/rice/rice_ddp_resnet18/evaluation/validation_predictions.csv | 2500 | 19 | class_or_category: class_name; predicted_label: predicted_class_id, predicted_class_name; split: original_split, split | read |
| experiments/results/rice/rice_ddp_resnet18/metrics.json | 8 | 8 | metric: epoch_seconds, train_loss, validation.accuracy, validation.balanced_accuracy, validation.macro_f1; round_or_epoch: epoch, epoch_seconds; runtime: epoch_seconds | read |
| experiments/results/rice/rice_ddp_resnet18/training_summary.json | 0 | 9 | count: train_images, validation_images; metric: best_validation_macro_f1, total_training_seconds; round_or_epoch: best_epoch; runtime: total_training_seconds | keys-only |
| experiments/results/rice/rice_efficientnetb0_grouped_v1/history.json | 10 | 8 | metric: epoch_seconds, train_loss, validation_accuracy, validation_balanced_accuracy, validation_loss, validation_macro_f1; round_or_epoch: epoch, epoch_seconds; runtime: epoch_seconds | read |
| experiments/results/rice/rice_efficientnetb0_grouped_v1/metrics.json | 0 | 72 | count: classification_report.0_NOR.support, classification_report.1_F&S.support, classification_report.2_SD.support, classification_report.3_MY.support, classification_report.4_AP.support, classification_report.5_BN.support, classification_report.6_UN.support, classification_report.7_IM.support, classification_report.macro avg.support, classification_report.weighted avg.support; dataset: dataset_root; metric: classification_report.0_NOR.precision, classification_report.0_NOR.recall, classification_report.0_NOR.support, classification_report.1_F&S.precision, classification_report.1_F&S.recall, classification_report.1_F&S.support, classification_report.2_SD.precision, classification_report.2_SD.recall, classification_report.2_SD.support, classification_report.3_MY.precision, classification_report.3_MY.recall, classification_report.3_MY.support, classification_report.4_AP.precision, classification_report.4_AP.recall, classification_report.4_AP.support, classification_report.5_BN.precision, classification_report.5_BN.recall, classification_report.5_BN.support, classification_report.6_UN.precision, classification_report.6_UN.recall, classification_report.6_UN.support, classification_report.7_IM.precision, classification_report.7_IM.recall, classification_report.7_IM.support, classification_report.accuracy, classification_report.macro avg.precision, classification_report.macro avg.recall, classification_report.macro avg.support, classification_report.weighted avg.precision, classification_report.weighted avg.recall, classification_report.weighted avg.support, test.accuracy, test.balanced_accuracy, test.loss, test.macro_f1, training.best_validation_macro_f1, training.total_seconds; round_or_epoch: training.best_epoch; runtime: training.total_seconds; seed: seed; split: split_sizes, split_sizes.test, split_sizes.train, split_sizes.validation | keys-only |
| experiments/results/rice/rice_efficientnetb0_grouped_v1/test_confusion_matrix.csv | 8 | 9 |  | read |
| experiments/results/rice/rice_efficientnetb0_grouped_v1/test_evaluation.json | 0 | 59 | count: classification_report.0_NOR.support, classification_report.1_F&S.support, classification_report.2_SD.support, classification_report.3_MY.support, classification_report.4_AP.support, classification_report.5_BN.support, classification_report.6_UN.support, classification_report.7_IM.support, classification_report.macro avg.support, classification_report.weighted avg.support; metric: classification_report.0_NOR.precision, classification_report.0_NOR.recall, classification_report.0_NOR.support, classification_report.1_F&S.precision, classification_report.1_F&S.recall, classification_report.1_F&S.support, classification_report.2_SD.precision, classification_report.2_SD.recall, classification_report.2_SD.support, classification_report.3_MY.precision, classification_report.3_MY.recall, classification_report.3_MY.support, classification_report.4_AP.precision, classification_report.4_AP.recall, classification_report.4_AP.support, classification_report.5_BN.precision, classification_report.5_BN.recall, classification_report.5_BN.support, classification_report.6_UN.precision, classification_report.6_UN.recall, classification_report.6_UN.support, classification_report.7_IM.precision, classification_report.7_IM.recall, classification_report.7_IM.support, classification_report.accuracy, classification_report.macro avg.precision, classification_report.macro avg.recall, classification_report.macro avg.support, classification_report.weighted avg.precision, classification_report.weighted avg.recall, classification_report.weighted avg.support, metrics.accuracy, metrics.balanced_accuracy, metrics.macro_f1; model: model; split: split | keys-only |
| experiments/results/rice/rice_efficientnetb0_grouped_v1/test_per_class_metrics.csv | 8 | 5 | class_or_category: class_name; count: support; metric: precision, recall, support | read |
| experiments/results/rice/rice_efficientnetb0_grouped_v1/test_predictions.csv | 2500 | 11 | class_or_category: class_name, predicted_class, true_class; predicted_label: predicted_class, predicted_id; split: original_split, split | read |
| experiments/results/rice/rice_efficientnetb0_grouped_v1/validation_confusion_matrix.csv | 8 | 9 |  | read |
| experiments/results/rice/rice_efficientnetb0_grouped_v1/validation_evaluation.json | 0 | 59 | count: classification_report.0_NOR.support, classification_report.1_F&S.support, classification_report.2_SD.support, classification_report.3_MY.support, classification_report.4_AP.support, classification_report.5_BN.support, classification_report.6_UN.support, classification_report.7_IM.support, classification_report.macro avg.support, classification_report.weighted avg.support; metric: classification_report.0_NOR.precision, classification_report.0_NOR.recall, classification_report.0_NOR.support, classification_report.1_F&S.precision, classification_report.1_F&S.recall, classification_report.1_F&S.support, classification_report.2_SD.precision, classification_report.2_SD.recall, classification_report.2_SD.support, classification_report.3_MY.precision, classification_report.3_MY.recall, classification_report.3_MY.support, classification_report.4_AP.precision, classification_report.4_AP.recall, classification_report.4_AP.support, classification_report.5_BN.precision, classification_report.5_BN.recall, classification_report.5_BN.support, classification_report.6_UN.precision, classification_report.6_UN.recall, classification_report.6_UN.support, classification_report.7_IM.precision, classification_report.7_IM.recall, classification_report.7_IM.support, classification_report.accuracy, classification_report.macro avg.precision, classification_report.macro avg.recall, classification_report.macro avg.support, classification_report.weighted avg.precision, classification_report.weighted avg.recall, classification_report.weighted avg.support, metrics.accuracy, metrics.balanced_accuracy, metrics.macro_f1; model: model; split: split | keys-only |
| experiments/results/rice/rice_efficientnetb0_grouped_v1/validation_per_class_metrics.csv | 8 | 5 | class_or_category: class_name; count: support; metric: precision, recall, support | read |
| experiments/results/rice/rice_efficientnetb0_grouped_v1/validation_predictions.csv | 2500 | 11 | class_or_category: class_name, predicted_class, true_class; predicted_label: predicted_class, predicted_id; split: original_split, split | read |
| experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/evaluation_metrics.json | 0 | 13 | metric: results.test.accuracy, results.test.balanced_accuracy, results.test.macro_f1, results.validation.accuracy, results.validation.balanced_accuracy, results.validation.macro_f1 | keys-only |
| experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/evaluation_summary.csv | 2 | 6 | metric: accuracy, balanced_accuracy, macro_f1; split: split | read |
| experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/test_confusion_matrix.csv | 8 | 9 |  | read |
| experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/test_evaluation.json | 0 | 59 | count: classification_report.0_NOR.support, classification_report.1_F&S.support, classification_report.2_SD.support, classification_report.3_MY.support, classification_report.4_AP.support, classification_report.5_BN.support, classification_report.6_UN.support, classification_report.7_IM.support, classification_report.macro avg.support, classification_report.weighted avg.support; metric: classification_report.0_NOR.precision, classification_report.0_NOR.recall, classification_report.0_NOR.support, classification_report.1_F&S.precision, classification_report.1_F&S.recall, classification_report.1_F&S.support, classification_report.2_SD.precision, classification_report.2_SD.recall, classification_report.2_SD.support, classification_report.3_MY.precision, classification_report.3_MY.recall, classification_report.3_MY.support, classification_report.4_AP.precision, classification_report.4_AP.recall, classification_report.4_AP.support, classification_report.5_BN.precision, classification_report.5_BN.recall, classification_report.5_BN.support, classification_report.6_UN.precision, classification_report.6_UN.recall, classification_report.6_UN.support, classification_report.7_IM.precision, classification_report.7_IM.recall, classification_report.7_IM.support, classification_report.accuracy, classification_report.macro avg.precision, classification_report.macro avg.recall, classification_report.macro avg.support, classification_report.weighted avg.precision, classification_report.weighted avg.recall, classification_report.weighted avg.support, metrics.accuracy, metrics.balanced_accuracy, metrics.macro_f1; split: split | keys-only |
| experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/test_per_class_metrics.csv | 8 | 5 | class_or_category: class_name; count: support; metric: precision, recall, support | read |
| experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/test_predictions.csv | 2500 | 11 | class_or_category: class_name, predicted_class, true_class; predicted_label: predicted_class, predicted_id; split: original_split, split | read |
| experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/validation_confusion_matrix.csv | 8 | 9 |  | read |
| experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/validation_evaluation.json | 0 | 59 | count: classification_report.0_NOR.support, classification_report.1_F&S.support, classification_report.2_SD.support, classification_report.3_MY.support, classification_report.4_AP.support, classification_report.5_BN.support, classification_report.6_UN.support, classification_report.7_IM.support, classification_report.macro avg.support, classification_report.weighted avg.support; metric: classification_report.0_NOR.precision, classification_report.0_NOR.recall, classification_report.0_NOR.support, classification_report.1_F&S.precision, classification_report.1_F&S.recall, classification_report.1_F&S.support, classification_report.2_SD.precision, classification_report.2_SD.recall, classification_report.2_SD.support, classification_report.3_MY.precision, classification_report.3_MY.recall, classification_report.3_MY.support, classification_report.4_AP.precision, classification_report.4_AP.recall, classification_report.4_AP.support, classification_report.5_BN.precision, classification_report.5_BN.recall, classification_report.5_BN.support, classification_report.6_UN.precision, classification_report.6_UN.recall, classification_report.6_UN.support, classification_report.7_IM.precision, classification_report.7_IM.recall, classification_report.7_IM.support, classification_report.accuracy, classification_report.macro avg.precision, classification_report.macro avg.recall, classification_report.macro avg.support, classification_report.weighted avg.precision, classification_report.weighted avg.recall, classification_report.weighted avg.support, metrics.accuracy, metrics.balanced_accuracy, metrics.macro_f1; split: split | keys-only |
| experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/validation_per_class_metrics.csv | 8 | 5 | class_or_category: class_name; count: support; metric: precision, recall, support | read |
| experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/validation_predictions.csv | 2500 | 11 | class_or_category: class_name, predicted_class, true_class; predicted_label: predicted_class, predicted_id; split: original_split, split | read |
| experiments/results/rice/rice_fedavg_iid_mobilenetv2/metrics.json | 5 | 7 | metric: validation.accuracy, validation.balanced_accuracy, validation.macro_f1; round_or_epoch: round; runtime: elapsed_seconds_from_start | read |
| experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/evaluation_metrics.json | 0 | 53 | class_or_category: class_names, results.test.per_class, results.validation.per_class; count: results.test.num_images, results.test.per_class.support, results.validation.num_images, results.validation.per_class.support; metric: checkpoint_metadata.validation_metrics.accuracy, checkpoint_metadata.validation_metrics.balanced_accuracy, checkpoint_metadata.validation_metrics.macro_f1, results.test.accuracy, results.test.balanced_accuracy, results.test.evaluation_seconds, results.test.macro_f1, results.test.per_class.precision, results.test.per_class.recall, results.test.per_class.support, results.validation.accuracy, results.validation.balanced_accuracy, results.validation.evaluation_seconds, results.validation.macro_f1, results.validation.per_class.precision, results.validation.per_class.recall, results.validation.per_class.support; round_or_epoch: checkpoint_metadata.round; runtime: results.test.evaluation_seconds, results.validation.evaluation_seconds; seed: checkpoint_metadata.arguments.seed | keys-only |
| experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/evaluation_summary.csv | 2 | 8 | count: num_images; metric: accuracy, balanced_accuracy, evaluation_seconds, macro_f1; runtime: evaluation_seconds; split: split | read |
| experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/test_confusion_matrix.csv | 8 | 9 |  | read |
| experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/test_per_class_metrics.csv | 8 | 5 | class_or_category: class_name; count: support; metric: precision, recall, support | read |
| experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/test_predictions.csv | 2500 | 19 | class_or_category: class_name; predicted_label: predicted_class_id, predicted_class_name; split: original_split, split | read |
| experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/validation_confusion_matrix.csv | 8 | 9 |  | read |
| experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/validation_per_class_metrics.csv | 8 | 5 | class_or_category: class_name; count: support; metric: precision, recall, support | read |
| experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/validation_predictions.csv | 2500 | 19 | class_or_category: class_name; predicted_label: predicted_class_id, predicted_class_name; split: original_split, split | read |
| experiments/results/rice/rice_fedavg_iid_resnet18/metrics.json | 5 | 7 | metric: validation.accuracy, validation.balanced_accuracy, validation.macro_f1; round_or_epoch: round; runtime: elapsed_seconds_from_start | read |
| experiments/results/rice/rice_fedavg_noniid_mobilenetv2/evaluation/evaluation_metrics.json | 0 | 13 | metric: results.test.accuracy, results.test.balanced_accuracy, results.test.macro_f1, results.validation.accuracy, results.validation.balanced_accuracy, results.validation.macro_f1 | keys-only |
| … 209 more rows |  |  |  |  |

## Existing recognized charts and figures

| relative_path | inferred_visual_type | width | height | metadata_status |
| --- | --- | --- | --- | --- |
| experiments/results/rice/rice_ddp_resnet18/evaluation/test_confusion_matrix.png | confusion_matrix | 1790 | 1578 | read |
| experiments/results/rice/rice_ddp_resnet18/evaluation/validation_confusion_matrix.png | confusion_matrix | 1790 | 1578 | read |
| experiments/results/rice/rice_efficientnetb0_grouped_v1/confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| experiments/results/rice/rice_efficientnetb0_grouped_v1/test_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| experiments/results/rice/rice_efficientnetb0_grouped_v1/validation_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/test_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/validation_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/test_confusion_matrix.png | confusion_matrix | 1790 | 1578 | read |
| experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/validation_confusion_matrix.png | confusion_matrix | 1790 | 1578 | read |
| experiments/results/rice/rice_fedavg_noniid_mobilenetv2/evaluation/test_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| experiments/results/rice/rice_fedavg_noniid_mobilenetv2/evaluation/validation_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| experiments/results/rice/rice_fedavg_noniid_resnet18/evaluation/test_confusion_matrix.png | confusion_matrix | 1790 | 1578 | read |
| experiments/results/rice/rice_fedavg_noniid_resnet18/evaluation/validation_confusion_matrix.png | confusion_matrix | 1790 | 1578 | read |
| experiments/results/rice/rice_mobilenetv2_grouped_v1/confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| experiments/results/rice/rice_mobilenetv2_grouped_v1/test_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| experiments/results/rice/rice_mobilenetv2_grouped_v1/validation_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| experiments/results/rice/rice_resnet18_grouped_v1/confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| experiments/results/rice/rice_resnet18_grouped_v1/test_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| experiments/results/rice/rice_resnet18_grouped_v1/validation_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| experiments/results/wheat/wheat_resnet18_grouped_v2/test_07_confusion_matrix.png | confusion_matrix | 1638 | 1417 | read |
| experiments/results/wheat/wheat_resnet18_grouped_v2/test_confusion_matrix.png | confusion_matrix | 1638 | 1417 | read |
| experiments/results/wheat/wheat_resnet18_grouped_v2/validation_confusion_matrix.png | confusion_matrix | 1638 | 1417 | read |
| experiments/results/wheat/wheat_resnet18_grouped_v3_sqrt_weights/test_07_confusion_matrix.png | confusion_matrix | 1638 | 1417 | read |
| experiments/results/wheat/wheat_resnet18_grouped_v3_sqrt_weights/test_confusion_matrix.png | confusion_matrix | 1638 | 1417 | read |
| experiments/results/wheat/wheat_resnet18_grouped_v3_sqrt_weights/validation_confusion_matrix.png | confusion_matrix | 1638 | 1417 | read |
| paper/figures/wheat_v3_test07_confusion_matrix_normalized.png | confusion_matrix | 2154 | 1853 | read |
| results/Rice/Centralized/EfficientNetB0/seed42/confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Centralized/EfficientNetB0/seed42/test_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Centralized/EfficientNetB0/seed42/validation_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Centralized/MobileNetV2/seed42/confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Centralized/MobileNetV2/seed42/test_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Centralized/MobileNetV2/seed42/validation_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Centralized/ResNet18/seed123/confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Centralized/ResNet18/seed2026/confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Centralized/ResNet18/seed42/confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Centralized/ResNet18/seed42/test_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Centralized/ResNet18/seed42/validation_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/DDP/ResNet18/Final/evaluation/test_confusion_matrix.png | confusion_matrix | 1790 | 1578 | read |
| results/Rice/DDP/ResNet18/Final/evaluation/validation_confusion_matrix.png | confusion_matrix | 1790 | 1578 | read |
| results/Rice/Federated/FedAvg/IID/MobileNetV2/evaluation/test_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Federated/FedAvg/IID/MobileNetV2/evaluation/validation_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Federated/FedAvg/IID/ResNet18/evaluation/test_confusion_matrix.png | confusion_matrix | 1790 | 1578 | read |
| results/Rice/Federated/FedAvg/IID/ResNet18/evaluation/validation_confusion_matrix.png | confusion_matrix | 1790 | 1578 | read |
| results/Rice/Federated/FedAvg/NonIID_alpha0p1/MobileNetV2/evaluation/test_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Federated/FedAvg/NonIID_alpha0p1/MobileNetV2/evaluation/validation_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Federated/FedAvg/NonIID_alpha0p5/MobileNetV2/evaluation/test_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Federated/FedAvg/NonIID_alpha0p5/MobileNetV2/evaluation/validation_confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| results/Rice/Federated/FedAvg/NonIID_alpha0p5/ResNet18/evaluation/test_confusion_matrix.png | confusion_matrix | 1790 | 1578 | read |
| results/Rice/Federated/FedAvg/NonIID_alpha0p5/ResNet18/evaluation/validation_confusion_matrix.png | confusion_matrix | 1790 | 1578 | read |
| results/Rice/SupportingEvidence/ResNet18Proof/confusion_matrix.png | confusion_matrix | 1610 | 1417 | read |
| … 7 more rows |  |  |  |  |

## Highest-priority chart opportunities

| graph_name | chart_type | source_file | status | detected_columns_or_asset | suggested_output_filename |
| --- | --- | --- | --- | --- | --- |
| Accuracy by round/epoch | line | streamlit-ui/streamlit-ui/data/experiment_results.csv | Ready from structured data | distribution, method, selected_round, accuracy | experiment_results_accuracy_convergence.png |
| Accuracy by round/epoch | line | streamlit-ui/streamlit-ui/data/round_metrics.csv | Ready from structured data | distribution, method, round, accuracy | round_metrics_accuracy_convergence.png |
| Accuracy comparison across experiments | grouped horizontal bar | experiments/results/tables/rice_architecture_evaluation.csv | Ready from structured data | model, accuracy | rice_architecture_evaluation_accuracy_comparison.png |
| Accuracy comparison across experiments | grouped horizontal bar | results/Reports/FinalReportArchive/tables/rice_architecture_evaluation.csv | Ready from structured data | model, accuracy | rice_architecture_evaluation_accuracy_comparison.png |
| Accuracy comparison across experiments | grouped horizontal bar | results/Rice/Federated/Summaries/Presentation_tables/table_1_alpha0p5_comprehensive.csv | Ready from structured data | Approach, Backbone, Accuracy | table_1_alpha0p5_comprehensive_accuracy_comparison.png |
| Accuracy comparison across experiments | grouped horizontal bar | streamlit-ui/streamlit-ui/data/experiment_results.csv | Ready from structured data | architecture, dataset, distribution, method, accuracy | experiment_results_accuracy_comparison.png |
| Accuracy comparison across experiments | grouped horizontal bar | streamlit-ui/streamlit-ui/data/round_metrics.csv | Ready from structured data | architecture, dataset, distribution, method, accuracy | round_metrics_accuracy_comparison.png |
| Class distribution across federated clients | 100% stacked bar or heatmap | streamlit-ui/streamlit-ui/data/client_statistics.csv | Ready from structured data | alpha, class_entropy, client, images | client_statistics_client_class_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/provenance/wheat_grouped_split_v2/class_allocation.csv | Ready from structured data | label, total_images, validation_images | class_allocation_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_ddp_resnet18/evaluation/evaluation_metrics.json | Ready from structured data | class_names, results.test.num_images, results.test.per_class, results.test.per_class.support, results.validation.num_images, results.validation.per_class, results.validation.per_class.support | evaluation_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_ddp_resnet18/evaluation/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_ddp_resnet18/evaluation/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_efficientnetb0_grouped_v1/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_efficientnetb0_grouped_v1/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/evaluation_metrics.json | Ready from structured data | class_names, results.test.num_images, results.test.per_class, results.test.per_class.support, results.validation.num_images, results.validation.per_class, results.validation.per_class.support | evaluation_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_fedavg_noniid_mobilenetv2/evaluation/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_fedavg_noniid_mobilenetv2/evaluation/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_fedavg_noniid_resnet18/evaluation/evaluation_metrics.json | Ready from structured data | class_names, results.test.num_images, results.test.per_class, results.test.per_class.support, results.validation.num_images, results.validation.per_class, results.validation.per_class.support | evaluation_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_fedavg_noniid_resnet18/evaluation/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_fedavg_noniid_resnet18/evaluation/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_mobilenetv2_grouped_v1/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_mobilenetv2_grouped_v1/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_resnet18_grouped_v1/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/rice/rice_resnet18_grouped_v1/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/wheat/wheat_resnet18_grouped_v2/test_07_per_class_metrics.csv | Ready from structured data | class_name, support | test_07_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/wheat/wheat_resnet18_grouped_v2/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/wheat/wheat_resnet18_grouped_v2/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/wheat/wheat_resnet18_grouped_v3_sqrt_weights/test_07_per_class_metrics.csv | Ready from structured data | class_name, support | test_07_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/wheat/wheat_resnet18_grouped_v3_sqrt_weights/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | experiments/results/wheat/wheat_resnet18_grouped_v3_sqrt_weights/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Centralized/EfficientNetB0/seed42/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Centralized/EfficientNetB0/seed42/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Centralized/MobileNetV2/seed42/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Centralized/MobileNetV2/seed42/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Centralized/ResNet18/seed42/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Centralized/ResNet18/seed42/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/DDP/ResNet18/Final/evaluation/evaluation_metrics.json | Ready from structured data | class_names, results.test.num_images, results.test.per_class, results.test.per_class.support, results.validation.num_images, results.validation.per_class, results.validation.per_class.support | evaluation_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/DDP/ResNet18/Final/evaluation/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/DDP/ResNet18/Final/evaluation/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Federated/FedAvg/IID/MobileNetV2/evaluation/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Federated/FedAvg/IID/MobileNetV2/evaluation/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Federated/FedAvg/IID/ResNet18/evaluation/evaluation_metrics.json | Ready from structured data | class_names, results.test.num_images, results.test.per_class, results.test.per_class.support, results.validation.num_images, results.validation.per_class, results.validation.per_class.support | evaluation_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Federated/FedAvg/IID/ResNet18/evaluation/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Federated/FedAvg/IID/ResNet18/evaluation/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Federated/FedAvg/NonIID_alpha0p1/MobileNetV2/evaluation/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Federated/FedAvg/NonIID_alpha0p1/MobileNetV2/evaluation/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Federated/FedAvg/NonIID_alpha0p5/MobileNetV2/evaluation/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Federated/FedAvg/NonIID_alpha0p5/MobileNetV2/evaluation/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Federated/FedAvg/NonIID_alpha0p5/ResNet18/evaluation/evaluation_metrics.json | Ready from structured data | class_names, results.test.num_images, results.test.per_class, results.test.per_class.support, results.validation.num_images, results.validation.per_class, results.validation.per_class.support | evaluation_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Federated/FedAvg/NonIID_alpha0p5/ResNet18/evaluation/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Rice/Federated/FedAvg/NonIID_alpha0p5/ResNet18/evaluation/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Wheat/Centralized/ResNet18/GroupedV2_FullInverseWeights/test_07_per_class_metrics.csv | Ready from structured data | class_name, support | test_07_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Wheat/Centralized/ResNet18/GroupedV2_FullInverseWeights/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Wheat/Centralized/ResNet18/GroupedV2_FullInverseWeights/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Wheat/Centralized/ResNet18/GroupedV3_SqrtWeights/test_07_per_class_metrics.csv | Ready from structured data | class_name, support | test_07_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Wheat/Centralized/ResNet18/GroupedV3_SqrtWeights/test_per_class_metrics.csv | Ready from structured data | class_name, support | test_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | results/Wheat/Centralized/ResNet18/GroupedV3_SqrtWeights/validation_per_class_metrics.csv | Ready from structured data | class_name, support | validation_per_class_metrics_category_distribution.png |
| Dataset category distribution | horizontal bar | streamlit-ui/streamlit-ui/data/dataset_categories.csv | Ready from structured data | category, count, dataset | dataset_categories_category_distribution.png |
| Dirichlet alpha vs Accuracy | line with markers | streamlit-ui/streamlit-ui/data/experiment_results.csv | Ready if multiple alpha values exist | alpha, client, method, accuracy | experiment_results_alpha_vs_accuracy.png |
| Dirichlet alpha vs Accuracy | line with markers | streamlit-ui/streamlit-ui/data/round_metrics.csv | Ready if multiple alpha values exist | alpha, client, method, accuracy | round_metrics_alpha_vs_accuracy.png |
| Dirichlet alpha vs Loss | line with markers | streamlit-ui/streamlit-ui/data/round_metrics.csv | Ready if multiple alpha values exist | alpha, client, method, loss | round_metrics_alpha_vs_loss.png |
| Dirichlet alpha vs Macro F1 | line with markers | streamlit-ui/streamlit-ui/data/experiment_results.csv | Ready if multiple alpha values exist | alpha, client, method, macro_f1 | experiment_results_alpha_vs_macro_f1.png |
| Dirichlet alpha vs Macro F1 | line with markers | streamlit-ui/streamlit-ui/data/round_metrics.csv | Ready if multiple alpha values exist | alpha, client, method, macro_f1 | round_metrics_alpha_vs_macro_f1.png |
| Display existing Confusion Matrix | existing saved image | experiments/results/rice/rice_ddp_resnet18/evaluation/test_confusion_matrix.png | Existing asset; no regeneration required | experiments/results/rice/rice_ddp_resnet18/evaluation/test_confusion_matrix.png | test_confusion_matrix.png |
| Display existing Confusion Matrix | existing saved image | experiments/results/rice/rice_ddp_resnet18/evaluation/validation_confusion_matrix.png | Existing asset; no regeneration required | experiments/results/rice/rice_ddp_resnet18/evaluation/validation_confusion_matrix.png | validation_confusion_matrix.png |
| Display existing Confusion Matrix | existing saved image | experiments/results/rice/rice_efficientnetb0_grouped_v1/confusion_matrix.png | Existing asset; no regeneration required | experiments/results/rice/rice_efficientnetb0_grouped_v1/confusion_matrix.png | confusion_matrix.png |
| Display existing Confusion Matrix | existing saved image | experiments/results/rice/rice_efficientnetb0_grouped_v1/test_confusion_matrix.png | Existing asset; no regeneration required | experiments/results/rice/rice_efficientnetb0_grouped_v1/test_confusion_matrix.png | test_confusion_matrix.png |
| Display existing Confusion Matrix | existing saved image | experiments/results/rice/rice_efficientnetb0_grouped_v1/validation_confusion_matrix.png | Existing asset; no regeneration required | experiments/results/rice/rice_efficientnetb0_grouped_v1/validation_confusion_matrix.png | validation_confusion_matrix.png |
| Display existing Confusion Matrix | existing saved image | experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/test_confusion_matrix.png | Existing asset; no regeneration required | experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/test_confusion_matrix.png | test_confusion_matrix.png |
| Display existing Confusion Matrix | existing saved image | experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/validation_confusion_matrix.png | Existing asset; no regeneration required | experiments/results/rice/rice_fedavg_iid_mobilenetv2/evaluation/validation_confusion_matrix.png | validation_confusion_matrix.png |
| Display existing Confusion Matrix | existing saved image | experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/test_confusion_matrix.png | Existing asset; no regeneration required | experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/test_confusion_matrix.png | test_confusion_matrix.png |
| Display existing Confusion Matrix | existing saved image | experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/validation_confusion_matrix.png | Existing asset; no regeneration required | experiments/results/rice/rice_fedavg_iid_resnet18/evaluation/validation_confusion_matrix.png | validation_confusion_matrix.png |
| Display existing Confusion Matrix | existing saved image | experiments/results/rice/rice_fedavg_noniid_mobilenetv2/evaluation/test_confusion_matrix.png | Existing asset; no regeneration required | experiments/results/rice/rice_fedavg_noniid_mobilenetv2/evaluation/test_confusion_matrix.png | test_confusion_matrix.png |
| Display existing Confusion Matrix | existing saved image | experiments/results/rice/rice_fedavg_noniid_mobilenetv2/evaluation/validation_confusion_matrix.png | Existing asset; no regeneration required | experiments/results/rice/rice_fedavg_noniid_mobilenetv2/evaluation/validation_confusion_matrix.png | validation_confusion_matrix.png |
| Display existing Confusion Matrix | existing saved image | experiments/results/rice/rice_fedavg_noniid_resnet18/evaluation/test_confusion_matrix.png | Existing asset; no regeneration required | experiments/results/rice/rice_fedavg_noniid_resnet18/evaluation/test_confusion_matrix.png | test_confusion_matrix.png |
| Display existing Confusion Matrix | existing saved image | experiments/results/rice/rice_fedavg_noniid_resnet18/evaluation/validation_confusion_matrix.png | Existing asset; no regeneration required | experiments/results/rice/rice_fedavg_noniid_resnet18/evaluation/validation_confusion_matrix.png | validation_confusion_matrix.png |
| … 124 more rows |  |  |  |  |  |

## Recommended dashboard selection order

Use this order when deciding what to display:

1. **Final experiment comparison:** Centralized, FedAvg, FedPer and FedRep using Macro-F1.
2. **Dataset category distribution:** Separate rice and wheat category-count figures.
3. **IID/non-IID client class distribution:** Prefer a 100% stacked bar or heatmap.
4. **Client image counts:** Show how much data each client received.
5. **Round-by-round convergence:** Macro-F1 and loss for FedAvg, FedPer and FedRep.
6. **Confusion matrices:** One selector plus side-by-side comparison mode.
7. **Alpha sensitivity:** Compare α values only when multiple verified α experiments exist.
8. **Seed stability:** Add mean and standard deviation when multiple seeds exist.
9. **Adaptive weights:** Add when the new client-weight logs become available.
10. **Network effects:** Add only when latency/bandwidth/packet-loss data is recorded.

## Important limitations

- A recommendation marked **Ready** means the required column pattern was detected; it does not guarantee every row is complete or scientifically verified.
- A recommendation marked **Ready if...** requires multiple values, such as multiple alpha values or random seeds.
- Text/log files are only keyword-scanned. Their contents may need a dedicated parser before plotting.
- The audit does not load model checkpoints or execute training code.
- Raw dataset images are inventoried, but only a small sample per directory is opened for dimensions to keep the scan practical on WSL-mounted drives.

## Files to share for the next step

Share these two files after running the audit:

- `project_audit/audit_report.md`
- `project_audit/candidate_graphs.csv`

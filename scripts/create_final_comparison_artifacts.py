import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/scratch/project_2019765/grain_research")
INPUT = ROOT / "final_results/final_model_comparison.json"
OUTPUT_DIR = ROOT / "final_results"

with open(INPUT, "r", encoding="utf-8") as handle:
    data = json.load(handle)

name_map = {
    "centralized_resnet18_v3": "Centralized V3",
    "fedavg_iid_round3": "FedAvg IID",
    "fedavg_noniid_round3": "FedAvg non-IID",
    "ddp_resnet18_v3": "DDP (2 GPUs)",
}

rows = []

for internal_name, details in data["results"].items():
    metrics = details["metrics"]

    rows.append({
        "Method": name_map[internal_name],
        "Validation Macro-F1": metrics["validation"]["macro_f1"],
        "Test Set 1 Macro-F1": metrics["test_set_1"]["macro_f1"],
        "Test Set 2 Macro-F1": metrics["test_set_2"]["macro_f1"],
    })

csv_path = OUTPUT_DIR / "final_model_comparison.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

methods = [row["Method"] for row in rows]
validation = [row["Validation Macro-F1"] for row in rows]
test1 = [row["Test Set 1 Macro-F1"] for row in rows]
test2 = [row["Test Set 2 Macro-F1"] for row in rows]

x = list(range(len(methods)))
width = 0.25

plt.figure(figsize=(11, 6))
plt.bar([value - width for value in x], validation, width, label="Validation")
plt.bar(x, test1, width, label="Test Set 1")
plt.bar([value + width for value in x], test2, width, label="Test Set 2")

plt.xticks(x, methods)
plt.ylabel("Macro-F1 (%)")
plt.ylim(0, 100)
plt.title("Final Grain Classification Comparison")
plt.legend()
plt.tight_layout()

chart_path = OUTPUT_DIR / "final_model_comparison.png"
plt.savefig(chart_path, dpi=300)
print("Saved:", csv_path)
print("Saved:", chart_path)

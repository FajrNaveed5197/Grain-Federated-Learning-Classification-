import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import calculate_metrics, format_metrics

class_names = [
    "Black Germ",
    "Broken",
    "Fusarium",
    "Insect",
    "Moldy",
    "Sound",
    "Spotted",
    "Sprouted",
]

y_true = [0, 1, 2, 3, 4, 5, 6, 7]
y_pred = [0, 1, 2, 3, 4, 5, 7, 7]

metrics = calculate_metrics(y_true, y_pred, class_names)

print(format_metrics(metrics))
print("Confusion matrix:")
for row in metrics["confusion_matrix"]:
    print(row)

print("Metrics test passed.")

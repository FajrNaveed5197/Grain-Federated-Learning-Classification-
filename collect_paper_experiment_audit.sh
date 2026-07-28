#!/usr/bin/env bash
#
# Collect experiment history, metrics, configurations, dataset notes,
# Git chronology, failed attempts, and seed results for the paper.
#
# Run from anywhere inside the Git repository:
#   chmod +x collect_paper_experiment_audit.sh
#   ./collect_paper_experiment_audit.sh
#
# Final uploadable output:
#   paper_experiment_audit.tar.gz
#

set -uo pipefail

timestamp="$(date +%Y%m%d_%H%M%S)"

# Locate the repository root.
if repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    cd "$repo_root"
else
    echo "ERROR: Run this script from inside the Git repository."
    exit 1
fi

audit_dir="paper/audit_${timestamp}"
combined_file="paper_experiment_audit_${timestamp}.txt"
archive_file="paper_experiment_audit_${timestamp}.tar.gz"
latest_archive="paper_experiment_audit.tar.gz"
latest_combined="paper_experiment_audit.txt"

mkdir -p "$audit_dir"

# Choose an available Python interpreter.
if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "ERROR: Python is required to read JSON and CSV result files."
    exit 1
fi

section() {
    local title="$1"
    {
        echo
        echo "================================================================================"
        echo "$title"
        echo "================================================================================"
    } | tee -a "$audit_dir/00_run_log.txt"
}

run_capture() {
    local title="$1"
    local output="$2"
    shift 2

    section "$title"
    {
        echo "Command: $*"
        echo
        "$@"
    } >"$output" 2>&1 || true

    echo "Saved: $output"
}

echo "Repository root: $repo_root" | tee "$audit_dir/00_run_log.txt"
echo "Started: $(date --iso-8601=seconds)" | tee -a "$audit_dir/00_run_log.txt"
echo "Python: $PYTHON_BIN" | tee -a "$audit_dir/00_run_log.txt"

# ---------------------------------------------------------------------------
# 00. Repository identity and current state
# ---------------------------------------------------------------------------
section "Repository identity"

{
    echo "Repository root: $repo_root"
    echo "Collection timestamp: $(date --iso-8601=seconds)"
    echo
    echo "Current branch:"
    git branch --show-current 2>/dev/null || true
    echo
    echo "HEAD commit:"
    git rev-parse HEAD 2>/dev/null || true
    echo
    echo "HEAD description:"
    git log -1 --date=iso --pretty=format:'%ad | %H | %s' 2>/dev/null || true
    echo
    echo "Remote URLs:"
    git remote -v 2>/dev/null || true
    echo
    echo "Working-tree status:"
    git status --short 2>/dev/null || true
} >"$audit_dir/00_repository_identity.txt"

# ---------------------------------------------------------------------------
# 01. Result-file inventory
# ---------------------------------------------------------------------------
section "Result-file inventory"

if [[ -d experiments/results ]]; then
    find experiments/results -type f \
        \( -iname "*metrics*.json" \
        -o -iname "*summary*.json" \
        -o -iname "*summary*.csv" \
        -o -iname "*comparison*.csv" \
        -o -iname "*history*.json" \
        -o -iname "*evaluation*.json" \
        -o -iname "*evaluation*.csv" \
        -o -iname "*split*.json" \
        -o -iname "*split*.csv" \
        -o -iname "*confusion*.csv" \
        -o -iname "*per_class*.csv" \
        -o -iname "*predictions*.csv" \) \
        | sort >"$audit_dir/01_result_files.txt"
else
    echo "experiments/results directory not found." >"$audit_dir/01_result_files.txt"
fi

echo "Saved: $audit_dir/01_result_files.txt"

# ---------------------------------------------------------------------------
# 02. Extract metrics from JSON and CSV artifacts
# ---------------------------------------------------------------------------
section "Extract all recorded metrics"

"$PYTHON_BIN" - <<'PY' >"$audit_dir/02_all_metrics.txt" 2>&1
from __future__ import annotations

from pathlib import Path
import csv
import json

root = Path("experiments/results")

wanted_keys = {
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "precision",
    "recall",
    "f1",
    "support",
    "validation_accuracy",
    "validation_balanced_accuracy",
    "validation_macro_f1",
    "validation_weighted_f1",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "best_epoch",
    "best_validation_macro_f1",
    "training_seconds",
    "training_time_seconds",
    "elapsed_seconds",
    "total_seconds",
    "epoch_seconds",
    "wall_clock_seconds",
    "seed",
    "num_clients",
    "num_rounds",
    "local_epochs",
    "dirichlet_alpha",
    "world_size",
    "batch_size",
    "global_batch_size",
    "learning_rate",
    "weight_decay",
}

def search_json(value, prefix=""):
    results = []

    if isinstance(value, dict):
        for key, child in value.items():
            location = f"{prefix}.{key}" if prefix else key

            if key in wanted_keys:
                results.append((location, child))

            results.extend(search_json(child, location))

    elif isinstance(value, list):
        for index, child in enumerate(value):
            results.extend(search_json(child, f"{prefix}[{index}]"))

    return results


if not root.exists():
    print(f"Missing directory: {root}")
    raise SystemExit(0)


for path in sorted(root.rglob("*.json")):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"\nFAILED TO READ JSON: {path}\n{exc}")
        continue

    values = search_json(data)

    if values:
        print("\n" + "=" * 100)
        print(path)
        print("=" * 100)

        for key, value in values:
            print(f"{key}: {value}")


csv_name_tokens = (
    "summary",
    "comparison",
    "evaluation",
    "metrics",
    "seed",
    "efficiency",
    "class",
    "confusion",
    "split",
)

for path in sorted(root.rglob("*.csv")):
    name = path.name.lower()

    if not any(token in name for token in csv_name_tokens):
        continue

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except Exception as exc:
        print(f"\nFAILED TO READ CSV: {path}\n{exc}")
        continue

    print("\n" + "=" * 100)
    print(path)
    print("=" * 100)

    # Limit very large prediction files while retaining useful headers/examples.
    max_rows = 100 if len(rows) > 100 else len(rows)
    for row in rows[:max_rows]:
        print(" | ".join(row))

    if len(rows) > max_rows:
        print(f"... truncated: displayed {max_rows} of {len(rows)} rows")
PY

echo "Saved: $audit_dir/02_all_metrics.txt"

# ---------------------------------------------------------------------------
# 03. Hyperparameters and configurations
# ---------------------------------------------------------------------------
section "Collect hyperparameters and configuration references"

search_roots=()
for candidate in scripts src slurm experiments archive docs README.md README.txt; do
    [[ -e "$candidate" ]] && search_roots+=("$candidate")
done

if ((${#search_roots[@]})); then
    find "${search_roots[@]}" \
        -type f \
        \( -name "*.py" \
        -o -name "*.sh" \
        -o -name "*.json" \
        -o -name "*.yaml" \
        -o -name "*.yml" \
        -o -name "*.md" \
        -o -name "*.txt" \
        -o -name "*.csv" \) \
        -print0 2>/dev/null \
    | xargs -0 -r grep -niE \
        "num_clients|clients|num_rounds|rounds|local_epochs|local epoch|dirichlet|alpha|non.?iid|iid|seed|epochs|warm.?up|fine.?tun|batch.?size|global.?batch|learning.?rate|optimizer|adamw|weight.?decay|scheduler|image.?size|augmentation|class.?weight|inverse|square.?root|sqrt|checkpoint|pretrained|world.?size|nproc|mixed.?precision|amp|loss|cross.?entropy" \
        2>/dev/null \
        >"$audit_dir/03_hyperparameters.txt" || true
else
    echo "No expected source/configuration directories found." \
        >"$audit_dir/03_hyperparameters.txt"
fi

grep -iE \
    "fedavg|flower|num_clients|num_rounds|local_epochs|dirichlet|alpha|non.?iid|partition|seed|checkpoint|initial|client" \
    "$audit_dir/03_hyperparameters.txt" \
    >"$audit_dir/03a_federated_config.txt" || true

grep -iE \
    "ddp|distributed|world.?size|nproc|global.?batch|batch.?size|epochs|training.?time|elapsed|checkpoint|seed|gh200|gpu" \
    "$audit_dir/03_hyperparameters.txt" \
    >"$audit_dir/03b_ddp_config.txt" || true

grep -iE \
    "resnet|mobilenet|efficientnet|warm.?up|fine.?tun|batch.?size|optimizer|adamw|learning.?rate|weight.?decay|scheduler|augmentation|image.?size|class.?weight|cross.?entropy" \
    "$audit_dir/03_hyperparameters.txt" \
    >"$audit_dir/03c_centralized_config.txt" || true

echo "Saved: $audit_dir/03_hyperparameters.txt"
echo "Saved: $audit_dir/03a_federated_config.txt"
echo "Saved: $audit_dir/03b_ddp_config.txt"
echo "Saved: $audit_dir/03c_centralized_config.txt"

# ---------------------------------------------------------------------------
# 04. V1/V2/V3 progression and experiment directory tree
# ---------------------------------------------------------------------------
section "Collect V1/V2/V3 progression"

find . \
    -path "./.git" -prune -o \
    -path "./.venv-paper" -prune -o \
    -path "./.venv" -prune -o \
    -type f \
    \( -name "*.md" \
    -o -name "*.txt" \
    -o -name "*.py" \
    -o -name "*.sh" \
    -o -name "*.json" \
    -o -name "*.csv" \
    -o -name "*.yaml" \
    -o -name "*.yml" \) \
    -print0 2>/dev/null \
| xargs -0 -r grep -niE \
    "v1_first_finetuning|v2_additional_finetuning|v3|version 1|version 2|version 3|additional fine.?tun|continued training|best checkpoint|changed weights|square.?root|full inverse|leakage|group.?aware|legacy" \
    2>/dev/null \
    >"$audit_dir/04_version_progression.txt" || true

if [[ -d experiments ]]; then
    find experiments -maxdepth 8 -type d \
        | sort \
        | grep -Ei \
            "centralized|federated|fedavg|ddp|mobilenet|efficientnet|resnet|v1|v2|v3|rice|wheat" \
        >"$audit_dir/04a_experiment_directories.txt" || true
else
    echo "experiments directory not found." \
        >"$audit_dir/04a_experiment_directories.txt"
fi

echo "Saved: $audit_dir/04_version_progression.txt"
echo "Saved: $audit_dir/04a_experiment_directories.txt"

# ---------------------------------------------------------------------------
# 05. Git chronology
# ---------------------------------------------------------------------------
section "Collect Git chronology"

git log --all --reverse \
    --date=short \
    --pretty=format:'%ad | %h | %s' 2>/dev/null \
| grep -Ei \
    "wheat|rice|resnet|mobilenet|efficientnet|fedavg|federated|flower|ddp|distributed|leak|group|split|weight|seed|evaluation|confusion|mask|ablation|dataset|checkpoint" \
    >"$audit_dir/05_git_chronology.txt" || true

git log --all --reverse \
    --date=iso \
    --name-status \
    --pretty=format:'COMMIT %ad | %h | %s' \
    -- experiments scripts src slurm 2>/dev/null \
| grep -Ei \
    "COMMIT|wheat|rice|fedavg|federated|ddp|resnet|mobilenet|efficientnet|group|split|weight|seed|evaluation|dataset|checkpoint" \
    >"$audit_dir/05a_git_files_chronology.txt" || true

echo "Saved: $audit_dir/05_git_chronology.txt"
echo "Saved: $audit_dir/05a_git_files_chronology.txt"

# ---------------------------------------------------------------------------
# 06. Failed, rejected, legacy, and incomplete experiments
# ---------------------------------------------------------------------------
section "Collect failed, rejected, legacy, and incomplete experiments"

find . \
    -path "./.git" -prune -o \
    -path "./.venv-paper" -prune -o \
    -path "./.venv" -prune -o \
    -type f \
    \( -name "*.md" \
    -o -name "*.txt" \
    -o -name "*.json" \
    -o -name "*.csv" \
    -o -name "*.py" \
    -o -name "*.sh" \
    -o -name "*.yaml" \
    -o -name "*.yml" \) \
    -print0 2>/dev/null \
| xargs -0 -r grep -niE \
    "failed|failure|rejected|invalid|legacy|leakage|overlap|21 groups|924|98 percent|98%|old checkpoint|new rice|transfer|4\.33|3\.42|masked|mask ablation|peer.?to.?peer|p2p|more clients|multiple clients|repeat.*seed|seed 123|seed 2026|planned|future work|todo|not completed|smoke" \
    2>/dev/null \
    >"$audit_dir/06_failed_and_incomplete.txt" || true

find . \
    -path "./.git" -prune -o \
    -path "./.venv-paper" -prune -o \
    -path "./.venv" -prune -o \
    -type f \
    -print0 2>/dev/null \
| xargs -0 -r grep -niE \
    "evaluate_all_final_models|old.*checkpoint|checkpoint.*rice|cross.?dataset|transfer.*rice|4\.33|3\.42" \
    2>/dev/null \
    >"$audit_dir/06a_cross_dataset_transfer.txt" || true

echo "Saved: $audit_dir/06_failed_and_incomplete.txt"
echo "Saved: $audit_dir/06a_cross_dataset_transfer.txt"

# ---------------------------------------------------------------------------
# 07. Dataset sources, class mappings, and Test-07 definition
# ---------------------------------------------------------------------------
section "Collect dataset details and class mappings"

dataset_roots=()
for candidate in README.md README.txt docs scripts src experiments data manifests archive; do
    [[ -e "$candidate" ]] && dataset_roots+=("$candidate")
done

if ((${#dataset_roots[@]})); then
    find "${dataset_roots[@]}" \
        -type f \
        \( -name "*.md" \
        -o -name "*.txt" \
        -o -name "*.py" \
        -o -name "*.json" \
        -o -name "*.csv" \
        -o -name "*.yaml" \
        -o -name "*.yml" \) \
        -print0 2>/dev/null \
    | xargs -0 -r grep -niE \
        "dataset source|data source|downloaded from|provided by|license|citation|doi|rice dataset|wheat dataset|class names|class_to_idx|class mapping|label map|0_NOR|1_F.S|F.S|test_07|test-07|Test 07|capture session|capture group|filename prefix|group identifier|NOR|Black Germ|Fusarium|Sprouted" \
        2>/dev/null \
        >"$audit_dir/07_dataset_details.txt" || true

    grep -RniE \
        "0_NOR|1_F.S|2_SD|3_MY|4_AP|5_BN|6_UN|7_IM|Black Germ|Broken|Fusarium|Insect|Moldy|Sound|Spotted|Sprouted" \
        "${dataset_roots[@]}" 2>/dev/null \
        >"$audit_dir/07a_class_mappings.txt" || true
else
    echo "No expected dataset/source directories found." \
        >"$audit_dir/07_dataset_details.txt"
    echo "No expected dataset/source directories found." \
        >"$audit_dir/07a_class_mappings.txt"
fi

echo "Saved: $audit_dir/07_dataset_details.txt"
echo "Saved: $audit_dir/07a_class_mappings.txt"

# ---------------------------------------------------------------------------
# 08. Rice multi-seed experiment
# ---------------------------------------------------------------------------
section "Collect rice seed files and contents"

if [[ -d experiments/results/rice ]]; then
    find experiments/results/rice \
        -type f \
        | grep -Ei "seed|stability|repeat|summary|comparison|resnet18" \
        | sort \
        >"$audit_dir/08_rice_seed_files.txt" || true
else
    echo "experiments/results/rice directory not found." \
        >"$audit_dir/08_rice_seed_files.txt"
fi

: >"$audit_dir/08a_rice_seed_contents.txt"

while IFS= read -r file; do
    [[ -f "$file" ]] || continue

    {
        echo
        echo "================================================================================"
        echo "$file"
        echo "================================================================================"

        case "$file" in
            *.json)
                "$PYTHON_BIN" -m json.tool "$file" 2>/dev/null | sed -n '1,500p'
                ;;
            *.csv)
                sed -n '1,250p' "$file"
                ;;
            *.txt|*.md|*.log)
                sed -n '1,500p' "$file"
                ;;
        esac
    } >>"$audit_dir/08a_rice_seed_contents.txt"
done <"$audit_dir/08_rice_seed_files.txt"

echo "Saved: $audit_dir/08_rice_seed_files.txt"
echo "Saved: $audit_dir/08a_rice_seed_contents.txt"

# ---------------------------------------------------------------------------
# 09. Relevant documentation and notes inventory
# ---------------------------------------------------------------------------
section "Collect documentation and notes inventory"

find . \
    -path "./.git" -prune -o \
    -path "./.venv-paper" -prune -o \
    -path "./.venv" -prune -o \
    -type f \
    \( -name "*.md" \
    -o -name "*.txt" \
    -o -name "*.tex" \
    -o -name "*.rst" \) \
    -print 2>/dev/null \
| sort \
| grep -Ei \
    "readme|result|experiment|paper|rice|wheat|fed|ddp|distributed|summary|note|report|comparison|method" \
    >"$audit_dir/09_documentation_inventory.txt" || true

echo "Saved: $audit_dir/09_documentation_inventory.txt"

# ---------------------------------------------------------------------------
# 10. Compact directory tree
# ---------------------------------------------------------------------------
section "Collect compact repository tree"

{
    echo "Top-level:"
    find . -maxdepth 2 \
        -path "./.git" -prune -o \
        -path "./.venv-paper" -prune -o \
        -path "./.venv" -prune -o \
        -print 2>/dev/null | sort

    echo
    echo "Experiment results:"
    if [[ -d experiments/results ]]; then
        find experiments/results -maxdepth 5 -print 2>/dev/null | sort
    fi
} >"$audit_dir/10_repository_tree.txt"

echo "Saved: $audit_dir/10_repository_tree.txt"

# ---------------------------------------------------------------------------
# 11. Create a single combined text report
# ---------------------------------------------------------------------------
section "Create combined text report"

{
    echo "PAPER EXPERIMENT AUDIT"
    echo "Repository: $repo_root"
    echo "Generated: $(date --iso-8601=seconds)"
    echo

    for file in "$audit_dir"/*.txt; do
        echo
        echo
        echo "################################################################################"
        echo "FILE: $file"
        echo "################################################################################"
        cat "$file"
    done
} >"$combined_file"

cp "$combined_file" "$latest_combined"

echo "Saved: $combined_file"
echo "Saved convenience copy: $latest_combined"

# ---------------------------------------------------------------------------
# 12. Package everything for upload
# ---------------------------------------------------------------------------
section "Create uploadable archive"

tar -czf "$archive_file" "$audit_dir" "$combined_file"
cp "$archive_file" "$latest_archive"

echo "Saved: $archive_file"
echo "Saved convenience copy: $latest_archive"

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
{
    echo
    echo "================================================================================"
    echo "AUDIT COMPLETE"
    echo "================================================================================"
    echo "Repository: $repo_root"
    echo "Audit directory: $audit_dir"
    echo "Combined report: $combined_file"
    echo "Uploadable archive: $archive_file"
    echo
    echo "Convenience files:"
    echo "  $latest_combined"
    echo "  $latest_archive"
    echo
    echo "Upload this file to ChatGPT:"
    echo "  $latest_archive"
    echo
    echo "File sizes:"
    ls -lh "$combined_file" "$archive_file" "$latest_combined" "$latest_archive" 2>/dev/null || true
    echo
    echo "Completed: $(date --iso-8601=seconds)"
} | tee -a "$audit_dir/00_run_log.txt"

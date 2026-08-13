from __future__ import annotations

import base64
import html
import io
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st


APP_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    """
    Locate the real repository root even when the Streamlit app is nested
    inside streamlit-ui/streamlit-ui.
    """
    candidates = [start, *start.parents]

    for candidate in candidates:
        if (
            (candidate / "results").exists()
            and (candidate / "experiments").exists()
            and (candidate / "src").exists()
        ):
            return candidate

    # Safe fallback for unusual copies of the UI.
    return start.parent


PROJECT_ROOT = find_project_root(APP_DIR)
REPO_ROOT = PROJECT_ROOT

DATA_DIR = APP_DIR / "data"
ASSET_DIR = APP_DIR / "assets"
SAMPLE_DIR = ASSET_DIR / "dataset_samples"
CHART_DIR = ASSET_DIR / "generated_charts"
CM_DIR = ASSET_DIR / "confusion_matrices"

FINAL_RESULTS_TABLE = (
    PROJECT_ROOT
    / "results"
    / "Rice"
    / "Federated"
    / "Summaries"
    / "Presentation_tables"
    / "table_1_alpha0p5_comprehensive.csv"
)

CENTRALIZED_ARCHITECTURE_TABLE = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "tables"
    / "rice_architecture_evaluation.csv"
)

EXPERIMENTS = DATA_DIR / "experiment_results.csv"
CATEGORIES = DATA_DIR / "dataset_categories.csv"
SPLITS = DATA_DIR / "dataset_splits.csv"
CLIENTS = DATA_DIR / "client_statistics.csv"
METADATA = DATA_DIR / "project_metadata.json"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CM_MARKERS = ("confusion", "conf_matrix", "conf-matrix", "confmatrix", "_cm", "cm_")

st.set_page_config(
    page_title="Federated Grain Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.html("""
    <style>
      :root {
        --green:#245C3A; --dark:#173D28; --gold:#D0A646;
        --bg:#F6F7F2; --card:#FFFFFF; --border:#DDE5DA; --muted:#68756C;
      }
      .stApp { background:var(--bg); }
      [data-testid="stSidebar"] { background:var(--dark); }
      [data-testid="stSidebar"] * { color:#F7F8F4; }
      .block-container { max-width:1420px; padding-top:1.8rem; padding-bottom:3rem; }
      .hero {
        background:linear-gradient(120deg,var(--green),var(--dark));
        color:white; padding:1.7rem 2rem; border-radius:1.1rem; margin-bottom:1.2rem;
      }
      .hero small { color:#E7D6A4; font-weight:700; letter-spacing:.09em; text-transform:uppercase; }
      .hero h1 { color:white; margin:.45rem 0 .55rem; line-height:1.05; font-size:clamp(1.8rem,4vw,3rem); }
      .hero p { color:rgba(255,255,255,.82); max-width:850px; margin:0; }
      .eyebrow { color:var(--green); font-weight:700; letter-spacing:.1em; text-transform:uppercase; font-size:.78rem; }
      .page-title { color:var(--dark); margin:.15rem 0 0; font-size:clamp(2rem,4vw,3rem); }
      .page-copy { color:var(--muted); max-width:900px; margin:.5rem 0 1.25rem; }
      .simple-card { background:white; border:1px solid var(--border); border-radius:1rem; padding:1.1rem; height:100%; }
      [data-testid="stMetric"] { background:white; border:1px solid var(--border); border-radius:.95rem; padding:.9rem 1rem; }
      [data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:.8rem; overflow:hidden; }
      .gallery-shell { overflow:hidden; background:white; border:1px solid var(--border); border-radius:1rem; padding:.8rem 0; }
      .gallery-track { display:flex; gap:.75rem; width:max-content; padding-left:.8rem; animation:scroll 35s linear infinite; }
      .gallery-shell:hover .gallery-track { animation-play-state:paused; }
      .gallery-card { width:200px; flex:0 0 200px; overflow:hidden; border-radius:.8rem; border:1px solid #E5EAE2; background:#FAFBF8; }
      .gallery-card img { width:100%; height:140px; object-fit:cover; display:block; }
      .gallery-card div { padding:.55rem .7rem; font-size:.8rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
      .placeholder { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.75rem; }
      .placeholder div { min-height:125px; display:grid; place-items:center; text-align:center; background:white; border:1px dashed #BAC8B7; border-radius:.8rem; color:var(--green); }
      @keyframes scroll { from{transform:translateX(0)} to{transform:translateX(-50%)} }
      @media (prefers-reduced-motion:reduce){ .gallery-track{animation:none} }
    </style>
    """)


def header(label: str, title: str, copy: str) -> None:
    st.html(
        f'<div class="eyebrow">{html.escape(label)}</div>'
        f'<h1 class="page-title">{html.escape(title)}</h1>'
        f'<p class="page-copy">{html.escape(copy)}</p>'
    )


def read_csv(path: Path, columns: list[str]) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(columns=columns), f"Missing file: {path}"
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        return pd.DataFrame(columns=columns), f"Could not read {path.name}: {exc}"
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        return pd.DataFrame(columns=columns), f"{path.name} is missing: {', '.join(missing)}"
    return frame.dropna(how="all").reset_index(drop=True), None


def strings(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    return frame


def numbers(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def experiments_data() -> tuple[pd.DataFrame, str | None]:
    cols = [
        "dataset","architecture","method","distribution","alpha","client","seed",
        "accuracy","macro_f1","selected_round","status","confusion_matrix_path","notes"
    ]
    frame, error = read_csv(EXPERIMENTS, cols)
    if frame.empty:
        return frame, error
    frame = strings(frame, [
        "dataset","architecture","method","distribution","client",
        "status","confusion_matrix_path","notes"
    ])
    frame = numbers(frame, ["alpha","seed","accuracy","macro_f1","selected_round"])
    frame["client"] = frame["client"].replace("", "Global")
    return frame, error


def categories_data() -> tuple[pd.DataFrame, str | None]:
    cols = ["dataset","category","count","source"]
    frame, error = read_csv(CATEGORIES, cols)
    if not frame.empty:
        frame = strings(frame, ["dataset","category","source"])
        frame = numbers(frame, ["count"])
    return frame, error


def splits_data() -> tuple[pd.DataFrame, str | None]:
    cols = ["dataset","split","count","capture_group_overlap","source"]
    frame, error = read_csv(SPLITS, cols)
    if not frame.empty:
        frame = strings(frame, ["dataset","split","source"])
        frame = numbers(frame, ["count","capture_group_overlap"])
    return frame, error


def clients_data() -> tuple[pd.DataFrame, str | None]:
    cols = ["dataset","distribution","alpha","client","images","capture_groups","class_entropy","source"]
    frame, error = read_csv(CLIENTS, cols)
    if not frame.empty:
        frame = strings(frame, ["dataset","distribution","client","source"])
        frame = numbers(frame, ["alpha","images","capture_groups","class_entropy"])
    return frame, error


def metadata_data() -> dict:
    defaults = {
        "total_images":None,
        "dataset_count":2,
        "federated_clients":3,
        "federated_rounds":5,
        "capture_group_overlap":0,
    }
    if not METADATA.exists():
        return defaults
    try:
        loaded = json.loads(METADATA.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return defaults
    return {**defaults, **loaded} if isinstance(loaded, dict) else defaults



def infer_experiment_fields(path: Path) -> dict[str, object]:
    """Infer experiment metadata from an artifact path."""
    relative = path.as_posix()
    lower = relative.lower()
    parts = [part.lower() for part in path.parts]

    dataset = (
        "Rice"
        if "rice" in parts or "/rice/" in lower
        else "Wheat"
        if "wheat" in parts or "/wheat/" in lower
        else "Unknown"
    )

    if "fedavg" in lower:
        method = "FedAvg"
    elif "fedper" in lower:
        method = "FedPer"
    elif "fedrep" in lower:
        method = "FedRep"
    elif "/ddp/" in lower or "_ddp_" in lower:
        method = "DDP"
    else:
        method = "Centralized"

    if "mobilenetv2" in lower:
        architecture = "MobileNetV2"
    elif "efficientnetb0" in lower:
        architecture = "EfficientNetB0"
    elif "resnet18" in lower:
        architecture = "ResNet18"
    else:
        architecture = "Unknown"

    alpha = None
    alpha_patterns = (
        r"alpha[_=\-]?0[._p]?1",
        r"alpha[_=\-]?0[._p]?5",
    )

    if re.search(alpha_patterns[0], lower):
        alpha = 0.1
    elif re.search(alpha_patterns[1], lower):
        alpha = 0.5

    non_iid = (
        "noniid" in lower
        or "non_iid" in lower
        or "non-iid" in lower
    )

    if method == "Centralized":
        setting = "Centralized"
    elif method == "DDP":
        setting = "Distributed"
    elif non_iid:
        setting = (
            f"Non-IID α={alpha:g}"
            if alpha is not None
            else "Non-IID"
        )
    else:
        setting = "IID"

    return {
        "dataset": dataset,
        "method": method,
        "architecture": architecture,
        "distribution": setting,
        "alpha": alpha,
    }


def recursive_find_number(data: object, wanted_key: str) -> float | None:
    """Find the first numeric value whose normalized key matches wanted_key."""
    wanted = token(wanted_key)

    if isinstance(data, dict):
        for key, value in data.items():
            if token(key) == wanted:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

        for value in data.values():
            result = recursive_find_number(value, wanted_key)
            if result is not None:
                return result

    elif isinstance(data, list):
        for value in data:
            result = recursive_find_number(value, wanted_key)
            if result is not None:
                return result

    return None


def metrics_from_dict(data: object) -> dict[str, float | None]:
    if not isinstance(data, dict):
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
        }

    return {
        "accuracy": recursive_find_number(data, "accuracy"),
        "balanced_accuracy": recursive_find_number(
            data,
            "balanced_accuracy",
        ),
        "macro_f1": recursive_find_number(data, "macro_f1"),
    }


def append_result_row(
    rows: list[dict[str, object]],
    *,
    fields: dict[str, object],
    metrics: dict[str, float | None],
    evaluation_basis: str,
    source: Path,
    client: str = "Global",
) -> None:
    if all(metrics.get(name) is None for name in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    )):
        return

    rows.append(
        {
            "dataset": fields["dataset"],
            "method": fields["method"],
            "architecture": fields["architecture"],
            "distribution": fields["distribution"],
            "alpha": fields["alpha"],
            "client": client,
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "evaluation_basis": evaluation_basis,
            "source": str(source),
        }
    )


def collect_client_metric_dicts(
    data: object,
    path: str = "",
) -> list[tuple[str, dict[str, float | None]]]:
    """
    Collect client-level metric dictionaries from personalized test files.
    This is intentionally conservative and only accepts dictionaries that
    contain Macro-F1 or Accuracy.
    """
    collected: list[tuple[str, dict[str, float | None]]] = []

    if isinstance(data, dict):
        direct = metrics_from_dict(data)
        has_metric = (
            direct["accuracy"] is not None
            or direct["macro_f1"] is not None
        )

        client_value = (
            data.get("client")
            or data.get("client_id")
            or data.get("name")
        )

        path_client_match = re.search(
            r"(client[_\s-]*\d+)",
            path,
            flags=re.IGNORECASE,
        )

        client_name = (
            str(client_value)
            if client_value is not None
            else path_client_match.group(1)
            if path_client_match
            else ""
        )

        if has_metric and client_name:
            collected.append((client_name, direct))

        for key, value in data.items():
            child_path = f"{path}.{key}" if path else str(key)
            collected.extend(
                collect_client_metric_dicts(value, child_path)
            )

    elif isinstance(data, list):
        for index, value in enumerate(data):
            collected.extend(
                collect_client_metric_dicts(
                    value,
                    f"{path}[{index}]",
                )
            )

    return collected


@st.cache_data(show_spinner=False)
def authoritative_results_data(
    project_root_string: str,
) -> pd.DataFrame:
    """
    Build one catalog from the real result artifacts.

    Sources are processed in this order:
    1. Verified presentation summary.
    2. evaluation_summary.csv test rows.
    3. evaluation_metrics.json test dictionaries.
    4. personalized test_metrics.json client results.

    Duplicates are removed without dropping alpha=0.1 or wheat experiments.
    """
    root = Path(project_root_string)
    rows: list[dict[str, object]] = []

    final_table = (
        root
        / "results"
        / "Rice"
        / "Federated"
        / "Summaries"
        / "Presentation_tables"
        / "table_1_alpha0p5_comprehensive.csv"
    )

    if final_table.exists():
        try:
            frame = pd.read_csv(final_table)

            required = {
                "Approach",
                "Data setting",
                "Backbone",
                "Accuracy",
                "Balanced Acc.",
                "Macro-F1",
                "Evaluation basis",
            }

            if required.issubset(frame.columns):
                for _, row in frame.iterrows():
                    setting = str(row["Data setting"]).strip()
                    alpha = (
                        0.1
                        if "0.1" in setting
                        else 0.5
                        if "0.5" in setting
                        else None
                    )
                    client = (
                        "Client mean"
                        if "mean of 3 clients"
                        in str(row["Evaluation basis"]).lower()
                        else "Global"
                    )

                    rows.append(
                        {
                            "dataset": "Rice",
                            "method": str(row["Approach"]).strip(),
                            "architecture": str(row["Backbone"]).strip(),
                            "distribution": setting,
                            "alpha": alpha,
                            "client": client,
                            "accuracy": pd.to_numeric(
                                row["Accuracy"],
                                errors="coerce",
                            ),
                            "balanced_accuracy": pd.to_numeric(
                                row["Balanced Acc."],
                                errors="coerce",
                            ),
                            "macro_f1": pd.to_numeric(
                                row["Macro-F1"],
                                errors="coerce",
                            ),
                            "evaluation_basis": str(
                                row["Evaluation basis"]
                            ).strip(),
                            "source": str(final_table),
                        }
                    )
        except (OSError, UnicodeError, pd.errors.ParserError):
            pass

    search_roots = [
        root / "results",
        root / "experiments" / "results",
    ]

    for search_root in search_roots:
        if not search_root.exists():
            continue

        # ---------------------------------------------------------------
        # Standard split-level evaluation summaries
        # ---------------------------------------------------------------
        for path in search_root.rglob("evaluation_summary.csv"):
            try:
                frame = pd.read_csv(path)
            except (OSError, UnicodeError, pd.errors.ParserError):
                continue

            fields = infer_experiment_fields(path)
            split_column = (
                "split"
                if "split" in frame.columns
                else None
            )

            selected = frame.copy()

            if split_column:
                split_values = (
                    selected[split_column]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                )
                test_mask = split_values.isin(
                    {"test", "test_07", "test07"}
                )
                if test_mask.any():
                    selected = selected[test_mask]

            for _, row in selected.iterrows():
                metrics = {
                    "accuracy": pd.to_numeric(
                        row.get("accuracy"),
                        errors="coerce",
                    ),
                    "balanced_accuracy": pd.to_numeric(
                        row.get("balanced_accuracy"),
                        errors="coerce",
                    ),
                    "macro_f1": pd.to_numeric(
                        row.get("macro_f1"),
                        errors="coerce",
                    ),
                }

                basis = (
                    str(row.get(split_column, "Test")).strip()
                    if split_column
                    else "Test"
                )

                append_result_row(
                    rows,
                    fields=fields,
                    metrics=metrics,
                    evaluation_basis=basis,
                    source=path,
                )

        # ---------------------------------------------------------------
        # Standard JSON evaluation artifacts
        # ---------------------------------------------------------------
        for path in search_root.rglob("evaluation_metrics.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue

            fields = infer_experiment_fields(path)
            test_data = None

            if isinstance(data, dict):
                results = data.get("results")
                if isinstance(results, dict):
                    test_data = results.get("test")

                if test_data is None:
                    test_data = data.get("test")

            if isinstance(test_data, dict):
                append_result_row(
                    rows,
                    fields=fields,
                    metrics=metrics_from_dict(test_data),
                    evaluation_basis="Test",
                    source=path,
                )

        # ---------------------------------------------------------------
        # Personalized FedPer/FedRep test results
        # ---------------------------------------------------------------
        for path in search_root.rglob("test_metrics.json"):
            lower = path.as_posix().lower()

            if "fedper" not in lower and "fedrep" not in lower:
                continue

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue

            fields = infer_experiment_fields(path)
            client_rows = collect_client_metric_dicts(data)

            # De-duplicate recursive matches for the same client and values.
            unique_clients: dict[
                tuple[str, float | None, float | None],
                tuple[str, dict[str, float | None]],
            ] = {}

            for client_name, metrics in client_rows:
                key = (
                    client_name,
                    metrics.get("accuracy"),
                    metrics.get("macro_f1"),
                )
                unique_clients[key] = (client_name, metrics)

            usable = list(unique_clients.values())

            for client_name, metrics in usable:
                append_result_row(
                    rows,
                    fields=fields,
                    metrics=metrics,
                    evaluation_basis="Personalized test client",
                    source=path,
                    client=client_name.replace("_", " ").title(),
                )

            if len(usable) >= 2:
                mean_metrics: dict[str, float | None] = {}

                for metric_name in (
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                ):
                    values = [
                        metrics.get(metric_name)
                        for _, metrics in usable
                        if metrics.get(metric_name) is not None
                    ]
                    mean_metrics[metric_name] = (
                        sum(values) / len(values)
                        if values
                        else None
                    )

                append_result_row(
                    rows,
                    fields=fields,
                    metrics=mean_metrics,
                    evaluation_basis=(
                        f"Mean of {len(usable)} personalized clients"
                    ),
                    source=path,
                    client="Client mean",
                )

    columns = [
        "dataset",
        "method",
        "architecture",
        "distribution",
        "alpha",
        "client",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "evaluation_basis",
        "source",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)

    for column in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "alpha",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in (
        "dataset",
        "method",
        "architecture",
        "distribution",
        "client",
        "evaluation_basis",
        "source",
    ):
        frame[column] = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    frame["_source_priority"] = frame["source"].apply(
        lambda value: (
            0
            if "Presentation_tables" in value
            else 1
            if f"{Path('results')}" in value
            else 2
        )
    )

    frame = frame.sort_values(
        [
            "_source_priority",
            "dataset",
            "method",
            "distribution",
            "architecture",
            "client",
        ]
    )

    dedup_columns = [
        "dataset",
        "method",
        "architecture",
        "distribution",
        "client",
        "evaluation_basis",
    ]

    frame = frame.drop_duplicates(
        subset=dedup_columns,
        keep="first",
    )

    return frame.drop(columns="_source_priority").reset_index(drop=True)


def alpha_from_json_or_path(data: object, path: Path) -> float | None:
    lower = path.as_posix().lower()

    if re.search(r"alpha[_=\-]?0[._p]?1", lower):
        return 0.1
    if re.search(r"alpha[_=\-]?0[._p]?5", lower):
        return 0.5

    value = recursive_find_number(data, "alpha")

    if value in (0.1, 0.5):
        return value

    return value


@st.cache_data(show_spinner=False)
def discovered_client_partitions(
    project_root_string: str,
) -> pd.DataFrame:
    """
    Discover client sample counts from real federated metrics.json files.

    Exact alpha-tagged result folders are preferred. Generic duplicate
    experiment folders are retained only when they add a missing setting.
    """
    root = Path(project_root_string)
    rows: list[dict[str, object]] = []

    search_roots = [
        root / "results",
        root / "experiments" / "results",
    ]

    for search_root in search_roots:
        if not search_root.exists():
            continue

        for path in search_root.rglob("metrics.json"):
            lower = path.as_posix().lower()

            if not any(
                method in lower
                for method in ("fedavg", "fedper", "fedrep")
            ):
                continue

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue

            history = data.get("history") if isinstance(data, dict) else None

            if not isinstance(history, list) or not history:
                continue

            first_round = history[0]

            if not isinstance(first_round, dict):
                continue

            clients = first_round.get("clients")

            if not isinstance(clients, list):
                continue

            fields = infer_experiment_fields(path)
            alpha = alpha_from_json_or_path(data, path)

            # Keep non-IID data identifiable even when an old duplicate
            # folder did not encode alpha in its name.
            if (
                str(fields["distribution"]).startswith("Non-IID")
                and alpha is not None
            ):
                fields["distribution"] = f"Non-IID α={alpha:g}"
                fields["alpha"] = alpha

            for client in clients:
                if not isinstance(client, dict):
                    continue

                name = (
                    client.get("client")
                    or client.get("client_id")
                    or client.get("name")
                )
                samples = (
                    client.get("num_samples")
                    or client.get("images")
                    or client.get("sample_count")
                )

                try:
                    images = int(samples)
                except (TypeError, ValueError):
                    continue

                rows.append(
                    {
                        "dataset": fields["dataset"],
                        "method": fields["method"],
                        "architecture": fields["architecture"],
                        "distribution": fields["distribution"],
                        "alpha": fields["alpha"],
                        "client": str(name).replace("_", " ").title(),
                        "images": images,
                        "source": str(path),
                    }
                )

    columns = [
        "dataset",
        "method",
        "architecture",
        "distribution",
        "alpha",
        "client",
        "images",
        "source",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="coerce")
    frame["images"] = pd.to_numeric(frame["images"], errors="coerce")

    frame["_priority"] = frame["source"].apply(
        lambda value: (
            0
            if "/results/" in value.replace("\\\\", "/")
            and "experiments/results" not in value.replace("\\\\", "/")
            else 1
        )
    )

    frame = frame.sort_values("_priority")

    frame = frame.drop_duplicates(
        subset=[
            "dataset",
            "method",
            "architecture",
            "distribution",
            "alpha",
            "client",
            "images",
        ],
        keep="first",
    )

    return frame.drop(columns="_priority").reset_index(drop=True)


def first_existing_chart(
    filenames: list[str],
    caption: str,
) -> bool:
    for filename in filenames:
        path = CHART_DIR / filename
        if path.exists():
            st.image(path, caption=caption, width="stretch")
            return True

    st.info(
        "No saved chart matches this exact selection yet. "
        "The verified result rows are shown below."
    )
    return False


def metric_display(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.4f}%"
    except (TypeError, ValueError):
        return "—"

def fmt_int(value: object) -> str:
    try:
        return "—" if value is None or pd.isna(value) else f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "—"


def token(value: object) -> str:
    text = str(value).strip().lower().replace("non-iid", "non_iid")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def unique(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame:
        return []
    return sorted(v for v in frame[column].dropna().astype(str).str.strip().unique() if v)


def static_chart(filename: str, caption: str, missing: str | None = None) -> bool:
    path = CHART_DIR / filename
    if not path.exists():
        st.info(missing or f"Static chart `{filename}` is not generated yet.")
        return False
    st.image(path, caption=caption, width="stretch")
    return True


def image_uri(path: Path) -> str:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((500, 350))
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=82)
    except (OSError, UnidentifiedImageError):
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def render_gallery(dataset: str | None = None) -> None:
    files = sorted(
        p for p in SAMPLE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ) if SAMPLE_DIR.exists() else []

    if dataset:
        wanted = dataset.strip().lower()
        files = [
            path
            for path in files
            if wanted in {part.lower() for part in path.parts}
        ]

    rice_code_labels = {
        "0_NOR": "NOR",
        "1_FAS": "F&S",
        "1_F&S": "F&S",
        "2_SD": "SD",
        "3_MY": "MY",
        "4_AP": "AP",
        "5_BN": "BN",
        "6_UN": "UN",
        "7_IM": "IM",
    }

    cards = []

    for path in files[:24]:
        uri = image_uri(path)

        if not uri:
            continue

        stem = path.stem
        path_parts = {part.lower() for part in path.parts}

        if "rice" in path_parts:
            code = stem.removeprefix("rice_")
            display = rice_code_labels.get(code, code)
            label = f"Rice · {display}"

        elif "wheat" in path_parts:
            category = stem.removeprefix("wheat_")
            category = re.sub(r"[_\\-]+", " ", category).title()
            label = f"Wheat · {category}"

        else:
            label = re.sub(r"[_\\-]+", " ", stem).title()

        safe_label = html.escape(label)

        cards.append(
            f'<article class="gallery-card">'
            f'<img src="{uri}" alt="{safe_label}">'
            f'<div>{safe_label}</div>'
            f'</article>'
        )

    if not cards:
        dataset_text = html.escape(dataset or "dataset")
        st.html(
            '<div class="placeholder">'
            f'<div>🌾<br>No {dataset_text} samples found</div>'
            '<div>📁<br>Check dataset_samples folder</div>'
            '</div>'
        )
        st.caption(
            "Expected sample images under "
            "`assets/dataset_samples/rice/` and "
            "`assets/dataset_samples/wheat/`."
        )
        return

    st.html(
        f'<section class="gallery-shell">'
        f'<div class="gallery-track">{"".join(cards + cards)}</div>'
        f'</section>'
    )

def sidebar() -> str:
    st.sidebar.markdown("## 🌾 Grain FL")
    st.sidebar.caption("Research dashboard")
    page = st.sidebar.radio(
        "Navigation",
        (
            "Research Overview",
            "Dataset Explorer",
            "Experiment Comparison",
            "Federated Client Analysis",
            "Confusion Matrices",
            "Image Prediction",
            "Data Status",
        ),
    )
    st.sidebar.divider()
    st.sidebar.caption("Charts are generated once and saved as PNG/PDF.")
    return page


def overview() -> None:
    meta = metadata_data()
    results = authoritative_results_data(str(PROJECT_ROOT))

    st.html("""
    <section class="hero">
      <small>Research overview</small>
      <h1>Federated and Distributed Learning for Grain Classification</h1>
      <p>A concise interface for presenting the datasets, implemented methods, verified results and error analysis.</p>
    </section>
    """)

    dataset_count = (
        results["dataset"].replace("", pd.NA).dropna().nunique()
        if not results.empty
        else meta.get("dataset_count")
    )

    cols = st.columns(4)
    cols[0].metric("Images", fmt_int(meta.get("total_images")))
    cols[1].metric("Datasets", fmt_int(dataset_count))
    cols[2].metric("Federated clients", fmt_int(meta.get("federated_clients")))
    cols[3].metric("FL rounds", fmt_int(meta.get("federated_rounds")))

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Project flow")
        st.html(
            '<div class="simple-card"><strong>Dataset → Group-aware split → Training → Evaluation</strong>'
            '<p>Rice and wheat images are studied using centralized, distributed and federated learning.</p></div>'
        )
    with right:
        st.subheader("Implemented methods")
        st.html(
            '<div class="simple-card"><strong>Centralized:</strong> ResNet18, MobileNetV2, EfficientNetB0<br>'
            '<strong>Federated:</strong> FedAvg, FedPer, FedRep<br>'
            '<strong>Distributed:</strong> PyTorch DDP<br>'
            '<strong>Partitions:</strong> IID and non-IID, including α=0.1 and α=0.5</div>'
        )

    # ------------------------------------------------------------------
    # CURATED RESEARCH SNAPSHOT
    # The detailed Experiment Comparison page still keeps the complete
    # automatically discovered experiment catalog.
    # ------------------------------------------------------------------
    st.subheader("Performance snapshot")
    st.caption(
        "A concise test-focused summary of the experiments most useful for "
        "presentation. Detailed runs, individual clients, IID results and "
        "additional evaluation artifacts remain available in Experiment Comparison."
    )

    # ==============================================================
    # RICE
    # ==============================================================
    st.markdown("### 🌾 Rice — Final comparison")

    rice_rows = pd.DataFrame(
        [
            {
                "Method": "Centralized",
                "Architecture": "ResNet18",
                "Setting": "Centralized",
                "α": None,
                "Accuracy (%)": 99.0097,
                "Balanced Acc. (%)": 98.1184,
                "Macro-F1 (%)": 97.9416,
                "Note": "",
                "_highlight": "",
            },
            {
                "Method": "Centralized",
                "Architecture": "MobileNetV2",
                "Setting": "Centralized",
                "α": None,
                "Accuracy (%)": 99.0312,
                "Balanced Acc. (%)": 98.4235,
                "Macro-F1 (%)": 98.0446,
                "Note": "⭐ Best centralized",
                "_highlight": "best",
            },
            {
                "Method": "Centralized",
                "Architecture": "EfficientNetB0",
                "Setting": "Centralized",
                "α": None,
                "Accuracy (%)": 98.9236,
                "Balanced Acc. (%)": 98.2873,
                "Macro-F1 (%)": 97.8079,
                "Note": "",
                "_highlight": "",
            },
            {
                "Method": "DDP",
                "Architecture": "ResNet18",
                "Setting": "Distributed",
                "α": None,
                "Accuracy (%)": 98.9236,
                "Balanced Acc. (%)": 97.9743,
                "Macro-F1 (%)": 97.7568,
                "Note": "",
                "_highlight": "",
            },
            {
                "Method": "FedAvg",
                "Architecture": "MobileNetV2",
                "Setting": "Non-IID",
                "α": 0.1,
                "Accuracy (%)": 99.0527,
                "Balanced Acc. (%)": 98.5727,
                "Macro-F1 (%)": 98.1358,
                "Note": "⭐ Best FedAvg",
                "_highlight": "best",
            },
            {
                "Method": "FedAvg",
                "Architecture": "MobileNetV2",
                "Setting": "Non-IID",
                "α": 0.5,
                "Accuracy (%)": 99.0097,
                "Balanced Acc. (%)": 98.5142,
                "Macro-F1 (%)": 98.0484,
                "Note": "",
                "_highlight": "",
            },
            {
                "Method": "FedPer",
                "Architecture": "MobileNetV2",
                "Setting": "Non-IID",
                "α": 0.1,
                "Accuracy (%)": 99.0815,
                "Balanced Acc. (%)": 98.5258,
                "Macro-F1 (%)": 98.1672,
                "Note": "⭐ Best FedPer · 3-client mean",
                "_highlight": "best",
            },
            {
                "Method": "FedPer",
                "Architecture": "MobileNetV2",
                "Setting": "Non-IID",
                "α": 0.5,
                "Accuracy (%)": 98.9666,
                "Balanced Acc. (%)": 98.4170,
                "Macro-F1 (%)": 97.9198,
                "Note": "3-client mean",
                "_highlight": "",
            },
            {
                "Method": "FedRep",
                "Architecture": "MobileNetV2",
                "Setting": "Non-IID",
                "α": 0.1,
                "Accuracy (%)": 99.0169,
                "Balanced Acc. (%)": 98.3521,
                "Macro-F1 (%)": 98.0299,
                "Note": "3-client mean",
                "_highlight": "",
            },
            {
                "Method": "FedRep",
                "Architecture": "MobileNetV2",
                "Setting": "Non-IID",
                "α": 0.5,
                "Accuracy (%)": 99.1173,
                "Balanced Acc. (%)": 98.4231,
                "Macro-F1 (%)": 98.2075,
                "Note": "🏆 Best overall · 3-client mean",
                "_highlight": "overall",
            },
        ]
    )

    def highlight_rice(row: pd.Series) -> list[str]:
        marker = row.get("_highlight", "")
        if marker == "overall":
            style = (
                "background-color: rgba(208, 166, 70, 0.28); "
                "font-weight: 700;"
            )
        elif marker == "best":
            style = (
                "background-color: rgba(36, 92, 58, 0.12); "
                "font-weight: 600;"
            )
        else:
            style = ""
        return [style] * len(row)

    rice_display = rice_rows.drop(columns=["_highlight"])
    rice_styler = (
        rice_rows.style
        .apply(highlight_rice, axis=1)
        .hide(axis="columns", subset=["_highlight"])
        .format(
            {
                "α": lambda value: "—" if pd.isna(value) else f"{value:.1f}",
                "Accuracy (%)": "{:.4f}",
                "Balanced Acc. (%)": "{:.4f}",
                "Macro-F1 (%)": "{:.4f}",
            }
        )
    )

    st.dataframe(
        rice_styler,
        hide_index=True,
        width="stretch",
        height=390,
    )

    st.caption(
        "Rice centralized architecture rows use the common Seed-42 group-aware "
        "test protocol. FedPer and FedRep values shown here are means across "
        "the three personalized clients."
    )

    # ==============================================================
    # WHEAT
    # ==============================================================
    st.markdown("### 🌾 Wheat — Test-focused experiment summary")

    wheat_rows = pd.DataFrame(
        [
            {
                "Method": "Centralized",
                "Architecture": "MobileNetV2",
                "Setting": "Earlier split",
                "Test Macro-F1 (%)": 80.48,
                "Status / Note": "Historical",
                "_highlight": "",
            },
            {
                "Method": "Centralized",
                "Architecture": "ResNet18",
                "Setting": "Earlier split",
                "Test Macro-F1 (%)": 84.83,
                "Status / Note": "Historical",
                "_highlight": "",
            },
            {
                "Method": "Centralized",
                "Architecture": "ResNet18 V3",
                "Setting": "Earlier split",
                "Test Macro-F1 (%)": 86.03,
                "Status / Note": "⭐ Best historical centralized",
                "_highlight": "historical_best",
            },
            {
                "Method": "Centralized",
                "Architecture": "EfficientNetB0",
                "Setting": "Earlier split",
                "Test Macro-F1 (%)": None,
                "Status / Note": "Historical · exact test metric not verified",
                "_highlight": "",
            },
            {
                "Method": "FedAvg",
                "Architecture": "ResNet18",
                "Setting": "IID · Earlier split",
                "Test Macro-F1 (%)": 81.87,
                "Status / Note": "Historical",
                "_highlight": "",
            },
            {
                "Method": "FedAvg",
                "Architecture": "ResNet18",
                "Setting": "Non-IID · Earlier split",
                "Test Macro-F1 (%)": None,
                "Status / Note": "Historical · only validation Macro-F1 75.22% verified",
                "_highlight": "",
            },
            {
                "Method": "DDP",
                "Architecture": "ResNet18",
                "Setting": "Distributed · Earlier split",
                "Test Macro-F1 (%)": None,
                "Status / Note": "Historical · pipeline validated; test metric not verified",
                "_highlight": "",
            },
            {
                "Method": "Centralized",
                "Architecture": "ResNet18",
                "Setting": "Group-aware · Full inverse",
                "Test Macro-F1 (%)": 85.2522,
                "Status / Note": "Final leakage-free",
                "_highlight": "final",
            },
            {
                "Method": "Centralized",
                "Architecture": "ResNet18",
                "Setting": "Group-aware · Square-root",
                "Test Macro-F1 (%)": 86.7590,
                "Status / Note": "🏆 Best valid Wheat result",
                "_highlight": "overall",
            },
        ]
    )

    def highlight_wheat(row: pd.Series) -> list[str]:
        marker = row.get("_highlight", "")
        if marker == "overall":
            style = (
                "background-color: rgba(208, 166, 70, 0.28); "
                "font-weight: 700;"
            )
        elif marker in {"historical_best", "final"}:
            style = (
                "background-color: rgba(36, 92, 58, 0.12); "
                "font-weight: 600;"
            )
        else:
            style = ""
        return [style] * len(row)

    wheat_styler = (
        wheat_rows.style
        .apply(highlight_wheat, axis=1)
        .hide(axis="columns", subset=["_highlight"])
        .format(
            {
                "Test Macro-F1 (%)": (
                    lambda value: "—"
                    if pd.isna(value)
                    else f"{value:.4f}"
                ),
            }
        )
    )

    st.dataframe(
        wheat_styler,
        hide_index=True,
        width="stretch",
        height=355,
    )

    st.caption(
        "Historical Wheat rows are retained to show the experimental progression, "
        "but they used the earlier split with capture-group leakage and are not "
        "directly comparable with the final zero-overlap group-aware results. "
        "Only main Test results are shown here; Validation and Test-07 remain in "
        "the detailed experiment records."
    )


def dataset_explorer() -> None:
    header(
        "Dataset",
        "Dataset Explorer",
        "Explore representative class samples, verified category composition "
        "and the final leakage-aware dataset partitions.",
    )

    selected = st.selectbox(
        "Dataset",
        ["Rice", "Wheat"],
        key="dataset_explorer_dataset",
    )

    st.subheader(f"{selected} sample images")
    render_gallery(selected)
    st.write("")

    if selected == "Rice":
        rice_split_path = (
            PROJECT_ROOT
            / "experiments"
            / "results"
            / "provenance"
            / "rice_grouped_split"
            / "split_summary.json"
        )

        rice_class_path = (
            PROJECT_ROOT
            / "experiments"
            / "results"
            / "rice"
            / "rice_mobilenetv2_grouped_v1"
            / "test_per_class_metrics.csv"
        )

        rice_class_fallback = (
            PROJECT_ROOT
            / "results"
            / "Rice"
            / "Centralized"
            / "MobileNetV2"
            / "seed42"
            / "test_per_class_metrics.csv"
        )

        train_images = 21606
        validation_images = 4711
        test_images = 4645
        overlap = 0
        seed = 42

        if rice_split_path.exists():
            try:
                split_data = json.loads(
                    rice_split_path.read_text(encoding="utf-8")
                )

                train_images = int(
                    split_data.get("splits", {})
                    .get("train", {})
                    .get("images", train_images)
                )
                validation_images = int(
                    split_data.get("splits", {})
                    .get("validation", {})
                    .get("images", validation_images)
                )
                test_images = int(
                    split_data.get("splits", {})
                    .get("test", {})
                    .get("images", test_images)
                )
                seed = int(split_data.get("seed", seed))
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                pass

        rice_classes = pd.DataFrame(
            [
                ["0_NOR", 3012],
                ["1_F&S", 229],
                ["2_SD", 230],
                ["3_MY", 226],
                ["4_AP", 233],
                ["5_BN", 225],
                ["6_UN", 212],
                ["7_IM", 278],
            ],
            columns=["Class", "Test images"],
        )

        class_source = (
            rice_class_path
            if rice_class_path.exists()
            else rice_class_fallback
        )

        if class_source.exists():
            try:
                candidate = pd.read_csv(class_source)

                if {"class_name", "support"}.issubset(candidate.columns):
                    candidate = candidate[
                        ["class_name", "support"]
                    ].copy()
                    candidate["support"] = pd.to_numeric(
                        candidate["support"],
                        errors="coerce",
                    )
                    candidate = candidate.dropna(
                        subset=["class_name", "support"]
                    )

                    if not candidate.empty:
                        rice_classes = candidate.rename(
                            columns={
                                "class_name": "Class",
                                "support": "Test images",
                            }
                        )
                        rice_classes["Test images"] = (
                            rice_classes["Test images"].astype(int)
                        )
            except (
                OSError,
                UnicodeError,
                pd.errors.ParserError,
            ):
                pass

        total_images = train_images + validation_images + test_images

        metrics = st.columns(4)
        metrics[0].metric("Classification images", f"{total_images:,}")
        metrics[1].metric("Defined classes", f"{rice_classes['Class'].nunique()}")
        metrics[2].metric("Held-out test images", f"{test_images:,}")
        metrics[3].metric("Cross-split overlap", f"{overlap}")

        st.subheader("Class distribution")

        left, right = st.columns([1.2, 0.8])

        with left:
            static_chart(
                "dataset_rice_test_class_distribution.png",
                "Rice held-out test-set class distribution",
                (
                    "The Rice class-distribution chart is missing from "
                    "`assets/generated_charts/`. Re-run the verified "
                    "rice/wheat pie-chart generator."
                ),
            )

        with right:
            st.markdown("#### Rice classes")
            st.dataframe(
                rice_classes,
                hide_index=True,
                width="stretch",
                column_config={
                    "Test images": st.column_config.NumberColumn(
                        "Test images",
                        format="localized",
                    ),
                },
            )
            st.caption(
                "The class counts shown here are from the common "
                "held-out Rice test set used for final evaluation."
            )

        st.subheader("Group-aware data split")

        split_left, split_right = st.columns([1.25, 0.75])

        with split_left:
            static_chart(
                "dataset_rice_group_aware_split.png",
                "Rice train / validation / test split",
                (
                    "The saved Rice split chart is missing from "
                    "`assets/generated_charts/`."
                ),
            )

        with split_right:
            rice_split_table = pd.DataFrame(
                [
                    ["Train", train_images],
                    ["Validation", validation_images],
                    ["Test", test_images],
                ],
                columns=["Partition", "Images"],
            )

            st.dataframe(
                rice_split_table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Images": st.column_config.NumberColumn(
                        "Images",
                        format="localized",
                    ),
                },
            )

            st.markdown(
                f"**Seed:** {seed}  \\n"
                f"**Pairwise capture-group overlap:** {overlap}"
            )

        st.success(
            "The final Rice protocol keeps related capture groups "
            "separated across train, validation and test partitions."
        )

    else:
        wheat_class_path = (
            PROJECT_ROOT
            / "experiments"
            / "results"
            / "provenance"
            / "wheat_grouped_split_v2"
            / "class_allocation.csv"
        )

        wheat_classes = pd.DataFrame(
            [
                ["Black Germ", 2005],
                ["Broken", 59261],
                ["Fusarium", 1932],
                ["Insect", 15569],
                ["Moldy", 3006],
                ["Sound", 193090],
                ["Spotted", 4423],
                ["Sprouted", 6172],
            ],
            columns=["Class", "Development-pool images"],
        )

        if wheat_class_path.exists():
            try:
                candidate = pd.read_csv(wheat_class_path)

                if {"label", "total_images"}.issubset(candidate.columns):
                    candidate = candidate[
                        ["label", "total_images"]
                    ].copy()
                    candidate["total_images"] = pd.to_numeric(
                        candidate["total_images"],
                        errors="coerce",
                    )
                    candidate = candidate.dropna(
                        subset=["label", "total_images"]
                    )

                    if not candidate.empty:
                        wheat_classes = candidate.rename(
                            columns={
                                "label": "Class",
                                "total_images": "Development-pool images",
                            }
                        )
                        wheat_classes[
                            "Development-pool images"
                        ] = wheat_classes[
                            "Development-pool images"
                        ].astype(int)
            except (
                OSError,
                UnicodeError,
                pd.errors.ParserError,
            ):
                pass

        wheat_train = 257069
        wheat_validation = 28389
        wheat_test = 37143
        wheat_test07 = 27144
        development_pool = wheat_train + wheat_validation

        metrics = st.columns(4)
        metrics[0].metric("Development-pool images", f"{development_pool:,}")
        metrics[1].metric("Defined classes", f"{wheat_classes['Class'].nunique()}")
        metrics[2].metric("Main test images", f"{wheat_test:,}")
        metrics[3].metric("Test-07 images", f"{wheat_test07:,}")

        st.subheader("Class distribution")

        left, right = st.columns([1.2, 0.8])

        with left:
            static_chart(
                "dataset_wheat_category_distribution.png",
                "Wheat development-pool category distribution",
                (
                    "The Wheat category-distribution chart is missing from "
                    "`assets/generated_charts/`. Re-run the verified "
                    "rice/wheat pie-chart generator."
                ),
            )

        with right:
            st.markdown("#### Wheat classes")
            st.dataframe(
                wheat_classes.sort_values(
                    "Development-pool images",
                    ascending=False,
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Development-pool images":
                        st.column_config.NumberColumn(
                            "Images",
                            format="localized",
                        ),
                },
            )
            st.caption(
                "These counts represent the final Wheat development pool "
                "(train + validation), not the independent held-out test sets."
            )

        st.subheader("Final leakage-aware split")

        wheat_split_table = pd.DataFrame(
            [
                ["Train", wheat_train, 1448],
                ["Validation", wheat_validation, 253],
                ["Test", wheat_test, 95],
                ["Test-07", wheat_test07, 718],
            ],
            columns=["Partition", "Images", "Capture groups"],
        )

        split_left, split_right = st.columns([1.0, 1.0])

        with split_left:
            st.dataframe(
                wheat_split_table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Images": st.column_config.NumberColumn(
                        "Images",
                        format="localized",
                    ),
                    "Capture groups":
                        st.column_config.NumberColumn(
                            "Capture groups",
                            format="localized",
                        ),
                },
            )

        with split_right:
            st.html(
                """
                <div class="simple-card">
                    <strong>Leakage-aware protocol</strong>
                    <p>
                        Complete Wheat capture groups are retained within
                        partitions. The final split has zero pairwise
                        capture-group overlap.
                    </p>
                    <p>
                        Test and Test-07 are independent held-out
                        evaluation collections and are therefore not
                        included in the development-pool pie chart.
                    </p>
                </div>
                """
            )

        st.success(
            "The final Wheat split corrects the capture-group leakage "
            "identified in the earlier protocol."
        )

# =============================================================================
# COMPLETE CATALOG FOR THE REMAINING PAGES
# Research Overview and Dataset Explorer above are intentionally unchanged.
# =============================================================================


def _catalog_row(dataset, method, architecture, setting, split, *, accuracy=None,
                 balanced_accuracy=None, macro_f1=None, alpha=None, client="Global",
                 protocol="", status="Final", source="Verified project record"):
    return {
        "dataset": dataset, "method": method, "architecture": architecture,
        "setting": setting, "evaluation_split": split, "alpha": alpha,
        "client": client, "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy, "macro_f1": macro_f1,
        "protocol": protocol, "status": status, "source": source,
    }


def _curated_results() -> list[dict[str, object]]:
    rows = []
    rice = "Final group-aware"
    rows += [
        _catalog_row("Rice","Centralized","ResNet18","Group-aware","Test",accuracy=99.0097,balanced_accuracy=98.1184,macro_f1=97.9416,protocol=rice),
        _catalog_row("Rice","Centralized","MobileNetV2","Group-aware","Test",accuracy=99.0312,balanced_accuracy=98.4235,macro_f1=98.0446,protocol=rice),
        _catalog_row("Rice","Centralized","EfficientNetB0","Group-aware","Test",accuracy=98.9236,balanced_accuracy=98.2873,macro_f1=97.8079,protocol=rice),
        _catalog_row("Rice","DDP","ResNet18","Two GPUs","Test",accuracy=98.9236,balanced_accuracy=97.9743,macro_f1=97.7568,protocol=rice),
        _catalog_row("Rice","FedAvg","ResNet18","IID","Test",accuracy=99.1173,balanced_accuracy=98.2500,macro_f1=98.1136,protocol=rice),
        _catalog_row("Rice","FedAvg","ResNet18","Non-IID α=0.5","Test",accuracy=99.0958,balanced_accuracy=98.1965,macro_f1=98.0866,alpha=0.5,protocol=rice),
        _catalog_row("Rice","FedAvg","MobileNetV2","IID","Test",accuracy=98.9882,balanced_accuracy=98.4579,macro_f1=97.9948,protocol=rice),
        _catalog_row("Rice","FedAvg","MobileNetV2","Non-IID α=0.1","Test",accuracy=99.0527,balanced_accuracy=98.5727,macro_f1=98.1358,alpha=0.1,protocol=rice),
        _catalog_row("Rice","FedAvg","MobileNetV2","Non-IID α=0.5","Test",accuracy=99.0097,balanced_accuracy=98.5142,macro_f1=98.0484,alpha=0.5,protocol=rice),
        _catalog_row("Rice","FedPer","MobileNetV2","IID","Test",accuracy=99.0743,balanced_accuracy=98.5759,macro_f1=98.1579,client="Client mean",protocol=rice),
        _catalog_row("Rice","FedPer","MobileNetV2","Non-IID α=0.1","Test",accuracy=99.0814666667,balanced_accuracy=98.5258,macro_f1=98.1672333333,alpha=0.1,client="Client mean",protocol=rice),
        _catalog_row("Rice","FedPer","MobileNetV2","Non-IID α=0.5","Test",accuracy=98.9666333333,balanced_accuracy=98.4170,macro_f1=97.9198,alpha=0.5,client="Client mean",protocol=rice),
        _catalog_row("Rice","FedRep","MobileNetV2","IID","Test",accuracy=99.0599333333,balanced_accuracy=98.2254333333,macro_f1=98.0413333333,client="Client mean",protocol=rice),
        _catalog_row("Rice","FedRep","MobileNetV2","Non-IID α=0.1","Test",accuracy=99.0168666667,balanced_accuracy=98.3521333333,macro_f1=98.0299333333,alpha=0.1,client="Client mean",protocol=rice),
        _catalog_row("Rice","FedRep","MobileNetV2","Non-IID α=0.5","Test",accuracy=99.1173333333,balanced_accuracy=98.4231,macro_f1=98.2075333333,alpha=0.5,client="Client mean",protocol=rice),
    ]

    old = "Historical / earlier split"
    old_src = "Historical project evidence"
    for split, score in (("Validation",74.63),("Test",80.48),("Test-07",68.65)):
        rows.append(_catalog_row("Wheat","Centralized","MobileNetV2","Earlier split · V1",split,macro_f1=score,protocol=old,status="Historical",source=old_src))
    for split, score in (("Validation",78.09),("Test",84.83),("Test-07",72.42)):
        rows.append(_catalog_row("Wheat","Centralized","ResNet18","Earlier split · V1",split,macro_f1=score,protocol=old,status="Historical",source=old_src))
    for split, score in (("Validation",78.86),("Test",86.03),("Test-07",73.17)):
        rows.append(_catalog_row("Wheat","Centralized","ResNet18","Earlier split · V3 dynamic",split,macro_f1=score,protocol=old,status="Historical",source=old_src))
    rows.append(_catalog_row("Wheat","Centralized","EfficientNetB0","Earlier split","Test",protocol=old,status="Metric not verified",source=old_src))
    for split, score in (("Validation",76.18),("Test",81.87),("Test-07",70.19)):
        rows.append(_catalog_row("Wheat","FedAvg","ResNet18","IID · Earlier split",split,macro_f1=score,protocol=old,status="Historical",source=old_src))
    rows += [
        _catalog_row("Wheat","FedAvg","ResNet18","Non-IID · Earlier split","Validation",macro_f1=75.22,protocol=old,status="Historical",source=old_src),
        _catalog_row("Wheat","FedAvg","ResNet18","Non-IID · Earlier split","Test",protocol=old,status="Metric not verified",source=old_src),
        _catalog_row("Wheat","FedAvg","ResNet18","Non-IID · Earlier split","Test-07",protocol=old,status="Metric not verified",source=old_src),
        _catalog_row("Wheat","DDP","ResNet18","Distributed · Earlier split","Validation",protocol=old,status="Pipeline validated · metric not verified",source=old_src),
        _catalog_row("Wheat","DDP","ResNet18","Distributed · Earlier split","Test",protocol=old,status="Pipeline validated · metric not verified",source=old_src),
        _catalog_row("Wheat","DDP","ResNet18","Distributed · Earlier split","Test-07",protocol=old,status="Pipeline validated · metric not verified",source=old_src),
    ]

    final_wheat = "Final zero-overlap group-aware"
    values = {
        "Group-aware · Full inverse": {
            "Validation": (89.6368,78.9330,68.9652), "Test": (92.2408,85.5208,85.2522), "Test-07": (83.0791,74.5803,70.8532),
        },
        "Group-aware · Square-root": {
            "Validation": (93.5433,79.2635,75.2260), "Test": (93.4658,85.8020,86.7590), "Test-07": (86.4022,75.4806,74.3030),
        },
    }
    for setting, split_map in values.items():
        for split, m in split_map.items():
            rows.append(_catalog_row("Wheat","Centralized","ResNet18",setting,split,accuracy=m[0],balanced_accuracy=m[1],macro_f1=m[2],protocol=final_wheat,status="Final",source="Corrected Wheat evaluation"))
    return rows


def _eval_split(value: object) -> str:
    text = str(value).lower().replace("_", "-")
    if "test-07" in text or "test07" in text or "test 07" in text:
        return "Test-07"
    if "validation" in text or text.strip() == "val":
        return "Validation"
    return "Test"


def _auto_rice_results() -> list[dict[str, object]]:
    auto = authoritative_results_data(str(PROJECT_ROOT))
    if auto.empty:
        return []
    rows = []
    for _, r in auto[auto["dataset"].eq("Rice")].iterrows():
        alpha_raw = pd.to_numeric(r.get("alpha"), errors="coerce")
        alpha = None if pd.isna(alpha_raw) else float(alpha_raw)
        setting = str(r.get("distribution", "")).strip() or "Group-aware"
        if "non-iid" in setting.lower() and alpha is not None and "α=" not in setting:
            setting = f"Non-IID α={alpha:g}"
        if str(r.get("method")) == "Centralized" and setting.lower() == "centralized":
            setting = "Group-aware"
        if str(r.get("method")) == "DDP":
            setting = "Two GPUs"
        rows.append(_catalog_row(
            "Rice", str(r.get("method", "")), str(r.get("architecture", "")), setting,
            _eval_split(r.get("evaluation_basis", "Test")),
            accuracy=None if pd.isna(r.get("accuracy")) else float(r.get("accuracy")),
            balanced_accuracy=None if pd.isna(r.get("balanced_accuracy")) else float(r.get("balanced_accuracy")),
            macro_f1=None if pd.isna(r.get("macro_f1")) else float(r.get("macro_f1")),
            alpha=alpha, client=str(r.get("client", "Global")) or "Global",
            protocol="Final group-aware", status="Final", source=str(r.get("source", "")),
        ))
    return rows


@st.cache_data(show_spinner=False)
def comparison_results_data(project_root_string: str) -> pd.DataFrame:
    _ = project_root_string
    frame = pd.DataFrame(_curated_results() + _auto_rice_results())
    if frame.empty:
        return frame
    for col in ("accuracy","balanced_accuracy","macro_f1","alpha"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in ("dataset","method","architecture","setting","evaluation_split","client","protocol","status","source"):
        frame[col] = frame[col].fillna("").astype(str).str.strip()
    frame["_priority"] = frame["source"].apply(lambda s: 0 if s in {"Verified project record","Historical project evidence","Corrected Wheat evaluation"} else 1)
    frame = frame.sort_values("_priority")
    frame = frame.drop_duplicates(subset=["dataset","method","architecture","setting","evaluation_split","client"], keep="first")
    return frame.drop(columns="_priority").reset_index(drop=True)


def _ordered(values: list[str], preferred: list[str]) -> list[str]:
    clean = [v for v in values if v]
    return [v for v in preferred if v in clean] + [v for v in clean if v not in preferred]


def _show_chart_if_present(filenames: list[str], caption: str) -> bool:
    for filename in filenames:
        path = CHART_DIR / filename
        if path.exists():
            st.image(path, caption=caption, width="stretch")
            return True
    return False


def _metric_config() -> dict[str, object]:
    return {
        "accuracy": st.column_config.NumberColumn("Accuracy (%)", format="%.4f"),
        "balanced_accuracy": st.column_config.NumberColumn("Balanced Accuracy (%)", format="%.4f"),
        "macro_f1": st.column_config.NumberColumn("Macro-F1 (%)", format="%.4f"),
        "alpha": st.column_config.NumberColumn("Alpha", format="%.2f"),
    }


def experiment_comparison() -> None:
    header(
        "Results", "Experiment Comparison",
        "Explore the complete experiment catalog. Final leakage-aware results are kept "
        "separate from historical pipeline-development runs; missing metrics stay blank.",
    )
    results = comparison_results_data(str(PROJECT_ROOT))
    if results.empty:
        st.info("No experiment records are available.")
        return

    dataset = st.selectbox("Dataset", _ordered(unique(results,"dataset"), ["Rice","Wheat"]), key="cmp_dataset")
    rows = results[results["dataset"].eq(dataset)].copy()
    method = st.selectbox("Method", _ordered(unique(rows,"method"), ["Centralized","DDP","FedAvg","FedPer","FedRep"]), key="cmp_method")
    rows = rows[rows["method"].eq(method)]
    archs = _ordered(unique(rows,"architecture"), ["ResNet18","MobileNetV2","EfficientNetB0"])
    architecture = st.selectbox("Architecture", archs, key="cmp_arch", disabled=len(archs)==1)
    rows = rows[rows["architecture"].eq(architecture)]
    settings = sorted(unique(rows,"setting"), key=lambda x: ("earlier" in x.lower(), x.lower()))
    setting = st.selectbox("Data setting / experiment variant", settings, key="cmp_setting", disabled=len(settings)==1)
    rows = rows[rows["setting"].eq(setting)]
    splits = _ordered(unique(rows,"evaluation_split"), ["Test","Validation","Test-07"])
    split = st.selectbox("Evaluation split", splits, key="cmp_split", disabled=len(splits)==1)
    rows = rows[rows["evaluation_split"].eq(split)]
    clients = _ordered(unique(rows,"client"), ["Global","Client mean","Client 0","Client 1","Client 2"])
    client = st.selectbox("Evaluation target", clients, key="cmp_client", disabled=len(clients)==1) if clients else "Global"
    selected = rows[rows["client"].eq(client)].copy() if clients else rows.copy()
    if selected.empty:
        selected = rows.copy()

    metric_label = st.radio("Metric", ("Macro-F1","Accuracy","Balanced Accuracy"), horizontal=True, key="cmp_metric")
    metric_name = {"Macro-F1":"macro_f1","Accuracy":"accuracy","Balanced Accuracy":"balanced_accuracy"}[metric_label]
    row = selected.iloc[0] if not selected.empty else None

    cards = st.columns(4)
    cards[0].metric("Accuracy", metric_display(row["accuracy"]) if row is not None else "—")
    cards[1].metric("Balanced Accuracy", metric_display(row["balanced_accuracy"]) if row is not None else "—")
    cards[2].metric("Macro-F1", metric_display(row["macro_f1"]) if row is not None else "—")
    cards[3].metric("Evaluation", f"{split} · {client}")

    if row is not None:
        if "historical" in str(row["protocol"]).lower() or "earlier" in setting.lower():
            st.warning("This Wheat run used the earlier split and is shown only as historical pipeline-development evidence. It is not directly comparable with the final zero-overlap Wheat results.")
        elif dataset == "Wheat" and "final" in str(row["protocol"]).lower():
            st.success("Final corrected Wheat protocol: complete capture groups stay within partitions and pairwise capture-group overlap is zero.")
        if pd.isna(row[metric_name]):
            st.info(f"{metric_label} for this exact run/split is not verified in the saved evidence. The experiment remains visible, but no value is fabricated.")

    chart_candidates = [f"experiment_{token(dataset)}_{token(method)}_{token(setting)}_{metric_name}.png"]
    if dataset == "Rice" and metric_name == "macro_f1":
        chart_candidates += {
            "Centralized":["results_centralized_architecture_test_metrics.png"],
            "FedAvg":["results_fedavg_macro_f1.png"],
            "FedPer":["results_fedper_macro_f1.png"],
            "FedRep":["results_fedrep_macro_f1.png"],
            "DDP":["results_global_models_macro_f1.png"],
        }.get(method, [])
    if _show_chart_if_present(chart_candidates, f"{dataset} · {method} · {architecture} · {metric_label}"):
        st.caption("Pre-generated project chart; Streamlit does not recalculate it.")

    display_cols = ["dataset","method","architecture","setting","evaluation_split","alpha","client","accuracy","balanced_accuracy","macro_f1","protocol","status"]
    st.subheader("Selected experiment record")
    st.dataframe(selected[display_cols], hide_index=True, width="stretch", column_config=_metric_config())

    st.subheader(f"Complete {dataset} experiment coverage")
    st.caption("Blank metrics indicate a run that is recorded but whose exact value is not treated as verified.")
    coverage = results[results["dataset"].eq(dataset)][["method","architecture","setting","evaluation_split","alpha","client","accuracy","balanced_accuracy","macro_f1","protocol","status"]].sort_values(["method","architecture","setting","evaluation_split","client"], kind="stable")
    st.dataframe(coverage, hide_index=True, width="stretch", height=min(650, 70 + 34*len(coverage)), column_config=_metric_config())


# =============================================================================
# PAGE: FEDERATED CLIENT ANALYSIS
# =============================================================================


def client_analysis() -> None:
    header(
        "Clients", "Federated Client Analysis",
        "Inspect discovered client partition sizes and personalized FedPer/FedRep test "
        "performance without inventing missing client-level statistics.",
    )
    partitions = discovered_client_partitions(str(PROJECT_ROOT))
    results = comparison_results_data(str(PROJECT_ROOT))
    fed = results[results["method"].isin(["FedAvg","FedPer","FedRep"])].copy()

    datasets = _ordered(sorted(set(unique(partitions,"dataset")) | set(unique(fed,"dataset"))), ["Rice","Wheat"])
    if not datasets:
        st.info("No federated experiment evidence is available.")
        return
    dataset = st.selectbox("Dataset", datasets, key="client_dataset")
    p = partitions[partitions["dataset"].eq(dataset)].copy()
    r = fed[fed["dataset"].eq(dataset)].copy()

    methods = _ordered(sorted(set(unique(p,"method")) | set(unique(r,"method"))), ["FedAvg","FedPer","FedRep"])
    method = st.selectbox("Method", methods, key="client_method")
    p = p[p["method"].eq(method)]
    r = r[r["method"].eq(method)]

    archs = _ordered(sorted(set(unique(p,"architecture")) | set(unique(r,"architecture"))), ["ResNet18","MobileNetV2"])
    architecture = st.selectbox("Architecture", archs, key="client_arch", disabled=len(archs)==1)
    p = p[p["architecture"].eq(architecture)]
    r = r[r["architecture"].eq(architecture)]

    settings = _ordered(sorted(set(unique(p,"distribution")) | set(unique(r,"setting"))), ["IID","Non-IID α=0.1","Non-IID α=0.5","IID · Earlier split","Non-IID · Earlier split"])
    setting = st.selectbox("Partition / data setting", settings, key="client_setting", disabled=len(settings)==1) if settings else ""
    psel = p[p["distribution"].eq(setting)].copy() if not p.empty else p
    rsel = r[r["setting"].eq(setting)].copy() if not r.empty else r

    st.subheader("Client partition sizes")
    alpha_selected = None
    if psel.empty:
        if dataset == "Wheat":
            st.info("The Wheat federated pipeline is present in the experiment catalog, but no verified per-client sample-count artifact for this selection was discovered. Its global historical results remain available in Experiment Comparison.")
        else:
            st.info("No per-client sample-count artifact matches this selection.")
    else:
        alphas = sorted(psel["alpha"].dropna().unique())
        if len(alphas) > 1:
            alpha_selected = st.selectbox("Dirichlet alpha", alphas, format_func=lambda x: f"{x:g}", key="client_alpha")
            psel = psel[psel["alpha"].eq(alpha_selected)]
        elif alphas:
            alpha_selected = float(alphas[0])
        cards = st.columns(4)
        cards[0].metric("Clients", fmt_int(psel["client"].nunique()))
        cards[1].metric("Training images", fmt_int(psel["images"].sum()))
        cards[2].metric("Smallest client", fmt_int(psel["images"].min()))
        cards[3].metric("Largest client", fmt_int(psel["images"].max()))
        st.dataframe(
            psel[["dataset","method","architecture","distribution","alpha","client","images","source"]].sort_values("client"),
            hide_index=True, width="stretch",
            column_config={"alpha":st.column_config.NumberColumn("Alpha",format="%.2f"),"images":st.column_config.NumberColumn("Images",format="localized")},
        )
        candidates = []
        if alpha_selected is not None:
            a = str(alpha_selected).replace(".","_")
            candidates += [f"clients_{token(dataset)}_alpha_{a}_images.png",f"clients_{token(dataset)}_alpha_{a}_sample_counts.png"]
        if dataset == "Rice" and setting in {"IID","Non-IID α=0.5"}:
            candidates.append("clients_iid_vs_noniid_sample_counts.png")
        _show_chart_if_present(candidates, f"{dataset} · {method} · {setting} client sample counts")

    st.subheader("Federated evaluation results")
    if rsel.empty:
        st.info("No verified evaluation rows match this selection.")
    else:
        client_rows = rsel[rsel["client"].str.contains("Client",case=False,na=False)].copy()
        display = client_rows if not client_rows.empty else rsel
        st.dataframe(
            display[["setting","evaluation_split","alpha","client","accuracy","balanced_accuracy","macro_f1","protocol","status"]].sort_values(["client","evaluation_split"]),
            hide_index=True, width="stretch", column_config=_metric_config(),
        )

    if dataset == "Rice" and method == "FedAvg":
        path = CHART_DIR / "clients_fedavg_validation_macro_f1_by_round.png"
        if path.exists():
            st.subheader("FedAvg convergence")
            st.image(path, caption="FedAvg validation Macro-F1 over five rounds", width="stretch")
            st.caption("Validation history only; final comparison uses held-out test metrics.")


# =============================================================================
# PAGE: CONFUSION MATRICES
# =============================================================================


@st.cache_data(show_spinner=False)
def confusion_files(project_root_string: str) -> list[str]:
    root = Path(project_root_string)
    roots = [CM_DIR, root/"figures", root/"paper"/"figures", root/"results", root/"experiments"/"results"]
    found = set()
    for folder in roots:
        if not folder.exists():
            continue
        try:
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and any(marker in path.stem.lower() for marker in CM_MARKERS):
                    found.add(path.resolve())
        except OSError:
            pass
    return [str(p) for p in sorted(found,key=lambda p:str(p).lower())]


def _matrix_metadata(path: Path) -> dict[str,str]:
    text = path.as_posix().lower()
    dataset = "Rice" if "rice" in text else "Wheat" if "wheat" in text else "Unknown"
    method = "FedAvg" if "fedavg" in text else "FedPer" if "fedper" in text else "FedRep" if "fedrep" in text else "DDP" if ("ddp" in text or "distributed" in text) else "Centralized"
    architecture = "MobileNetV2" if ("mobilenetv2" in text or "mobilenet_v2" in text) else "EfficientNetB0" if ("efficientnetb0" in text or "efficientnet_b0" in text) else "ResNet18" if "resnet18" in text else "Unknown"
    if dataset=="Wheat" and "sqrt" in text:
        setting="Group-aware · Square-root"
    elif dataset=="Wheat" and "grouped_v2" in text:
        setting="Group-aware · Full inverse"
    elif re.search(r"alpha[_=\-]?0[._p]?1",text):
        setting="Non-IID α=0.1"
    elif re.search(r"alpha[_=\-]?0[._p]?5",text):
        setting="Non-IID α=0.5"
    elif "noniid" in text or "non_iid" in text or "non-iid" in text:
        setting="Non-IID"
    elif re.search(r"(^|[/_\-])iid([/_\-]|$)",text):
        setting="IID"
    elif method=="DDP":
        setting="Distributed"
    else:
        setting="Centralized / group-aware"
    evaluation = "Test-07" if any(x in text for x in ("test_07","test07","test-07")) else "Validation" if ("validation" in text or "/val" in text or "_val" in text) else "Test" if "test" in text else "Unspecified"
    match = re.search(r"client[_\- ]?(\d+)",text)
    client = f"Client {match.group(1)}" if match else "Global / unspecified"
    try: rel=str(path.relative_to(PROJECT_ROOT))
    except ValueError: rel=str(path)
    return {"dataset":dataset,"method":method,"architecture":architecture,"setting":setting,"evaluation":evaluation,"client":client,"relative_path":rel,"path":str(path)}


@st.cache_data(show_spinner=False)
def confusion_catalog(project_root_string: str) -> pd.DataFrame:
    return pd.DataFrame([_matrix_metadata(Path(p)) for p in confusion_files(project_root_string)])


def confusion_matrices() -> None:
    header("Evaluation","Confusion Matrices","Filter every saved confusion matrix by dataset, method, architecture, data setting and evaluation split, then inspect one or compare two.")
    catalog = confusion_catalog(str(PROJECT_ROOT))
    if catalog.empty:
        st.info("No confusion matrices were found in local assets, figures, paper figures, results, or experiments/results.")
        return
    st.metric("Matrices discovered",len(catalog))
    top=st.columns(3)
    dataset=top[0].selectbox("Dataset",["All",*_ordered(unique(catalog,"dataset"),["Rice","Wheat","Unknown"])],key="cm_dataset")
    work=catalog if dataset=="All" else catalog[catalog["dataset"].eq(dataset)]
    method=top[1].selectbox("Method",["All",*_ordered(unique(work,"method"),["Centralized","DDP","FedAvg","FedPer","FedRep"])],key="cm_method")
    if method!="All": work=work[work["method"].eq(method)]
    arch=top[2].selectbox("Architecture",["All",*_ordered(unique(work,"architecture"),["ResNet18","MobileNetV2","EfficientNetB0","Unknown"])],key="cm_arch")
    if arch!="All": work=work[work["architecture"].eq(arch)]
    bottom=st.columns(3)
    setting=bottom[0].selectbox("Data setting",["All",*unique(work,"setting")],key="cm_setting")
    if setting!="All": work=work[work["setting"].eq(setting)]
    evaluation=bottom[1].selectbox("Evaluation split",["All",*_ordered(unique(work,"evaluation"),["Test","Validation","Test-07","Unspecified"])],key="cm_eval")
    if evaluation!="All": work=work[work["evaluation"].eq(evaluation)]
    query=bottom[2].text_input("Path contains",placeholder="alpha0p1 client_0",key="cm_query").lower().strip()
    if query:
        terms=query.split(); work=work[work["path"].str.lower().apply(lambda v:all(t in v for t in terms))]
    if work.empty:
        st.warning("No confusion matrix matches the current filters."); return
    st.caption(f"{len(work)} matrix/matrices match the current filters.")
    labels={f"{r.dataset} · {r.method} · {r.architecture} · {r.setting} · {r.evaluation} · {r.client} — {r.relative_path}":Path(r.path) for r in work.itertuples()}
    mode=st.radio("View",("Single","Compare two"),horizontal=True,key="cm_mode")
    names=list(labels)
    if mode=="Single":
        name=st.selectbox("Matrix",names,key="cm_single"); st.image(labels[name],caption=name,width="stretch")
    else:
        cols=st.columns(2)
        with cols[0]:
            name=st.selectbox("Left matrix",names,index=0,key="cm_left"); st.image(labels[name],caption=name,width="stretch")
        with cols[1]:
            name=st.selectbox("Right matrix",names,index=1 if len(names)>1 else 0,key="cm_right"); st.image(labels[name],caption=name,width="stretch")
    with st.expander("Matching matrix files"):
        st.dataframe(work[["dataset","method","architecture","setting","evaluation","client","relative_path"]],hide_index=True,width="stretch")


# =============================================================================
# PAGE: IMAGE PREDICTION
# =============================================================================


CHECKPOINT_EXTENSIONS={".pt",".pth",".ckpt"}


@st.cache_data(show_spinner=False)
def checkpoint_files(project_root_string: str) -> list[str]:
    root=Path(project_root_string); found=set()
    for folder in (root/"results",root/"experiments"/"results",root/"checkpoints"):
        if not folder.exists(): continue
        try:
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in CHECKPOINT_EXTENSIONS: found.add(path.resolve())
        except OSError: pass
    return [str(p) for p in sorted(found,key=lambda p:str(p).lower())]


def _checkpoint_metadata(text: str) -> tuple[str,str,str]:
    lower=text.lower()
    dataset="Rice" if "rice" in lower else "Wheat" if "wheat" in lower else "Unknown"
    architecture="MobileNetV2" if ("mobilenetv2" in lower or "mobilenet_v2" in lower) else "EfficientNetB0" if ("efficientnetb0" in lower or "efficientnet_b0" in lower) else "ResNet18" if "resnet18" in lower else "Unknown"
    method="FedPer" if "fedper" in lower else "FedRep" if "fedrep" in lower else "FedAvg" if ("fedavg" in lower or "global_model" in lower) else "DDP" if "ddp" in lower else "Centralized"
    return dataset,architecture,method


def _class_names(dataset: str) -> list[str]:
    if dataset=="Rice":
        return ["0_NOR — Normal","1_F&S — Fusarium and Shriveled","2_SD — Sprouted","3_MY — Moldy","4_AP — Attacked by Pests","5_BN — Broken","6_UN — Unripe","7_IM — Impurities"]
    return ["Black Germ","Broken","Fusarium","Insect","Moldy","Sound","Spotted","Sprouted"]


def _safe_state_dict(source):
    import torch
    obj=torch.load(source,map_location="cpu",weights_only=True)
    state=obj
    if isinstance(obj,dict):
        for key in ("model_state_dict","state_dict","global_model_state_dict","model"):
            if isinstance(obj.get(key),dict): state=obj[key]; break
    if not isinstance(state,dict): raise ValueError("Checkpoint does not contain a recognizable state dictionary.")
    cleaned={}
    for key,value in state.items():
        if not isinstance(key,str): continue
        for prefix in ("module.","_orig_mod."):
            if key.startswith(prefix): key=key[len(prefix):]
        cleaned[key]=value
    if not cleaned: raise ValueError("No model parameters were found in the checkpoint.")
    return cleaned


def _build_model(architecture: str,num_classes: int):
    from torch import nn
    from torchvision import models
    if architecture=="ResNet18":
        model=models.resnet18(weights=None); model.fc=nn.Linear(model.fc.in_features,num_classes); return model
    if architecture=="MobileNetV2":
        model=models.mobilenet_v2(weights=None); model.classifier[1]=nn.Linear(model.classifier[1].in_features,num_classes); return model
    if architecture=="EfficientNetB0":
        model=models.efficientnet_b0(weights=None); model.classifier[1]=nn.Linear(model.classifier[1].in_features,num_classes); return model
    raise ValueError(f"Unsupported architecture: {architecture}")


def _predict(image: Image.Image,state: dict,architecture: str,dataset: str) -> pd.DataFrame:
    import torch
    from torchvision import transforms
    labels=_class_names(dataset); model=_build_model(architecture,len(labels)); model.load_state_dict(state,strict=True); model.eval()
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    x=transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode(): probs=torch.softmax(model(x),dim=1)[0]
    values,indices=torch.topk(probs,k=min(3,len(labels)))
    return pd.DataFrame([{"rank":i+1,"predicted_class":labels[idx],"softmax_score":score*100} for i,(score,idx) in enumerate(zip(values.tolist(),indices.tolist()))])


def prediction() -> None:
    header("Inference","Image Prediction","Run local CPU inference with a compatible saved project checkpoint. No synthetic prediction is shown when a checkpoint is unavailable or incompatible.")
    upload=st.file_uploader("Upload a grain image",type=["jpg","jpeg","png","webp","bmp"],key="pred_image")
    if upload is None: st.info("Upload an image to begin."); return
    try:
        image=Image.open(upload); image.load(); image=image.convert("RGB")
    except (UnidentifiedImageError,OSError) as exc:
        st.error(f"Could not open image: {exc}"); return
    left,right=st.columns([1.05,1])
    with left:
        st.image(image,caption=upload.name,width="stretch"); st.caption(f"{image.width} × {image.height} · RGB")
    checkpoints=[Path(p) for p in checkpoint_files(str(PROJECT_ROOT))]
    with right:
        modes=["Repository checkpoint","Upload checkpoint"] if checkpoints else ["Upload checkpoint"]
        mode=st.radio("Checkpoint source",modes,horizontal=True,key="pred_mode")
        path=None; cp_upload=None; inferred=("Unknown","Unknown","Unknown")
        if mode=="Repository checkpoint":
            def label(p):
                ds,arch,method=_checkpoint_metadata(str(p))
                try: rel=p.relative_to(PROJECT_ROOT)
                except ValueError: rel=p
                return f"{ds} · {method} · {arch} — {rel}"
            mapping={label(p):p for p in checkpoints}; chosen=st.selectbox("Checkpoint",list(mapping),key="pred_cp"); path=mapping[chosen]; inferred=_checkpoint_metadata(str(path))
        else:
            cp_upload=st.file_uploader("Upload .pt / .pth / .ckpt",type=["pt","pth","ckpt"],key="pred_cp_upload")
            if cp_upload is not None: inferred=_checkpoint_metadata(cp_upload.name)
        ds_opts=["Rice","Wheat"]; dataset=st.selectbox("Dataset / class mapping",ds_opts,index=ds_opts.index(inferred[0]) if inferred[0] in ds_opts else 0,key="pred_dataset")
        arch_opts=["ResNet18","MobileNetV2","EfficientNetB0"]; architecture=st.selectbox("Architecture",arch_opts,index=arch_opts.index(inferred[1]) if inferred[1] in arch_opts else 0,key="pred_arch")
        if inferred[2] in {"FedPer","FedRep"}: st.warning("FedPer/FedRep may store shared and private states separately. Use a compatible complete client checkpoint, or a centralized/FedAvg/DDP checkpoint.")
        run=st.button("Run prediction",type="primary",width="stretch")
    if not run:
        if not checkpoints: st.caption("No repository checkpoint was discovered. You can upload a compatible saved checkpoint even when large model files are not committed to Git.")
        return
    if mode=="Upload checkpoint" and cp_upload is None: st.error("Upload a checkpoint before running prediction."); return
    try:
        if path is not None: state=_safe_state_dict(path)
        else:
            cp_upload.seek(0); state=_safe_state_dict(io.BytesIO(cp_upload.read()))
        pred=_predict(image,state,architecture,dataset)
    except ModuleNotFoundError:
        st.error("PyTorch/torchvision is not installed in the Streamlit environment. Run the dashboard from the project environment containing the model dependencies."); return
    except Exception as exc:
        st.error("The checkpoint is not compatible with the selected architecture/class mapping. No prediction was produced."); st.exception(exc); return
    st.subheader("Prediction"); st.metric("Predicted class",pred.iloc[0]["predicted_class"])
    st.caption("Scores are softmax outputs from the selected checkpoint; they are not calibrated probabilities.")
    st.dataframe(pred,hide_index=True,width="stretch",column_config={"rank":st.column_config.NumberColumn("Rank",format="%d"),"predicted_class":"Class","softmax_score":st.column_config.NumberColumn("Softmax score (%)",format="%.2f")})


# =============================================================================
# PAGE: DATA STATUS
# =============================================================================


def data_status() -> None:
    header("Files","Data Status","Audit the sources behind the dashboard and confirm which experiment, client, matrix, chart and checkpoint artifacts are discoverable.")
    results=comparison_results_data(str(PROJECT_ROOT)); clients=discovered_client_partitions(str(PROJECT_ROOT)); matrices=confusion_catalog(str(PROJECT_ROOT)); checkpoints=checkpoint_files(str(PROJECT_ROOT))
    charts=list(CHART_DIR.glob("*.png")) if CHART_DIR.exists() else []
    samples=[p for p in SAMPLE_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS] if SAMPLE_DIR.exists() else []
    cards=st.columns(6)
    cards[0].metric("Experiment rows",len(results)); cards[1].metric("Methods",results["method"].nunique() if not results.empty else 0); cards[2].metric("Client rows",len(clients)); cards[3].metric("Confusion matrices",len(matrices)); cards[4].metric("Checkpoints",len(checkpoints)); cards[5].metric("Static PNG charts",len(charts))
    st.subheader("Experiment coverage")
    if not results.empty:
        coverage=results.groupby(["dataset","method"],as_index=False).agg(experiment_rows=("method","size"),architectures=("architecture",lambda s:", ".join(sorted(set(s)))),settings=("setting",lambda s:", ".join(sorted(set(s)))))
        st.dataframe(coverage,hide_index=True,width="stretch")
    st.subheader("Repository resources")
    resources=[("Detected repository root",PROJECT_ROOT),("Rice final summary",FINAL_RESULTS_TABLE),("Centralized architecture table",CENTRALIZED_ARCHITECTURE_TABLE),("results/",PROJECT_ROOT/"results"),("experiments/results/",PROJECT_ROOT/"experiments"/"results"),("Generated charts",CHART_DIR),("Dataset samples",SAMPLE_DIR),("Local confusion-matrix assets",CM_DIR)]
    frame=pd.DataFrame([{"resource":label,"path":str(path),"available":path.exists()} for label,path in resources])
    st.dataframe(frame,hide_index=True,width="stretch",column_config={"available":st.column_config.CheckboxColumn("Available")})
    st.dataframe(pd.DataFrame([{"artifact":"Dataset sample images","count":len(samples)},{"artifact":"Generated PNG charts","count":len(charts)},{"artifact":"Client partition rows","count":len(clients)},{"artifact":"Confusion matrices","count":len(matrices)},{"artifact":"Model checkpoints","count":len(checkpoints)}]),hide_index=True,width="stretch")
    if checkpoints:
        with st.expander("Discovered model checkpoints"):
            rows=[]
            for value in checkpoints:
                p=Path(value); ds,arch,method=_checkpoint_metadata(value)
                try: rel=p.relative_to(PROJECT_ROOT)
                except ValueError: rel=p
                rows.append({"dataset":ds,"method":method,"architecture":arch,"path":str(rel)})
            st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    st.code("cd streamlit-ui/streamlit-ui\nsource ../../.venv/bin/activate\npython -m streamlit run app.py",language="bash")
    st.caption(f"Detected repository root: {PROJECT_ROOT}")


def main() -> None:
    inject_css()
    page = sidebar()
    routes = {
        "Research Overview":overview,
        "Dataset Explorer":dataset_explorer,
        "Experiment Comparison":experiment_comparison,
        "Federated Client Analysis":client_analysis,
        "Confusion Matrices":confusion_matrices,
        "Image Prediction":prediction,
        "Data Status":data_status,
    }
    try:
        routes[page]()
    except Exception as exc:
        st.error("This page encountered an unexpected error.")
        st.exception(exc)


if __name__ == "__main__":
    main()

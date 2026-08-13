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
REPO_ROOT = APP_DIR.parent
DATA_DIR = APP_DIR / "data"
ASSET_DIR = APP_DIR / "assets"
SAMPLE_DIR = ASSET_DIR / "dataset_samples"
CHART_DIR = ASSET_DIR / "generated_charts"
CM_DIR = ASSET_DIR / "confusion_matrices"

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


def render_gallery() -> None:
    files = sorted(
        p for p in SAMPLE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ) if SAMPLE_DIR.exists() else []

    cards = []
    for path in files[:24]:
        uri = image_uri(path)
        if uri:
            label = html.escape(re.sub(r"[_\-]+", " ", path.stem).title())
            cards.append(
                f'<article class="gallery-card"><img src="{uri}" alt="{label}"><div>{label}</div></article>'
            )

    if not cards:
        st.html(
            '<div class="placeholder"><div>🌾<br>Rice sample</div>'
            '<div>🌾<br>Wheat sample</div><div>🔬<br>Healthy grain</div>'
            '<div>🧪<br>Fault category</div></div>'
        )
        st.caption("Add images to `assets/dataset_samples/`.")
        return

    st.html(f'<section class="gallery-shell"><div class="gallery-track">{"".join(cards + cards)}</div></section>')


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
    exp, error = experiments_data()

    st.html("""
    <section class="hero">
      <small>Research overview</small>
      <h1>Federated and Distributed Learning for Grain Classification</h1>
      <p>A concise interface for presenting the datasets, implemented methods, verified results and error analysis.</p>
    </section>
    """)

    cols = st.columns(4)
    cols[0].metric("Images", fmt_int(meta.get("total_images")))
    cols[1].metric("Datasets", fmt_int(meta.get("dataset_count")))
    cols[2].metric("Federated clients", fmt_int(meta.get("federated_clients")))
    cols[3].metric("FL rounds", fmt_int(meta.get("federated_rounds")))

    if error:
        st.error(error)

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
            '<strong>Partitions:</strong> IID and non-IID</div>'
        )

    st.subheader("Performance snapshot")
    rows = exp[exp[["accuracy","macro_f1"]].notna().any(axis=1)].copy()
    if rows.empty:
        st.info("No verified results are available.")
        return

    st.dataframe(
        rows[[
            "dataset","architecture","method","distribution","alpha","client",
            "accuracy","macro_f1","status"
        ]],
        hide_index=True,
        width="stretch",
        column_config={
            "accuracy":st.column_config.NumberColumn("Accuracy (%)", format="%.4f"),
            "macro_f1":st.column_config.NumberColumn("Macro-F1 (%)", format="%.4f"),
            "alpha":st.column_config.NumberColumn("Alpha", format="%.2f"),
        },
    )


def dataset_explorer() -> None:
    header(
        "Dataset",
        "Dataset Explorer",
        "View representative grain images, class composition and the "
        "capture-group-aware rice data split.",
    )

    # ---------------------------------------------------------------------
    # Dataset image gallery
    # ---------------------------------------------------------------------
    st.subheader("Sample images")
    render_gallery()

    st.write("")

    # ---------------------------------------------------------------------
    # Pre-generated rice and wheat category charts
    # ---------------------------------------------------------------------
    st.subheader("Dataset category composition")

    left, right = st.columns(2)

    with left:
        static_chart(
            "dataset_rice_test_class_distribution.png",
            "Rice test-set class distribution",
            (
                "The rice class-distribution chart was not found. Run "
                "`python generate_rice_wheat_pie_charts.py --root .` "
                "from the repository root."
            ),
        )

    with right:
        static_chart(
            "dataset_wheat_category_distribution.png",
            "Wheat development-pool category allocation",
            (
                "The wheat category-distribution chart was not found. Run "
                "`python generate_rice_wheat_pie_charts.py --root .` "
                "from the repository root."
            ),
        )

    st.caption(
        "The rice chart represents the held-out rice test split. "
        "The wheat chart represents the development-pool allocation recorded "
        "in the verified wheat class-allocation file."
    )

    st.write("")

    # ---------------------------------------------------------------------
    # Verified capture-group-aware rice split
    # ---------------------------------------------------------------------
    st.subheader("Capture-group-aware rice split")

    split_chart, split_information = st.columns([1.35, 0.65])

    with split_chart:
        static_chart(
            "dataset_rice_group_aware_split.png",
            "Rice train, validation and test split",
            (
                "The verified rice split chart was not found. Run "
                "`python generate_dashboard_static_charts.py --root . "
                "--output streamlit-ui/streamlit-ui/assets/generated_charts` "
                "from the repository root."
            ),
        )

    with split_information:
        st.markdown(
            """
            <div class="simple-card">
                <h3 style="margin-top:0;color:#173D28;">
                    Split summary
                </h3>
                <p><strong>Training:</strong> 21,606 images</p>
                <p><strong>Validation:</strong> 4,711 images</p>
                <p><strong>Testing:</strong> 4,645 images</p>
                <p><strong>Total:</strong> 30,962 images</p>
                <hr style="border:none;border-top:1px solid #DDE5DA;">
                <p><strong>Capture-group overlap:</strong> 0</p>
                <p><strong>Seed:</strong> 42</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Capture-group-aware splitting keeps related image captures in only "
        "one partition, reducing train/test leakage."
    )


def architecture_options(exp: pd.DataFrame, dataset: str, method: str) -> list[str]:
    if method in {"FedAvg","FedPer","FedRep"}:
        return ["MobileNetV2"]
    values = exp[exp["dataset"].eq(dataset) & exp["method"].eq(method)]["architecture"]
    found = sorted(v for v in values.unique() if v)
    return found or ["ResNet18","MobileNetV2","EfficientNetB0"]


def distribution_options(exp: pd.DataFrame, dataset: str, method: str) -> list[str]:
    if method == "Centralized":
        return ["Centralized"]
    values = set(
        exp[
            exp["dataset"].eq(dataset)
            & exp["method"].eq(method)
            & exp["distribution"].isin(["IID","Non-IID"])
        ]["distribution"]
    )
    found = [v for v in ["IID","Non-IID"] if v in values]
    return found or ["IID","Non-IID"]


def experiment_comparison() -> None:
    header(
        "Results",
        "Experiment Comparison",
        "Select an experiment. Streamlit displays saved PNG files instead of plotting in real time.",
    )
    exp, error = experiments_data()
    if error:
        st.error(error)

    exp = exp[exp["dataset"].ne("") & exp["method"].ne("")].copy()
    if exp.empty:
        st.info("No valid experiments are available.")
        return

    dataset = st.selectbox("Dataset", unique(exp, "dataset"))
    dataset_rows = exp[exp["dataset"].eq(dataset)].copy()

    preferred = ["Centralized","FedAvg","FedPer","FedRep","DDP"]
    existing = unique(dataset_rows, "method")
    methods = [m for m in preferred if m in existing] + [m for m in existing if m not in preferred]
    method = st.selectbox("Method", methods)

    architectures = architecture_options(exp, dataset, method)
    architecture = st.selectbox(
        "Architecture",
        architectures,
        disabled=len(architectures) == 1,
        help="MobileNetV2 is automatically selected for FedAvg, FedPer and FedRep." if method in {"FedAvg","FedPer","FedRep"} else None,
    )

    distributions = distribution_options(exp, dataset, method)
    distribution = st.selectbox(
        "Distribution",
        distributions,
        disabled=len(distributions) == 1,
        help="Centralized is locked for centralized experiments. Federated methods use IID or non-IID.",
    )

    selected_rows = dataset_rows[
        dataset_rows["method"].eq(method)
        & dataset_rows["architecture"].eq(architecture)
        & dataset_rows["distribution"].eq(distribution)
    ].copy()

    cols = st.columns(4)
    cols[0].metric("Dataset", dataset)
    cols[1].metric("Method", method)
    cols[2].metric("Architecture", architecture)
    cols[3].metric("Distribution", distribution)

    metric_label = st.radio("Metric", ("Macro-F1","Accuracy"), horizontal=True)
    metric = "macro_f1" if metric_label == "Macro-F1" else "accuracy"

    static_chart(
        f"experiment_{token(dataset)}_{token(method)}_{token(distribution)}_{metric}.png",
        f"{dataset} · {method} · {distribution} · {metric_label}",
        "This saved chart has not been generated. Run `python generate_static_charts.py`.",
    )

    st.subheader("Matching rows")
    if selected_rows.empty:
        st.warning("No row exactly matches the selection.")
    else:
        st.dataframe(
            selected_rows[[
                "dataset","architecture","method","distribution","alpha","client",
                "seed","accuracy","macro_f1","selected_round","status","notes"
            ]],
            hide_index=True,
            width="stretch",
        )

    st.subheader(f"All recorded {dataset} results")
    st.dataframe(
        dataset_rows[[
            "architecture","method","distribution","alpha","client","accuracy","macro_f1","status"
        ]],
        hide_index=True,
        width="stretch",
    )


def client_analysis() -> None:
    header("Clients", "Federated Client Analysis", "Display pre-generated client distribution charts.")
    clients, error = clients_data()
    if error:
        st.error(error)
    clients = clients[clients["dataset"].ne("") & clients["client"].ne("")].copy()
    if clients.empty:
        st.info("No client statistics are available.")
        return

    dataset = st.selectbox("Dataset", unique(clients, "dataset"))
    dataset_rows = clients[clients["dataset"].eq(dataset)]
    alphas = sorted(dataset_rows["alpha"].dropna().unique())
    if not alphas:
        st.info("No alpha values are available.")
        return
    alpha = st.selectbox("Dirichlet alpha", alphas, format_func=lambda x: f"{x:g}")
    rows = dataset_rows[dataset_rows["alpha"].eq(alpha)]

    cols = st.columns(4)
    cols[0].metric("Clients", fmt_int(rows["client"].nunique()))
    cols[1].metric("Images", fmt_int(rows["images"].sum()))
    cols[2].metric("Capture groups", fmt_int(rows["capture_groups"].sum()))
    cols[3].metric("Mean entropy", f"{rows['class_entropy'].mean():.4f}" if rows["class_entropy"].notna().any() else "—")

    alpha_token = token(f"{alpha:g}")
    left, right = st.columns(2)
    with left:
        static_chart(
            f"clients_{token(dataset)}_alpha_{alpha_token}_images.png",
            f"{dataset} client images, α={alpha:g}",
        )
    with right:
        static_chart(
            f"clients_{token(dataset)}_alpha_{alpha_token}_entropy.png",
            f"{dataset} class entropy, α={alpha:g}",
        )

    st.dataframe(rows, hide_index=True, width="stretch")


@st.cache_data(show_spinner=False)
def confusion_files() -> list[str]:
    roots = [CM_DIR, REPO_ROOT/"figures", REPO_ROOT/"results"]
    found = set()
    for root in roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    name = path.stem.lower()
                    if any(marker in name for marker in CM_MARKERS):
                        found.add(path.resolve())
        except OSError:
            pass
    return [str(path) for path in sorted(found)]


def confusion_matrices() -> None:
    header("Evaluation", "Confusion Matrices", "Select one saved matrix or compare two.")
    paths = [Path(p) for p in confusion_files()]
    if not paths:
        st.info("No confusion matrices found in assets/confusion_matrices, figures, or results.")
        return

    query = st.text_input("Filter filename", placeholder="fedavg non iid").lower().strip()
    if query:
        terms = query.split()
        paths = [p for p in paths if all(term in str(p).lower() for term in terms)]
    if not paths:
        st.warning("No matrix matches the filter.")
        return

    labels = {f"{p.stem.replace('_',' ').title()} — {p}":p for p in paths}
    mode = st.radio("View", ("Single","Compare two"), horizontal=True)

    if mode == "Single":
        label = st.selectbox("Matrix", list(labels))
        st.image(labels[label], caption=str(labels[label]), width="stretch")
        return

    cols = st.columns(2)
    with cols[0]:
        left_label = st.selectbox("Left matrix", list(labels), index=0, key="cm_left")
        st.image(labels[left_label], caption=str(labels[left_label]), width="stretch")
    with cols[1]:
        idx = 1 if len(labels) > 1 else 0
        right_label = st.selectbox("Right matrix", list(labels), index=idx, key="cm_right")
        st.image(labels[right_label], caption=str(labels[right_label]), width="stretch")


def prediction() -> None:
    header("Inference", "Image Prediction", "Preview the upload interface. Real inference remains disabled.")
    uploaded = st.file_uploader("Upload a grain image", type=["jpg","jpeg","png","webp","bmp"])
    if uploaded is None:
        st.info("Upload an image to preview it.")
        return
    try:
        image = Image.open(uploaded)
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        st.error(f"Could not open image: {exc}")
        return
    left, right = st.columns([1.2,1])
    with left:
        st.image(image, caption=uploaded.name, width="stretch")
    with right:
        st.write(f"**Dimensions:** {image.width} × {image.height}")
        st.write(f"**Mode:** {image.mode}")
        st.write(f"**Format:** {image.format or 'Unknown'}")
        st.warning("No prediction is shown until the verified model is connected.")


def data_status() -> None:
    header("Files", "Data Status", "Check input files and generated static charts.")
    paths = [EXPERIMENTS,CATEGORIES,SPLITS,CLIENTS,CHART_DIR,SAMPLE_DIR,CM_DIR]
    frame = pd.DataFrame({
        "resource":[p.name for p in paths],
        "path":[str(p) for p in paths],
        "available":[p.exists() for p in paths],
    })
    st.dataframe(frame, hide_index=True, width="stretch")
    generated = list(CHART_DIR.glob("*.png")) if CHART_DIR.exists() else []
    st.metric("Generated PNG charts", len(generated))
    st.code("python generate_static_charts.py\npython -m streamlit run app.py", language="bash")


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

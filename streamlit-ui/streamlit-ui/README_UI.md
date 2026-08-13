# Streamlit Dashboard V2

This version keeps the same sidebar navigation but simplifies each page.

## Important design change

The app does not plot charts in real time.

```text
CSV data
  → generate_static_charts.py
  → saved PNG and PDF files
  → app.py displays the saved PNG
```

## Fix the nested folder first

Your current structure is:

```text
streamlit-ui/streamlit-ui/
```

From the repository root, flatten it:

```bash
cd /mnt/c/Users/fajrn/Desktop/GrainClassification/Grain-Federated-Learning-Classification-

cp -a streamlit-ui/streamlit-ui/. streamlit-ui/
rm -rf streamlit-ui/streamlit-ui
```

Before deleting, verify:

```bash
ls streamlit-ui
```

You should see:

```text
app.py
generate_static_charts.py
requirements.txt
assets
data
```

## Install packages

```bash
source .venv/bin/activate
python -m pip install -r streamlit-ui/requirements.txt
```

## Generate saved charts

```bash
cd streamlit-ui
python generate_static_charts.py
```

Files are written to:

```text
assets/generated_charts/
```

- PNG is displayed by Streamlit.
- PDF can be used in your paper or presentation.

## Run the website

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Correct experiment logic

- Centralized:
  - ResNet18, MobileNetV2 or EfficientNetB0.
  - Distribution is locked to Centralized.

- FedAvg, FedPer and FedRep:
  - Architecture is locked to MobileNetV2.
  - Distribution is IID or Non-IID.
  - Centralized cannot be selected.

## Wheat data

The dashboard does not invent wheat category or split counts. Wheat charts appear
only after actual rows are added to:

```text
data/dataset_categories.csv
data/dataset_splits.csv
```

Then rerun:

```bash
python generate_static_charts.py
```

# Crop Yield Prediction ML Project

This is a complete student machine learning project for **Crop Yield Prediction**, prepared to match the requirements in your provided guide (problem statement, dataset, tools, workflow, expected output, model comparison, and deployment).

## Project Structure

```text
ml project/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── sample_crop_yield_data.csv
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   ├── figures/
│   └── project_documentation.md
├── src/
│   ├── eda.py
│   └── train.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Features Implemented
- Regression project for `Yield_kg_per_ha`.
- EDA charts (distribution, correlations, top states).
- Model comparison:
  - Linear Regression
  - Random Forest
  - XGBoost (if available)
- Hyperparameter tuning using `RandomizedSearchCV`.
- Evaluation metrics: `MAE`, `RMSE`, `R2`.
- Best model saved for reuse.
- Streamlit dashboard for predictions.

## Setup

1) Create and activate virtual environment (recommended)

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

1) Generate EDA charts

```bash
python src/eda.py
```

2) Train + evaluate model

```bash
python src/train.py
```

Outputs:
- Model: `models/best_crop_yield_model.joblib`
- Metrics: `models/model_metrics.json`
- Feature importance: `models/feature_importance.csv` (if supported by best model)

3) Launch dashboard

```bash
streamlit run app/streamlit_app.py
```

## Deploy online (free)

Full steps: see **`DEPLOY.md`**.

Short version:

1. Put the project on **GitHub** (include `models/best_crop_yield_model.joblib` and `models/model_metrics.json` — they are allowed in git by `.gitignore`).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → New app → select repo → main file: **`app/streamlit_app.py`** → Deploy.

Optional: deploy on **Render** using `render.yaml` (see `DEPLOY.md`).

## Replacing With Real Dataset
- Replace `data/raw/sample_crop_yield_data.csv` with your real dataset.
- Keep required columns (or adapt code):
  - `State`, `Crop`, `Season`
  - `Rainfall_mm`, `Temperature_C`, `Humidity_pct`
  - `Fertilizer_kg_per_ha`, `Area_hectare`
  - Target: `Yield_kg_per_ha`

## Reference Alignment (from your PDF)
- Problem statement: included.
- Objectives: included.
- Tools and technologies: included.
- Step-by-step flow: implemented in code.
- Model comparison + tuning + evaluation: implemented.
- Deployment with Streamlit: implemented.
- Expected output visuals and prediction interface: implemented.

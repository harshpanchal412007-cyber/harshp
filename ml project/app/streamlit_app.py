from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
# Pickle avoids joblib import issues on Streamlit Cloud when deps install oddly.
MODEL_PATH = ROOT / "models" / "best_crop_yield_model.pkl"
METRICS_PATH = ROOT / "models" / "model_metrics.json"
DATA_PATH = ROOT / "data" / "raw" / "sample_crop_yield_data.csv"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def main() -> None:
    st.set_page_config(page_title="Crop Yield Prediction", layout="wide")
    st.title("Crop Yield Prediction Dashboard")
    st.write(
        "Predict crop yield (kg/hectare) from state, crop, season and environmental inputs."
    )

    if not MODEL_PATH.exists():
        st.error("Model not found (best_crop_yield_model.pkl). Run: python src/train.py")
        st.stop()

    model = load_model()
    df = pd.read_csv(DATA_PATH)

    with st.sidebar:
        st.header("Input Features")
        state = st.selectbox("State", sorted(df["State"].unique().tolist()))
        crop = st.selectbox("Crop", sorted(df["Crop"].unique().tolist()))
        season = st.selectbox("Season", sorted(df["Season"].unique().tolist()))
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=800.0, step=10.0)
        temperature = st.number_input("Temperature (C)", min_value=0.0, value=28.0, step=0.5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=72.0, step=1.0)
        fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=0.0, value=180.0, step=5.0)
        area = st.number_input("Area (hectare)", min_value=0.1, value=2.0, step=0.1)

    input_df = pd.DataFrame(
        [
            {
                "State": state,
                "Crop": crop,
                "Season": season,
                "Rainfall_mm": rainfall,
                "Temperature_C": temperature,
                "Humidity_pct": humidity,
                "Fertilizer_kg_per_ha": fertilizer,
                "Area_hectare": area,
            }
        ]
    )

    if st.button("Predict Yield", type="primary"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Yield: {prediction:.2f} kg/hectare")

    st.subheader("Model Performance")
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.json(metrics)
    else:
        st.info("Metrics not available yet. Run training first.")

    st.subheader("Sample Data Preview")
    st.dataframe(df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()

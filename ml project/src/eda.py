from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "sample_crop_yield_data.csv"
REPORT_DIR = ROOT / "reports" / "figures"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_DATA_PATH)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.histplot(df["Yield_kg_per_ha"], bins=12, kde=True)
    plt.title("Distribution of Crop Yield")
    plt.xlabel("Yield (kg per ha)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "yield_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    corr_df = df[["Rainfall_mm", "Temperature_C", "Humidity_pct", "Fertilizer_kg_per_ha", "Yield_kg_per_ha"]]
    sns.heatmap(corr_df.corr(numeric_only=True), annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    top_states = (
        df.groupby("State", as_index=False)["Yield_kg_per_ha"]
        .mean()
        .sort_values("Yield_kg_per_ha", ascending=False)
        .head(10)
    )
    sns.barplot(data=top_states, x="State", y="Yield_kg_per_ha")
    plt.title("Top 10 States by Average Yield")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "top_states_yield.png", dpi=200)
    plt.close()

    print(f"EDA charts saved to {REPORT_DIR}")


if __name__ == "__main__":
    main()

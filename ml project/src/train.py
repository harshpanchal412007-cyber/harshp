from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor

    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "sample_crop_yield_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_crop_yield_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    target = "Yield_kg_per_ha"
    features = [col for col in df.columns if col != target]
    feature_df = df[features]
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in features if col not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)}


def train_and_compare(df: pd.DataFrame) -> tuple[Pipeline, dict[str, dict[str, float]]]:
    target = "Yield_kg_per_ha"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(df)

    models: dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
        )

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        results[model_name] = evaluate(y_test, preds)
        fitted[model_name] = pipeline

    best_model_name = max(results, key=lambda name: results[name]["R2"])
    best_pipeline = fitted[best_model_name]

    if best_model_name == "RandomForest":
        param_grid = {
            "model__n_estimators": [150, 250, 400],
            "model__max_depth": [None, 8, 12, 16],
            "model__min_samples_split": [2, 4, 8],
            "model__min_samples_leaf": [1, 2, 4],
        }
        search = RandomizedSearchCV(
            estimator=best_pipeline,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            scoring="r2",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        tuned_model = search.best_estimator_
        tuned_preds = tuned_model.predict(X_test)
        results["RandomForest_Tuned"] = evaluate(y_test, tuned_preds)
        if results["RandomForest_Tuned"]["R2"] >= results[best_model_name]["R2"]:
            best_pipeline = tuned_model
            best_model_name = "RandomForest_Tuned"

    results["BestModel"] = {"name": best_model_name}
    return best_pipeline, results


def save_feature_importance(best_pipeline: Pipeline) -> None:
    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_)
    else:
        return

    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)
    fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    best_pipeline, metrics = train_and_compare(df)

    joblib.dump(best_pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_feature_importance(best_pipeline)

    print("Training complete.")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()

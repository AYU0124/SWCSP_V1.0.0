from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


# =============================================================================
# User configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DATASET_FILE = str(DATA_DIR / "SS-ML.xlsx")
OUT_DIR = str(OUTPUT_ROOT / "Full_results")
TARGET_COL = 11            # Excel column L; zero-based column index 11
SHEET_NAME = 0             # Excel sheet index or name


# =============================================================================
# Final selected LightGBM hyperparameters
# =============================================================================
FINAL_LGBM_PARAMS: Dict = {
    "n_estimators": 300,
    "num_leaves": 40,
    "max_depth": 8,
    "min_child_samples": 5,
    "min_child_weight": 2.09530148472498,
    "learning_rate": 0.0816387640149208,
    "subsample": 0.9057622586883922,
    "colsample_bytree": 0.6666062946284652,
    "reg_alpha": 0.12053981756705028,
    "reg_lambda": 2.075540643120789,
    "random_state": 26,
    "verbosity": -1,
    "force_row_wise": True,
    "n_jobs": 1,
}


# =============================================================================
# Basic utilities
# =============================================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_table(file_path: str, sheet_name: int | str = 0) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use xlsx/xls/csv.")

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Loaded dataset is empty.")

    return df.copy()


def split_features_target(
    df: pd.DataFrame,
    target_col: Optional[str | int] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target; columns after the target are treated as metadata."""
    if target_col is None:
        target_idx = df.shape[1] - 1
    elif isinstance(target_col, int):
        target_idx = target_col
        if target_idx < 0:
            target_idx += df.shape[1]
        if target_idx < 0 or target_idx >= df.shape[1]:
            raise IndexError(
                f"target_col index {target_col} is out of range for "
                f"{df.shape[1]} columns."
            )
    else:
        if target_col not in df.columns:
            raise ValueError(
                f"target_col='{target_col}' not found in columns: {list(df.columns)}"
            )
        target_idx = int(df.columns.get_loc(target_col))

    if target_idx == 0:
        raise ValueError("No feature columns exist to the left of the target column.")

    X = df.iloc[:, :target_idx].copy()
    y = df.iloc[:, target_idx].copy()
    target_name = df.columns[target_idx]

    ignored_cols = list(df.columns[target_idx + 1:])
    if ignored_cols:
        print(f"[Info] Metadata columns ignored after target: {ignored_cols}")

    if X.shape[1] == 0:
        raise ValueError("No feature columns found.")

    return (
        X.reset_index(drop=True),
        pd.Series(y).reset_index(drop=True).rename(target_name),
    )


def validate_numeric_inputs(X: pd.DataFrame, y: pd.Series) -> None:
    non_numeric_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    if non_numeric_cols:
        raise TypeError(
            "All feature columns must be numeric. "
            f"Non-numeric columns found: {non_numeric_cols}"
        )

    if not pd.api.types.is_numeric_dtype(y):
        raise TypeError("Target column must be numeric.")


def align_features(input_df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    missing = [c for c in expected_features if c not in input_df.columns]
    extra = [c for c in input_df.columns if c not in expected_features]

    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    if extra:
        print(f"[Info] Extra columns ignored: {extra}")

    return input_df[expected_features].copy()


# =============================================================================
# Final model training
# =============================================================================
def build_final_model(params: Optional[Dict] = None) -> LGBMRegressor:
    used_params = FINAL_LGBM_PARAMS.copy()
    if params is not None:
        used_params.update(params)
    return LGBMRegressor(**used_params)


def train_full_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: Optional[Dict] = None,
) -> LGBMRegressor:
    model = build_final_model(params=params)
    model.fit(X, y)
    return model


def save_training_bundle(
    model: LGBMRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: str,
    source_data_path: str,
) -> None:
    out_path = Path(out_dir)
    ensure_dir(out_path)

    model_path = out_path / "final_lgbm_model.joblib"
    metadata_path = out_path / "final_lgbm_metadata.json"
    feature_template_path = out_path / "prediction_template.csv"

    joblib.dump(model, model_path)

    metadata = {
        "model_type": "LGBMRegressor",
        "source_data_file": Path(source_data_path).name,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "target_name": str(y.name),
        "final_params": FINAL_LGBM_PARAMS,
        "training_target_summary": {
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "mean": float(np.mean(y)),
            "std": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
        },
        "recommended_usage": {
            "prediction_input_must_match_feature_order": True,
            "web_deployment_supported": True,
        },
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    pd.DataFrame(columns=X.columns).to_csv(
        feature_template_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 90)
    print("Full-data training finished.")
    print(f"Model saved:     {model_path.name}")
    print(f"Metadata saved:  {metadata_path.name}")
    print(f"Template saved:  {feature_template_path.name}")
    print("=" * 90)


def load_trained_bundle(model_dir: str):
    model_dir_path = Path(model_dir)
    model = joblib.load(model_dir_path / "final_lgbm_model.joblib")

    with (model_dir_path / "final_lgbm_metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata


def predict_dataframe(input_df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    model, metadata = load_trained_bundle(model_dir)
    expected_features = metadata["feature_names"]
    X = align_features(input_df, expected_features)

    preds = model.predict(X)
    result = input_df.copy()
    result["Predicted_Target"] = preds
    return result


# =============================================================================
# Main processes
# =============================================================================
def main_train() -> None:
    df = read_table(DATASET_FILE, sheet_name=SHEET_NAME)
    X, y = split_features_target(df, target_col=TARGET_COL)
    validate_numeric_inputs(X, y)

    model = train_full_model(X, y, params=None)
    save_training_bundle(
        model=model,
        X=X,
        y=y,
        out_dir=OUT_DIR,
        source_data_path=DATASET_FILE,
    )


# =============================================================================
# Entry
# =============================================================================
if __name__ == "__main__":
    main_train()
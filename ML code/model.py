import json
import os
import warnings
from typing import Dict, Optional
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.base import clone


warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DEFAULT_BEST_JSON = str(
    OUTPUT_ROOT / "HPO" / "best_params_8models_hybrid.json"
)

MODEL_ORDER = ["Dummy", "Linear", "Ridge", "Lasso", "DT", "RF", "GB", "SVR", "KNN", "MLP", "XGB", "LGBM"]
SCALED_MODELS = {"Linear", "Ridge", "Lasso", "SVR", "KNN", "MLP"}

BASELINE_DEFAULTS = {
    "Dummy": {
        "Params": {"strategy": "mean"},
        "FeatureSpace": "RAW",
        "CV_RMSE": np.nan,
    },
    "Linear": {
        "Params": {},
        "FeatureSpace": "SCALED",
        "CV_RMSE": np.nan,
    },
    "Ridge": {
        "Params": {"alpha": 1.0},
        "FeatureSpace": "SCALED",
        "CV_RMSE": np.nan,
    },
    "Lasso": {
        "Params": {"alpha": 0.01, "max_iter": 10000},
        "FeatureSpace": "SCALED",
        "CV_RMSE": np.nan,
    },
}

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def MAPE(y_true, y_pred, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def SMAPE(y_true, y_pred, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < eps, eps, denom)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def calc_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(MAPE(y_true, y_pred)),
        "SMAPE": float(SMAPE(y_true, y_pred)),
        "EVS": float(explained_variance_score(y_true, y_pred)),
    }

def build_boosting_curve_logs(
    model_name,
    fitted_model,
    X_tr,
    y_tr,
    X_val,
    y_val,
    evals_result_override=None,
):
    X_tr_arr = np.asarray(X_tr)
    X_val_arr = np.asarray(X_val)
    y_tr_arr = np.asarray(y_tr).ravel()
    y_val_arr = np.asarray(y_val).ravel()

    train_rmse_hist = []
    valid_rmse_hist = []
    train_r2_hist = []
    valid_r2_hist = []

    if model_name == "XGB":
        raw_evals = fitted_model.evals_result()
        train_rmse = raw_evals.get("validation_0", {}).get("rmse", [])
        valid_rmse = raw_evals.get("validation_1", {}).get("rmse", [])
        n_rounds = min(len(train_rmse), len(valid_rmse))

        if n_rounds == 0:
            return None

        for i in range(1, n_rounds + 1):
            pred_tr = fitted_model.predict(X_tr_arr, iteration_range=(0, i))
            pred_val = fitted_model.predict(X_val_arr, iteration_range=(0, i))

            train_rmse_hist.append(float(np.sqrt(mean_squared_error(y_tr_arr, pred_tr))))
            valid_rmse_hist.append(float(np.sqrt(mean_squared_error(y_val_arr, pred_val))))
            train_r2_hist.append(float(r2_score(y_tr_arr, pred_tr)))
            valid_r2_hist.append(float(r2_score(y_val_arr, pred_val)))

    elif model_name == "LGBM":
        booster = fitted_model.booster_
        evals_result = evals_result_override if evals_result_override is not None else {}
        train_rmse = evals_result.get("training", {}).get("rmse", [])
        valid_rmse = evals_result.get("valid_1", {}).get("rmse", [])
        n_rounds = min(len(train_rmse), len(valid_rmse))

        if n_rounds == 0:
            return None

        for i in range(1, n_rounds + 1):
            pred_tr = booster.predict(X_tr_arr, num_iteration=i)
            pred_val = booster.predict(X_val_arr, num_iteration=i)

            train_rmse_hist.append(float(np.sqrt(mean_squared_error(y_tr_arr, pred_tr))))
            valid_rmse_hist.append(float(np.sqrt(mean_squared_error(y_val_arr, pred_val))))
            train_r2_hist.append(float(r2_score(y_tr_arr, pred_tr)))
            valid_r2_hist.append(float(r2_score(y_val_arr, pred_val)))

    else:
        return None

    return {
        "training": {
            "rmse": train_rmse_hist,
            "r2": train_r2_hist,
        },
        "valid_1": {
            "rmse": valid_rmse_hist,
            "r2": valid_r2_hist,
        }
    }


def flatten_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def _strip_pipeline_prefix(params: Dict) -> Dict:
    cleaned = {}
    for k, v in params.items():
        if k.startswith("model__"):
            cleaned[k.replace("model__", "", 1)] = v
        else:
            cleaned[k] = v
    return cleaned


def _normalize_hidden_layer_sizes(params: Dict) -> Dict:
    params = params.copy()
    if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], list):
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    return params


def _normalize_best_config(raw_config: Dict) -> Dict:
    if not isinstance(raw_config, dict):
        raise ValueError("Best model config JSON must be a dict.")

    normalized = {}
    for model_name in MODEL_ORDER:
        if model_name not in raw_config:
            continue
        item = raw_config[model_name]
        if not isinstance(item, dict):
            raise ValueError(f"Config for model '{model_name}' must be a dict.")
        if "Params" not in item:
            raise ValueError(f"Config for model '{model_name}' is missing key: 'Params'.")

        params = item["Params"]
        if not isinstance(params, dict):
            raise ValueError(f"'Params' for model '{model_name}' must be a dict.")

        params = _strip_pipeline_prefix(params)
        params = _normalize_hidden_layer_sizes(params)

        normalized[model_name] = {
            "FeatureSpace": str(item.get("FeatureSpace", "SCALED" if model_name in SCALED_MODELS else "RAW")).upper(),
            "CV_RMSE": float(item.get("CV_RMSE", np.nan)),
            "Params": params,
        }

    missing = [
        m for m in MODEL_ORDER
        if (m not in normalized) and (m not in BASELINE_DEFAULTS)
    ]
    if missing:
        raise ValueError(f"Best model config JSON is missing models: {missing}")

    return normalized


def load_best_model_config(best_json_path: Optional[str] = None) -> Dict:
    path = best_json_path or os.environ.get("HPO_BEST_JSON", DEFAULT_BEST_JSON)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Best-params JSON not found: {path}\n"
            f"Please run HPO.py first, or pass best_json_path explicitly."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return _normalize_best_config(raw)


def get_model_params(model_name: str, best_model_config: Dict) -> Dict:
    if model_name in BASELINE_DEFAULTS:
        params = BASELINE_DEFAULTS[model_name]["Params"].copy()
        params = _strip_pipeline_prefix(params)
        params = _normalize_hidden_layer_sizes(params)
        return params

    params = best_model_config[model_name]["Params"].copy()
    params = _strip_pipeline_prefix(params)
    params = _normalize_hidden_layer_sizes(params)
    return params


def get_feature_space(model_name: str, best_model_config: Dict) -> str:
    if model_name in BASELINE_DEFAULTS:
        return BASELINE_DEFAULTS[model_name]["FeatureSpace"]
    return best_model_config[model_name]["FeatureSpace"]


def get_search_cv_rmse(model_name: str, best_model_config: Dict) -> float:
    if model_name in BASELINE_DEFAULTS:
        return float(BASELINE_DEFAULTS[model_name]["CV_RMSE"])
    return float(best_model_config[model_name]["CV_RMSE"])


def needs_scaling(model_name: str, best_model_config: Dict) -> bool:
    feature_space = get_feature_space(model_name, best_model_config).upper()
    if feature_space == "SCALED":
        return True
    if feature_space == "RAW":
        return False
    return model_name in SCALED_MODELS


def _make_estimator(model_name: str, best_model_config: Dict, random_state: int = 26):
    params = get_model_params(model_name, best_model_config)

    if model_name == "Dummy":
        return DummyRegressor(**params)

    if model_name == "Linear":
        return LinearRegression(**params)

    if model_name == "Ridge":
        return Ridge(**params)

    if model_name == "Lasso":
        return Lasso(**params)

    if model_name == "DT":
        params["random_state"] = random_state
        return DecisionTreeRegressor(**params)

    if model_name == "RF":
        params["random_state"] = random_state
        return RandomForestRegressor(**params)

    if model_name == "GB":
        params["random_state"] = random_state
        return GradientBoostingRegressor(**params)

    if model_name == "SVR":
        return SVR(**params)

    if model_name == "KNN":
        return KNeighborsRegressor(**params)

    if model_name == "MLP":
        params["random_state"] = random_state
        return MLPRegressor(**params)

    if model_name == "XGB":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed.")
        params["random_state"] = random_state
        return XGBRegressor(**params)

    if model_name == "LGBM":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed.")
        params["random_state"] = random_state
        return LGBMRegressor(**params)

    raise ValueError(f"Unsupported model: {model_name}")


def build_model(
    model_name: str,
    use_scaler: Optional[bool] = None,
    random_state: int = 26,
    best_json_path: Optional[str] = None,
    best_model_config: Optional[Dict] = None,
):
    if best_model_config is None:
        best_model_config = load_best_model_config(best_json_path)

    estimator = _make_estimator(
        model_name=model_name,
        best_model_config=best_model_config,
        random_state=random_state,
    )

    if use_scaler is None:
        use_scaler = needs_scaling(model_name, best_model_config)

    if use_scaler:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
    return Pipeline([
        ("model", estimator),
    ])

def split_pipeline_for_boosting(fold_model, X_tr, y_tr, X_val):
    """
    Return:
        estimator, X_tr_fit, X_val_fit, preprocess_or_none
    """
    if isinstance(fold_model, Pipeline):
        est = fold_model.steps[-1][1]

        if len(fold_model.steps) > 1:
            preprocess = Pipeline(fold_model.steps[:-1])
            X_tr_fit = preprocess.fit_transform(X_tr, y_tr)
            X_val_fit = preprocess.transform(X_val)
        else:
            preprocess = None
            X_tr_fit, X_val_fit = X_tr, X_val
    else:
        est = fold_model
        preprocess = None
        X_tr_fit, X_val_fit = X_tr, X_val

    return est, X_tr_fit, X_val_fit, preprocess


def run_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv,
    use_scaler: Optional[bool] = None,
    random_state: int = 26,
    save_dir: Optional[str] = None,
    best_json_path: Optional[str] = None,
    best_model_config: Optional[Dict] = None,
):
    cv_pred = np.zeros(len(y_train))
    cv_evals_results = []
    cv_fold_models = []
    X_train = pd.DataFrame(X_train).copy()
    X_test = pd.DataFrame(X_test).copy()
    y_train = pd.Series(np.asarray(y_train).ravel()).copy()
    y_test = pd.Series(np.asarray(y_test).ravel()).copy()

    if best_model_config is None:
        best_model_config = load_best_model_config(best_json_path)

    if use_scaler is None:
        use_scaler = needs_scaling(model_name, best_model_config)

    model = build_model(
        model_name=model_name,
        use_scaler=use_scaler,
        random_state=random_state,
        best_model_config=best_model_config,
    )

    X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train

    for train_idx, val_idx in cv.split(X_train_arr, y_train_arr):
        X_tr, y_tr = X_train_arr[train_idx], y_train_arr[train_idx]
        X_val, y_val = X_train_arr[val_idx], y_train_arr[val_idx]

        fold_model = clone(model)

        if model_name == "XGB":
            xgb_est, X_tr_fit, X_val_fit, _ = split_pipeline_for_boosting(
                fold_model,
                X_tr,
                y_tr,
                X_val,
            )

            # Set the evaluation metric only for recording training dynamics.
            if (
                    hasattr(xgb_est, "get_params")
                    and xgb_est.get_params().get("eval_metric") is None
            ):
                xgb_est.set_params(eval_metric="rmse")

            xgb_est.fit(
                X_tr_fit,
                y_tr,
                eval_set=[
                    (X_tr_fit, y_tr),
                    (X_val_fit, y_val),
                ],
                verbose=False,
            )

            fitted_boost_model = xgb_est
            cv_fold_models.append(xgb_est)

            if isinstance(fold_model, Pipeline):
                fold_model.steps[-1] = (
                    fold_model.steps[-1][0],
                    xgb_est,
                )
                cv_pred[val_idx] = fold_model.predict(X_val)
            else:
                cv_pred[val_idx] = xgb_est.predict(X_val)

            fold_curve_log = build_boosting_curve_logs(
                model_name="XGB",
                fitted_model=fitted_boost_model,
                X_tr=X_tr_fit,
                y_tr=y_tr,
                X_val=X_val_fit,
                y_val=y_val,
            )

            if fold_curve_log is not None:
                cv_evals_results.append(fold_curve_log)

        elif model_name == "LGBM":
            import lightgbm as lgb

            lgb_est, X_tr_fit, X_val_fit, _ = split_pipeline_for_boosting(
                fold_model, X_tr, y_tr, X_val
            )

            evals_result = {}

            lgb_est.fit(
                X_tr_fit, y_tr,
                eval_set=[(X_tr_fit, y_tr), (X_val_fit, y_val)],
                eval_names=["training", "valid_1"],
                eval_metric="rmse",
                callbacks=[lgb.record_evaluation(evals_result)],
            )
            fitted_boost_model = lgb_est
            cv_fold_models.append(lgb_est)

            if isinstance(fold_model, Pipeline):
                fold_model.steps[-1] = (fold_model.steps[-1][0], lgb_est)
                cv_pred[val_idx] = fold_model.predict(X_val)
            else:
                cv_pred[val_idx] = lgb_est.predict(X_val)

            fold_curve_log = build_boosting_curve_logs(
                model_name="LGBM",
                fitted_model=fitted_boost_model,
                X_tr=X_tr_fit,
                y_tr=y_tr,
                X_val=X_val_fit,
                y_val=y_val,
                evals_result_override=evals_result,
            )
            if fold_curve_log is not None:
                cv_evals_results.append(fold_curve_log)
        else:
            fold_model.fit(X_tr, y_tr)
            cv_pred[val_idx] = fold_model.predict(X_val)

    cv_metrics = calc_metrics(y_train, cv_pred)

    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = calc_metrics(y_train, train_pred)
    test_metrics = calc_metrics(y_test, test_pred)

    result = {
        "Model": model_name,
        "FeatureSpace": "SCALED" if use_scaler else "RAW",
        "SearchCV_RMSE": get_search_cv_rmse(model_name, best_model_config),
        "BestParams": get_model_params(model_name, best_model_config),
        "CV": cv_metrics,
        "Train": train_metrics,
        "Test": test_metrics,
    }

    prediction_df = pd.DataFrame({
        "dataset": (["train"] * len(y_train)) + (["test"] * len(y_test)),
        "y_true": np.concatenate([y_train.values.ravel(), y_test.values.ravel()]),
        "y_pred": np.concatenate([np.asarray(train_pred).ravel(), np.asarray(test_pred).ravel()]),
    })

    cv_prediction_df = pd.DataFrame({
        "dataset": "cv_oof_train",
        "y_true": y_train.values.ravel(),
        "y_pred": np.asarray(cv_pred).ravel(),
    })

    if save_dir is not None:
        ensure_dir(save_dir)

        result_row = {
            "Model": model_name,
            "FeatureSpace": result["FeatureSpace"],
            "SearchCV_RMSE": result["SearchCV_RMSE"],
            "BestParams": json.dumps(result["BestParams"], ensure_ascii=False),
            **flatten_metrics(cv_metrics, "CV"),
            **flatten_metrics(train_metrics, "Train"),
            **flatten_metrics(test_metrics, "Test"),
        }

        pd.DataFrame([result_row]).to_csv(
            os.path.join(save_dir, f"metrics_{model_name}.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        prediction_df.to_csv(
            os.path.join(save_dir, f"predictions_{model_name}.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        cv_prediction_df.to_csv(
            os.path.join(save_dir, f"cv_predictions_{model_name}.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        joblib.dump(model, os.path.join(save_dir, f"model_{model_name}.pkl"))

    return {
        "result": result,
        "model": model,
        "train_pred": np.asarray(train_pred).ravel(),
        "test_pred": np.asarray(test_pred).ravel(),
        "cv_pred": np.asarray(cv_pred).ravel(),
        "prediction_df": prediction_df,
        "cv_prediction_df": cv_prediction_df,
        "cv_evals_results": cv_evals_results,
        "cv_fold_models": cv_fold_models,
    }


def run_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv,
    save_root: str = "./Results_Final/",
    random_state: int = 26,
    best_json_path: Optional[str] = None,
):
    best_model_config = load_best_model_config(best_json_path)

    outputs = {}
    summary_rows = []

    for model_name in MODEL_ORDER:
        model_dir = os.path.join(save_root, model_name)
        ensure_dir(model_dir)
        ensure_dir(os.path.join(model_dir, "plots"))

        use_scaler = needs_scaling(model_name, best_model_config)

        print("=" * 90)
        print(
            f"Running {model_name} | "
            f"FeatureSpace={get_feature_space(model_name, best_model_config)} | "
            f"use_scaler={use_scaler}"
        )
        print("=" * 90)

        out = run_single_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv=cv,
            use_scaler=use_scaler,
            random_state=random_state,
            save_dir=model_dir,
            best_model_config=best_model_config,
        )
        outputs[model_name] = out

        row = {
            "Model": model_name,
            "FeatureSpace": out["result"]["FeatureSpace"],
            "SearchCV_RMSE": out["result"]["SearchCV_RMSE"],
            "BestParams": json.dumps(out["result"]["BestParams"], ensure_ascii=False),
            **flatten_metrics(out["result"]["CV"], "CV"),
            **flatten_metrics(out["result"]["Train"], "Train"),
            **flatten_metrics(out["result"]["Test"], "Test"),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("Test_RMSE", ascending=True).reset_index(drop=True)
    ensure_dir(save_root)
    summary_df.to_csv(os.path.join(save_root, "summary_metrics.csv"), index=False, encoding="utf-8-sig")
    summary_df.to_excel(os.path.join(save_root, "summary_metrics.xlsx"), index=False)

    return outputs, summary_df


if __name__ == "__main__":
    cfg = load_best_model_config()
    print("Loaded best model config from HPO JSON successfully.")
    print("Models:", list(cfg.keys()))

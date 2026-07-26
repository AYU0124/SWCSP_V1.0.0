import json
import os
import warnings
from copy import deepcopy
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ParameterGrid, cross_validate
from data_processing import load_and_preprocess_data, make_stratified_regression_cv


# Suppress non-critical warnings from model training and cross-validation.
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


# ========================= User configuration =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DATASET_FILE = str(DATA_DIR / "SS-ML.xlsx")
OUT_DIR = str(OUTPUT_ROOT / "HPO")

TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 52
USE_STRATIFIED_REGRESSION_SPLIT = True
N_STRATA_BINS = 5

CV_SPLITS = 10
CV_SHUFFLE = True
CV_RANDOM_STATE = 26
CV_N_STRATA_BINS = 5
MODEL_RANDOM_STATE = 26

PENALTY_WEIGHT = 25.0
PENALTY_GAP_THRESHOLD = 0.04
RMSE_TOL = 0.3

SCALED_MODELS = {"SVR", "KNN", "MLP"}

TOPK_FROM_COARSE = 8
MAX_FINE_CANDIDATES_PER_PARAM = 3

TRIALS_MLP = 200
TRIALS_XGB = 200
TRIALS_LGBM = 200

N_JOBS_CV = -1

# ========================= Basic utilities =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ========================= Cross-validation and scoring =========================
def make_cv():
    return make_stratified_regression_cv(
        n_splits=CV_SPLITS,
        shuffle=CV_SHUFFLE,
        random_state=CV_RANDOM_STATE,
        n_strata_bins=CV_N_STRATA_BINS,
    )


def evaluate_candidate_metrics(model, X, y, kf):
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    cvres = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=N_JOBS_CV,
        error_score="raise",
    )

    train_r2 = float(np.mean(cvres["train_r2"]))
    cv_r2 = float(np.mean(cvres["test_r2"]))
    cv_rmse = float(-np.mean(cvres["test_rmse"]))

    generalization_gap = float(train_r2 - cv_r2)
    penalty = float(PENALTY_WEIGHT * max(0.0, generalization_gap - PENALTY_GAP_THRESHOLD))

    return {
        "Primary_Score": cv_rmse,
        "CV_RMSE": cv_rmse,
        "Train_R2": train_r2,
        "CV_R2": cv_r2,
        "Generalization_Gap": generalization_gap,
        "Penalty": penalty,
        "RawCV": cvres,
    }


# ========================= Fine-grid helper functions =========================
def _unique_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        key = x if isinstance(x, (str, int, float, tuple)) else str(x)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def _neighbors_int(v: int):
    if v is None:
        return []
    if v <= 1:
        return [1, 2]
    return [max(1, v - 1), v, v + 1]


def _neighbors_float(v: float, low=None, high=None):
    if v is None:
        return []
    cand = [v - 0.1, v, v + 0.1]

    out = []
    for x in cand:
        if x <= 0:
            continue
        if low is not None and x < low:
            continue
        if high is not None and x > high:
            continue
        out.append(float(x))
    return out


def build_fine_grid_from_cvresults(cv_results: dict, coarse_grid: dict):
    params_list = cv_results["params"]
    ranks = cv_results["rank_test_score"]
    idx_sorted = np.argsort(ranks)[:TOPK_FROM_COARSE]
    top_params = [params_list[i] for i in idx_sorted]

    fine = {}
    for p, coarse_vals in coarse_grid.items():
        vals = [d.get(p) for d in top_params if p in d]
        vals = _unique_keep_order(vals)

        if len(vals) < 2:
            base = vals[0] if vals else (coarse_vals[0] if coarse_vals else None)

            if isinstance(base, int) and p not in {"random_state", "model__random_state"}:
                vals = _unique_keep_order(_neighbors_int(base))
            elif isinstance(base, float):
                if p in {"model__subsample", "model__colsample_bytree"}:
                    vals = _unique_keep_order(_neighbors_float(base, low=1e-6, high=1.0))
                elif p in {"model__learning_rate"}:
                    vals = _unique_keep_order(_neighbors_float(base, low=1e-6, high=None))
                else:
                    vals = _unique_keep_order(_neighbors_float(base))
            else:
                vals = _unique_keep_order(coarse_vals)
        fine[p] = vals[:MAX_FINE_CANDIDATES_PER_PARAM]

    return fine


# ========================= Hyperparameter search spaces =========================
def _coarse_spaces(n_features: int):
    spaces = {}

    spaces["DT"] = dict(
        ccp_alpha=[0.01, 0.05, 0.1, 0.2],
        max_depth=[3, 5, 7, None],
        max_features=[None, "sqrt"],
        min_samples_leaf=[2, 4, 6],
        min_samples_split=[4, 8, 12],
        splitter=["best"],
    )

    mf_alt = max(1, int(np.ceil(n_features / 3)))
    spaces["RF"] = dict(
        n_estimators=[100, 300, 500, 800],
        max_depth=[10, 15, 20, None],
        max_features=[mf_alt, "sqrt"],
        min_samples_leaf=[2, 4, 6, 8],
        min_samples_split=[4, 8, 12],
    )

    spaces["GB"] = dict(
        n_estimators=[100, 150, 250],
        learning_rate=[0.03, 0.05, 0.1],
        max_depth=[2, 3, 4],
        max_features=[None, "sqrt"],
        min_samples_leaf=[3, 5, 8],
        min_samples_split=[4, 8],
        subsample=[0.6, 0.8],
        alpha=[0.1],
    )

    spaces["SVR"] = dict(
        kernel=["rbf"],
        C=[10, 100, 1000, 3000],
        gamma=["scale", 0.001, 0.01, 0.1, 1.0],
        epsilon=[0.1, 0.2, 0.3, 0.5],
        max_iter=[3000],
    )

    spaces["KNN"] = dict(
        algorithm=["kd_tree", "auto"],
        leaf_size=[10, 20, 30],
        n_neighbors=[3, 5, 7],
        weights=["uniform", "distance"],
    )

    return spaces


def _bayes_spaces():
    spaces = {}

    spaces["MLP"] = dict(
        n_layers_low=1,
        n_layers_high=2,
        n_units_low=16,
        n_units_high=128,
        solver=["lbfgs", "adam"],
        max_iter_low=500,
        max_iter_high=2000,
        alpha_low=1e-4,
        alpha_high=1.0,
        learning_rate_init_low=1e-5,
        learning_rate_init_high=1e-2,
    )

    if XGBRegressor is not None:
        spaces["XGB"] = dict(
            n_estimators=[300, 500, 800, 1200],
            max_depth_low=3,
            max_depth_high=7,
            learning_rate_low=0.01,
            learning_rate_high=0.1,
            gamma_low=0.0,
            gamma_high=5.0,
            min_child_weight_low=1.0,
            min_child_weight_high=10.0,
            subsample_low=0.4,
            subsample_high=1.0,
            colsample_bytree_low=0.3,
            colsample_bytree_high=1.0,
            reg_alpha_low=0.01,
            reg_alpha_high=10.0,
            reg_lambda_low=0.1,
            reg_lambda_high=10.0,
        )

    if LGBMRegressor is not None:
        spaces["LGBM"] = dict(
            n_estimators = [100, 150, 200, 300, 400],
            num_leaves_low=15,
            num_leaves_high=40,
            max_depth_low=3,
            max_depth_high=9,
            min_child_samples_low=5,
            min_child_samples_high=50,
            min_child_weight_low=1.0,
            min_child_weight_high=10.0,
            learning_rate_low=0.01,
            learning_rate_high=0.1,
            subsample_low=0.5,
            subsample_high=1.0,
            colsample_bytree_low=0.5,
            colsample_bytree_high=1.0,
            reg_alpha_low=0.01,
            reg_alpha_high=10.0,
            reg_lambda_low=0.1,
            reg_lambda_high=10.0,
        )

    return spaces


# ========================= Grid-search model builders =========================
def _make_estimator_grid(model_name: str):
    rs = MODEL_RANDOM_STATE

    if model_name == "DT":
        return Pipeline([("model", DecisionTreeRegressor(random_state=rs))])

    if model_name == "RF":
        return Pipeline([("model", RandomForestRegressor(random_state=rs, n_jobs=1))])

    if model_name == "GB":
        return Pipeline([
            ("model", GradientBoostingRegressor(random_state=rs))
        ])

    if model_name == "SVR":
        return Pipeline([("scaler", StandardScaler()), ("model", SVR())])

    if model_name == "KNN":
        return Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())])

    raise ValueError(model_name)


def _grid_namespace(model_name: str, grid: dict) -> dict:
    return {f"model__{k}": v for k, v in grid.items()}


# ========================= Grid-search evaluation =========================
def evaluate_grid_candidates(estimator, param_grid: dict, X_train, y_train, kf):
    rows = []

    for idx, params in enumerate(ParameterGrid(param_grid), start=1):
        est = clone(estimator)
        est.set_params(**params)

        metrics = evaluate_candidate_metrics(est, X_train, y_train, kf)

        row = {
            "params": params,
            "mean_test_score": -metrics["CV_RMSE"],
            "mean_train_r2": metrics["Train_R2"],
            "mean_test_r2": metrics["CV_R2"],
            "mean_test_rmse": metrics["CV_RMSE"],
            "mean_gap": metrics["Generalization_Gap"],
            "mean_penalty": metrics["Penalty"],
            "candidate_index": idx,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["mean_test_rmse", "mean_penalty", "mean_gap"], ascending=[True, True, True]).reset_index(drop=True)
    df["rank_test_score"] = np.arange(1, len(df) + 1)

    best_rmse = float(df.iloc[0]["mean_test_rmse"])
    near_best = df[df["mean_test_rmse"] <= best_rmse + RMSE_TOL].copy()
    near_best = near_best.sort_values(["mean_penalty", "mean_gap", "mean_test_rmse"], ascending=[True, True, True]).reset_index(drop=True)

    best_row = near_best.iloc[0].to_dict()
    cv_results_like = {
        "params": df["params"].tolist(),
        "rank_test_score": df["rank_test_score"].tolist(),
    }
    return best_row, df, cv_results_like


def run_two_stage_grid(model_name: str, X_train, y_train, kf, coarse_grid: dict):
    est1 = _make_estimator_grid(model_name)
    coarse_best_row, coarse_df, coarse_cv_results = evaluate_grid_candidates(est1, coarse_grid, X_train, y_train, kf)

    coarse_best_params = coarse_best_row["params"]
    fine_grid = build_fine_grid_from_cvresults(coarse_cv_results, coarse_grid)
    for k, v in coarse_best_params.items():
        if k in fine_grid and v not in fine_grid[k]:
            fine_grid[k] = _unique_keep_order([v] + fine_grid[k])[:MAX_FINE_CANDIDATES_PER_PARAM]

    est2 = _make_estimator_grid(model_name)
    fine_best_row, fine_df, _ = evaluate_grid_candidates(est2, fine_grid, X_train, y_train, kf)

    return {
        "Model": model_name,
        "Best_Primary_Score": float(fine_best_row["mean_test_rmse"]),
        "Best_CV_RMSE": float(fine_best_row["mean_test_rmse"]),
        "Best_Train_R2": float(fine_best_row["mean_train_r2"]),
        "Best_CV_R2": float(fine_best_row["mean_test_r2"]),
        "Best_Gap": float(fine_best_row["mean_gap"]),
        "Best_Penalty": float(fine_best_row["mean_penalty"]),
        "BestParams": fine_best_row["params"],
        "CoarseBestParams": coarse_best_params,
        "FineGrid": fine_grid,
    }, coarse_df.head(30), fine_df.head(30)


# ========================= Bayesian-search model builders =========================
def build_bayes_model_from_trial(trial: optuna.trial.Trial, model_name: str, bayes_spaces: dict):
    rs = MODEL_RANDOM_STATE
    sp = bayes_spaces[model_name]

    if model_name == "MLP":
        n_layers = trial.suggest_int("n_layers", sp["n_layers_low"], sp["n_layers_high"])
        if n_layers == 1:
            hidden_layer_sizes = (trial.suggest_int("n_units_l0", 16, 128, step=16),)
        else:
            n0 = trial.suggest_int("n_units_l0", 32, 128, step=16)
            n1 = trial.suggest_int("n_units_l1", 16, n0, step=16)
            hidden_layer_sizes = (n0, n1)

        solver = trial.suggest_categorical("solver", sp["solver"])
        max_iter = trial.suggest_int("max_iter", sp["max_iter_low"], sp["max_iter_high"])
        alpha = trial.suggest_float("alpha", sp["alpha_low"], sp["alpha_high"], log=True)

        params = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": "relu",
            "solver": solver,
            "alpha": float(alpha),
            "max_iter": int(max_iter),
            "random_state": rs,
        }
        if solver == "adam":
            params["learning_rate_init"] = float(
                trial.suggest_float("learning_rate_init", sp["learning_rate_init_low"], sp["learning_rate_init_high"], log=True)
            )

        base_model = MLPRegressor(**params)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", base_model),
        ]), params

    if model_name == "XGB":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed")

        params = {
            "n_estimators": int(trial.suggest_categorical("n_estimators", sp["n_estimators"])),
            "learning_rate": float(trial.suggest_float("learning_rate", sp["learning_rate_low"], sp["learning_rate_high"], log=True)),
            "max_depth": int(trial.suggest_int("max_depth", sp["max_depth_low"], sp["max_depth_high"])),
            "gamma": float(trial.suggest_float("gamma", sp["gamma_low"], sp["gamma_high"])),
            "min_child_weight": float(trial.suggest_float("min_child_weight", sp["min_child_weight_low"], sp["min_child_weight_high"], log=True)),
            "subsample": float(trial.suggest_float("subsample", sp["subsample_low"], sp["subsample_high"])),
            "colsample_bytree": float(trial.suggest_float("colsample_bytree", sp["colsample_bytree_low"], sp["colsample_bytree_high"])),
            "reg_alpha": float(trial.suggest_float("reg_alpha", sp["reg_alpha_low"], sp["reg_alpha_high"], log=True)),
            "reg_lambda": float(trial.suggest_float("reg_lambda", sp["reg_lambda_low"], sp["reg_lambda_high"], log=True)),
            "objective": "reg:squarederror",
            "random_state": rs,
            "tree_method": "hist",
            "n_jobs": 1,
        }

        base_model = XGBRegressor(**params)
        return Pipeline([("model", base_model)]), params

    if model_name == "LGBM":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed")

        params = {
            "n_estimators": int(trial.suggest_categorical("n_estimators", sp["n_estimators"])),
            "num_leaves": int(trial.suggest_int("num_leaves", sp["num_leaves_low"], sp["num_leaves_high"])),
            "max_depth": int(trial.suggest_int("max_depth", sp["max_depth_low"], sp["max_depth_high"])),
            "min_child_samples": int(trial.suggest_int("min_child_samples", sp["min_child_samples_low"], sp["min_child_samples_high"])),
            "min_child_weight": float(trial.suggest_float("min_child_weight", sp["min_child_weight_low"], sp["min_child_weight_high"], log=True)),
            "learning_rate": float(trial.suggest_float("learning_rate", sp["learning_rate_low"], sp["learning_rate_high"], log=True)),
            "subsample": float(trial.suggest_float("subsample", sp["subsample_low"], sp["subsample_high"])),
            "colsample_bytree": float(trial.suggest_float("colsample_bytree", sp["colsample_bytree_low"], sp["colsample_bytree_high"])),
            "reg_alpha": float(trial.suggest_float("reg_alpha", sp["reg_alpha_low"], sp["reg_alpha_high"], log=True)),
            "reg_lambda": float(trial.suggest_float("reg_lambda", sp["reg_lambda_low"], sp["reg_lambda_high"], log=True)),
            "random_state": rs,
            "verbosity": -1,
            "force_row_wise": True,
            "n_jobs": 1,
        }

        base_model = LGBMRegressor(**params)
        return Pipeline([("model", base_model)]), params

    raise ValueError(model_name)


# ========================= Bayesian-search evaluation =========================
def run_bayes(model_name: str, X_train, y_train, kf, bayes_spaces: dict, n_trials: int):
    def objective(trial: optuna.trial.Trial):
        model, _ = build_bayes_model_from_trial(trial, model_name, bayes_spaces)
        metrics = evaluate_candidate_metrics(model, X_train, y_train, kf)
        trial.set_user_attr("cv_rmse", metrics["CV_RMSE"])
        trial.set_user_attr("train_r2", metrics["Train_R2"])
        trial.set_user_attr("cv_r2", metrics["CV_R2"])
        trial.set_user_attr("gap", metrics["Generalization_Gap"])
        trial.set_user_attr("penalty", metrics["Penalty"])
        return metrics["CV_RMSE"] + 0.2 * metrics["Penalty"]

    sampler = optuna.samplers.TPESampler(seed=CV_RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    trials_ok = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get("cv_rmse") is not None
    ]
    best_rmse = min(t.user_attrs["cv_rmse"] for t in trials_ok)
    near_best = [t for t in trials_ok if t.user_attrs["cv_rmse"] <= best_rmse + RMSE_TOL]
    selected_trial = sorted(
        near_best,
        key=lambda t: (
            t.user_attrs.get("penalty", np.inf),
            t.user_attrs.get("gap", np.inf),
            t.user_attrs.get("cv_rmse", np.inf),
        ),
    )[0]

    fixed = optuna.trial.FixedTrial(selected_trial.params)
    _, full_params = build_bayes_model_from_trial(fixed, model_name, bayes_spaces)

    return {
        "Model": model_name,
        "Best_CV_RMSE": float(selected_trial.user_attrs.get("cv_rmse", np.nan)),
        "Best_Train_R2": float(selected_trial.user_attrs.get("train_r2", np.nan)),
        "Best_CV_R2": float(selected_trial.user_attrs.get("cv_r2", np.nan)),
        "Best_Gap": float(selected_trial.user_attrs.get("gap", np.nan)),
        "Best_Penalty": float(selected_trial.user_attrs.get("penalty", np.nan)),
        "BestParams": full_params,
        "Trials": n_trials,
    }, study


# ========================= Main workflow =========================
def main():
    # Prepare output directory.
    ensure_dir(OUT_DIR)

    # Load data and create the train/test split.
    pack = load_and_preprocess_data(
        DATASET_FILE,
        test_size=TEST_SIZE,
        split_random_state=SPLIT_RANDOM_STATE,
        n_strata_bins=N_STRATA_BINS,
        use_stratified_regression_split=USE_STRATIFIED_REGRESSION_SPLIT,
    )

    X_train_raw = pack["X_train_raw"]
    y_train = np.ravel(pack["y_train"])

    pd.DataFrame(pack["split_summary_df"]).to_csv(
        os.path.join(OUT_DIR, "hpo_split_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Build search spaces and the cross-validation splitter.
    n_features = int(X_train_raw.shape[1])
    grid_spaces = _coarse_spaces(n_features=n_features)
    bayes_spaces = _bayes_spaces()
    cv = make_cv()

    grid_models = ["DT", "RF", "GB", "SVR", "KNN"]
    bayes_models = ["MLP", "XGB", "LGBM"]
    if XGBRegressor is None and "XGB" in bayes_models:
        bayes_models.remove("XGB")
    if LGBMRegressor is None and "LGBM" in bayes_models:
        bayes_models.remove("LGBM")

    summary_rows = []
    best_params_out = {}

    # Run two-stage grid search for classical models.
    for m in grid_models:
        use_scaled = m in SCALED_MODELS
        print("\n" + "=" * 90)
        print(f"[GRID] {m} | FeatureSpace: {'SCALED' if use_scaled else 'RAW'}")
        print("=" * 90)

        coarse_grid = deepcopy(grid_spaces[m])
        if m == "GB":
            try:
                _ = GradientBoostingRegressor(max_depth=3)
            except TypeError:
                coarse_grid.pop("max_depth", None)
        coarse_grid = _grid_namespace(m, coarse_grid)

        result, coarse_df, fine_df = run_two_stage_grid(m, X_train_raw, y_train, cv, coarse_grid)
        model_dir = os.path.join(OUT_DIR, m)
        ensure_dir(model_dir)
        coarse_df.to_csv(os.path.join(model_dir, "coarse_top30.csv"), index=False, encoding="utf-8-sig")
        fine_df.to_csv(os.path.join(model_dir, "fine_top30.csv"), index=False, encoding="utf-8-sig")

        best_params_out[m] = {
            "FeatureSpace": "SCALED" if use_scaled else "RAW",
            "Primary_Score": result["Best_Primary_Score"],
            "CV_RMSE": result["Best_CV_RMSE"],
            "Train_R2": result["Best_Train_R2"],
            "CV_R2": result["Best_CV_R2"],
            "Penalty": result["Best_Penalty"],
            "Params": result["BestParams"],
        }

        summary_rows.append({
            "Model": m,
            "FeatureSpace": "SCALED" if use_scaled else "RAW",
            "Method": "GRID",
            "CV_RMSE_Best": result["Best_CV_RMSE"],
            "Train_R2_Best": result["Best_Train_R2"],
            "CV_R2_Best": result["Best_CV_R2"],
            "Gap_Best": result["Best_Gap"],
            "Penalty_Best": result["Best_Penalty"],
            "BestParams": json.dumps(result["BestParams"], ensure_ascii=False),
        })

    trials_map = {"MLP": TRIALS_MLP, "XGB": TRIALS_XGB, "LGBM": TRIALS_LGBM}

    # Run Bayesian optimization for higher-capacity models.
    for m in bayes_models:
        use_scaled = m in SCALED_MODELS
        n_trials = trials_map[m]
        print("\n" + "=" * 90)
        print(f"[BAYES] {m} | FeatureSpace: {'SCALED' if use_scaled else 'RAW'} | Trials={n_trials}")
        print("=" * 90)

        result, study = run_bayes(m, X_train_raw, y_train, cv, bayes_spaces, n_trials=n_trials)
        model_dir = os.path.join(OUT_DIR, m)
        ensure_dir(model_dir)
        trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        if len(study.trials) > 0:
            extra_df = pd.DataFrame([
                {
                    "number": t.number,
                    "cv_rmse": t.user_attrs.get("cv_rmse", np.nan),
                    "train_r2": t.user_attrs.get("train_r2", np.nan),
                    "cv_r2": t.user_attrs.get("cv_r2", np.nan),
                    "gap": t.user_attrs.get("gap", np.nan),
                    "penalty": t.user_attrs.get("penalty", np.nan),
                }
                for t in study.trials
            ])
            trials_df = trials_df.merge(extra_df, on="number", how="left")
        trials_df.to_csv(os.path.join(model_dir, "trials.csv"), index=False, encoding="utf-8-sig")

        best_params_out[m] = {
            "FeatureSpace": "SCALED" if use_scaled else "RAW",
            "Primary_Score": result["Best_CV_RMSE"],
            "CV_RMSE": result["Best_CV_RMSE"],
            "Train_R2": result["Best_Train_R2"],
            "CV_R2": result["Best_CV_R2"],
            "Gap": result["Best_Gap"],
            "Penalty": result["Best_Penalty"],
            "Params": result["BestParams"],
        }

        summary_rows.append({
            "Model": m,
            "FeatureSpace": "SCALED" if use_scaled else "RAW",
            "Method": "BAYES",
            "CV_RMSE_Best": result["Best_CV_RMSE"],
            "Train_R2_Best": result["Best_Train_R2"],
            "CV_R2_Best": result["Best_CV_R2"],
            "Gap_Best": result["Best_Gap"],
            "Penalty_Best": result["Best_Penalty"],
            "BestParams": json.dumps(result["BestParams"], ensure_ascii=False),
            "Trials": n_trials,
        })

    # Save final summaries and selected parameters.
    summary_df = pd.DataFrame(summary_rows).sort_values(["CV_RMSE_Best", "Penalty_Best"], ascending=[True, True]).reset_index(drop=True)
    summary_df.to_csv(os.path.join(OUT_DIR, "hyperparam_hybrid_summary.csv"), index=False, encoding="utf-8-sig")
    summary_df.to_excel(os.path.join(OUT_DIR, "hyperparam_hybrid_summary.xlsx"), index=False)

    with open(os.path.join(OUT_DIR, "best_params_8models_hybrid.json"), "w", encoding="utf-8") as f:
        json.dump(best_params_out, f, ensure_ascii=False, indent=2)

    print("\n" + "-" * 90)
    print("Done. Hybrid hyperparameter search outputs:")
    print(os.path.join(OUT_DIR, "hyperparam_hybrid_summary.csv"))
    print(os.path.join(OUT_DIR, "hyperparam_hybrid_summary.xlsx"))
    print(os.path.join(OUT_DIR, "best_params_8models_hybrid.json"))
    print("-" * 90)


if __name__ == "__main__":
    main()

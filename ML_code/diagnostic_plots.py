# -*- coding: utf-8 -*-

import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

try:
    from sklearn.pipeline import Pipeline
except Exception:  # pragma: no cover
    Pipeline = None

warnings.filterwarnings("ignore")

DPI = 600
FIGURE2_FIGSIZE = (8.5, 6.5)
FONT_FAMILY = "Times New Roman"
BASE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 16

GLOBAL_FONTWEIGHT = "bold"
LABEL_FONTWEIGHT = "bold"
TICK_FONTWEIGHT = "bold"

# Shared plotting colors.
BLUE = "#406DA8"
RED = "#BC5D59"
MID_GREY = "#6B7280"
GRID_COLOR = "#7A7A7A"
GRID_ALPHA = 0.35
GRID_LINEWIDTH = 0.6
SPINE_LINEWIDTH = 1.5
LINEWIDTH = 1.5

COLOR_SCORE_TRAIN = BLUE
COLOR_SCORE_VALID = RED
COLOR_LOSS_TRAIN = BLUE
COLOR_LOSS_VALID = RED

# Shared panel geometry.
BOX_LEFT = 0.13
BOX_BOTTOM = 0.14
BOX_WIDTH = 0.73
BOX_HEIGHT = 0.76

LEGEND_FACE_COLOR = "white"
LEGEND_EDGE_COLOR = "#BBBBBB"
LEGEND_FRAME_ALPHA = 0.90

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = BASE_FONTSIZE
plt.rcParams["font.weight"] = GLOBAL_FONTWEIGHT
plt.rcParams["axes.labelweight"] = LABEL_FONTWEIGHT
plt.rcParams["axes.unicode_minus"] = False


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def style_axis(ax, grid: bool = True) -> None:
    """Apply the same bold academic axis style used in the reference figure."""
    if grid:
        ax.grid(True, linestyle="--", linewidth=GRID_LINEWIDTH, color=GRID_COLOR, alpha=GRID_ALPHA)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_LINEWIDTH)
    ax.tick_params(axis="both", which="major", length=5, width=1.0, labelsize=TICK_FONTSIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(TICK_FONTWEIGHT)
        tick.set_family(FONT_FAMILY)


def add_figure1_like_legend(
    ax,
    handles=None,
    labels=None,
    loc="upper left",
    ncol=1,
    frameon=True,
    **kwargs,
):
    """Legend style matched to the second plot in figure_output1.py."""
    legend_kwargs = dict(
        frameon=frameon,
        fontsize=LEGEND_FONTSIZE,
        loc=loc,
        ncol=ncol,
        borderpad=0.3,
        facecolor=LEGEND_FACE_COLOR,
        edgecolor=LEGEND_EDGE_COLOR,
        framealpha=LEGEND_FRAME_ALPHA,
    )
    legend_kwargs.update(kwargs)
    if handles is not None:
        legend_kwargs["handles"] = handles
    if labels is not None:
        legend_kwargs["labels"] = labels

    legend = ax.legend(**legend_kwargs)
    if legend is not None:
        legend.get_frame().set_linewidth(1.0)
        for txt in legend.get_texts():
            txt.set_fontweight("normal")
            txt.set_family(FONT_FAMILY)
    return legend


def add_shap_center_legend(
    ax,
    handles=None,
    labels=None,
    ncol=1,
    frameon=True,
    **kwargs,
):
    """Place the legend exactly at the center of the plotting area."""
    legend = add_figure1_like_legend(
        ax,
        handles=handles,
        labels=labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        bbox_transform=ax.transAxes,
        ncol=ncol,
        frameon=frameon,
        **kwargs,
    )
    if legend is not None:
        legend.set_zorder(30)
    return legend

def save_figure(fig, output_dir: str, file_stem: str) -> None:
    ensure_dir(output_dir)


    png_path = os.path.join(output_dir, f"{file_stem}.png")
    pdf_path = os.path.join(output_dir, f"{file_stem}.pdf")
    svg_path = os.path.join(output_dir, f"{file_stem}.svg")
    # Do not use bbox_inches="tight" so every plot keeps the same SHAP-contribution reference geometry.
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(pdf_path, dpi=DPI)
    fig.savefig(svg_path, dpi=DPI)
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {svg_path}")


def _is_pipeline(estimator) -> bool:
    return Pipeline is not None and isinstance(estimator, Pipeline)


def _get_model_name(estimator) -> str:
    if _is_pipeline(estimator):
        return estimator.steps[-1][1].__class__.__name__
    return estimator.__class__.__name__


def _safe_series(arr):
    return np.asarray(arr).ravel()


def _normalize_model_name(model_name: str, estimator) -> str:
    if model_name is not None and str(model_name).strip():
        return str(model_name).strip()
    return _get_model_name(estimator)


def _display_name(model_name: str) -> str:
    mapping = {
        "XGB": "XGBoost",
        "XGBOOST": "XGBoost",
        "LGBM": "LGBM",
        "LIGHTGBM": "LightGBM",
        "GB": "GradientBoosting",
        "RF": "RandomForest",
        "DT": "DecisionTree",
    }
    key = str(model_name).upper()
    return mapping.get(key, model_name)


def _extract_curve_from_single_result(single_fold_result: dict):
    if not isinstance(single_fold_result, dict) or not single_fold_result:
        return None

    train_block = None
    valid_block = None

    for key, value in single_fold_result.items():
        key_l = str(key).lower()
        if not isinstance(value, dict):
            continue
        if train_block is None and ("train" in key_l or key_l in {"validation_0", "training"}):
            train_block = value
        elif valid_block is None and ("valid" in key_l or key_l in {"validation_1", "valid_0", "valid_1"}):
            valid_block = value

    if train_block is None or valid_block is None:
        dict_values = [v for v in single_fold_result.values() if isinstance(v, dict)]
        if len(dict_values) >= 2:
            train_block = dict_values[0]
            valid_block = dict_values[1]

    if train_block is None or valid_block is None:
        return None

    parsed = {"train": {}, "valid": {}}
    for metric_name, values in train_block.items():
        parsed["train"][metric_name] = np.asarray(values, dtype=float)
    for metric_name, values in valid_block.items():
        parsed["valid"][metric_name] = np.asarray(values, dtype=float)
    return parsed


def _aggregate_cv_evals(model_cv_evals_results):
    if not model_cv_evals_results:
        return None

    parsed_folds = []
    for one in model_cv_evals_results:
        parsed = _extract_curve_from_single_result(one)
        if parsed is not None:
            parsed_folds.append(parsed)

    if not parsed_folds:
        return None

    common_metrics = set(parsed_folds[0]["train"].keys()) & set(parsed_folds[0]["valid"].keys())
    for parsed in parsed_folds[1:]:
        common_metrics &= set(parsed["train"].keys()) & set(parsed["valid"].keys())

    if not common_metrics:
        return None

    aggregated = {}
    for metric_name in sorted(common_metrics):
        train_arrays = [fold["train"][metric_name] for fold in parsed_folds if metric_name in fold["train"]]
        valid_arrays = [fold["valid"][metric_name] for fold in parsed_folds if metric_name in fold["valid"]]
        if not train_arrays or not valid_arrays:
            continue

        min_len = min(min(len(x) for x in train_arrays), min(len(x) for x in valid_arrays))
        train_mat = np.array([x[:min_len] for x in train_arrays], dtype=float)
        valid_mat = np.array([x[:min_len] for x in valid_arrays], dtype=float)

        aggregated[metric_name] = {
            "rounds": np.arange(1, min_len + 1),
            "train_mean": train_mat.mean(axis=0),
            "valid_mean": valid_mat.mean(axis=0),
            "train_std": train_mat.std(axis=0),
            "valid_std": valid_mat.std(axis=0),
            "train_all": train_mat,
            "valid_all": valid_mat,
        }
    return aggregated if aggregated else None


def _make_regression_strata(y, n_strata_bins=5):
    y = np.asarray(y).ravel()
    n_unique = np.unique(y).size
    effective_bins = max(2, min(int(n_strata_bins), int(n_unique)))

    try:
        ranks = np.argsort(np.argsort(y))
        scaled = ranks / max(len(y) - 1, 1)
        strata = np.floor(scaled * effective_bins).astype(int)
        strata = np.clip(strata, 0, effective_bins - 1)
        return strata
    except Exception:
        return np.zeros_like(y, dtype=int)


def _fixed_train_val_split(X, y, val_size=0.2, random_state=26, n_strata_bins=5):
    y_arr = _safe_series(y)
    strata = _make_regression_strata(y_arr, n_strata_bins=n_strata_bins)
    return train_test_split(
        X,
        y_arr,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=strata,
    )


def _extract_eval_result_from_fitted_estimator(estimator):
    candidate_objects = [estimator]
    if _is_pipeline(estimator):
        candidate_objects.append(estimator.steps[-1][1])

    for obj in candidate_objects:
        try:
            evals = obj.evals_result_
            if isinstance(evals, dict) and evals:
                return evals
        except Exception:
            pass
        try:
            evals = obj.evals_result()
            if isinstance(evals, dict) and evals:
                return evals
        except Exception:
            pass
    return None


def lgb_r2_eval(y_true, y_pred):
    return "r2", r2_score(y_true, y_pred), True


def _fit_with_eval_set(estimator, X_subtrain, y_subtrain, X_valid, y_valid):
    fit_estimator = clone(estimator)

    direct_attempts = [
        {
            "eval_set": [(X_subtrain, y_subtrain), (X_valid, y_valid)],
            "eval_metric": [lgb_r2_eval, "rmse"],
            "verbose": False,
        },
        {
            "eval_set": [(X_subtrain, y_subtrain), (X_valid, y_valid)],
            "eval_metric": [lgb_r2_eval, "rmse"],
        },
        {
            "eval_set": [(X_subtrain, y_subtrain), (X_valid, y_valid)],
            "verbose": False,
        },
        {
            "eval_set": [(X_subtrain, y_subtrain), (X_valid, y_valid)],
        },
    ]

    # Pass evaluation parameters to the final pipeline step when needed.
    pipeline_attempts = []
    if _is_pipeline(fit_estimator):
        last_step_name = fit_estimator.steps[-1][0]
        pipeline_attempts = [
            {
                f"{last_step_name}__eval_set": [(X_subtrain, y_subtrain), (X_valid, y_valid)],
                f"{last_step_name}__eval_metric": [lgb_r2_eval, "rmse"],
                f"{last_step_name}__verbose": False,
            },
            {
                f"{last_step_name}__eval_set": [(X_subtrain, y_subtrain), (X_valid, y_valid)],
                f"{last_step_name}__eval_metric": [lgb_r2_eval, "rmse"],
            },
            {
                f"{last_step_name}__eval_set": [(X_subtrain, y_subtrain), (X_valid, y_valid)],
                f"{last_step_name}__verbose": False,
            },
            {
                f"{last_step_name}__eval_set": [(X_subtrain, y_subtrain), (X_valid, y_valid)],
            },
        ]

    fit_attempts = direct_attempts + pipeline_attempts + [{}]

    last_error = None
    for params in fit_attempts:
        try:
            fit_estimator.fit(X_subtrain, y_subtrain, **params)
            return fit_estimator
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to fit estimator with fixed train/validation split. Last error: {last_error}"
    )


def plot_training_score_and_loss_curves(
    best_model,
    X_train,
    y_train,
    output_dir="./Plots",
    model_name="LGBM",
    val_size: float = 0.2,
    random_state: int = 26,
    n_strata_bins: int = 5,
):
    display_model_name = _display_name(_normalize_model_name(model_name, best_model))
    X_subtrain, X_valid, y_subtrain, y_valid = _fixed_train_val_split(
        X_train,
        y_train,
        val_size=val_size,
        random_state=random_state,
        n_strata_bins=n_strata_bins,
    )

    fitted_estimator = _fit_with_eval_set(best_model, X_subtrain, y_subtrain, X_valid, y_valid)
    evals_result = _extract_eval_result_from_fitted_estimator(fitted_estimator)
    aggregated = _aggregate_cv_evals([evals_result] if evals_result else None)

    if not aggregated:
        print("[Warning] Fixed train/validation eval history is empty or cannot be parsed. Skipping training curve plot.")
        return

    score_metric = None
    loss_metric = None
    for metric_name in aggregated.keys():
        name = str(metric_name).lower()
        if score_metric is None and "r2" in name:
            score_metric = metric_name
        if loss_metric is None and ("rmse" in name or "l2" in name or "mse" in name):
            loss_metric = metric_name

    if score_metric is None:
        print("[Warning] No R2 metric found in eval history. Score curve will be skipped.")
    if loss_metric is None:
        print("[Warning] No RMSE/L2-like metric found in eval history. Loss curve will be skipped.")
        return

    fallback_train_r2 = None
    fallback_valid_r2 = None
    if score_metric is None:
        try:
            n_rounds = len(aggregated[loss_metric]["rounds"])
            temp_train_r2 = []
            temp_valid_r2 = []

            for n in range(1, n_rounds + 1):
                est_n = clone(best_model)

                if hasattr(est_n, "n_estimators"):
                    est_n.set_params(n_estimators=n)
                elif hasattr(est_n, "num_boost_round"):
                    est_n.set_params(num_boost_round=n)
                elif _is_pipeline(est_n):
                    last_step_name = est_n.steps[-1][0]
                    last_estimator = est_n.steps[-1][1]
                    if hasattr(last_estimator, "n_estimators"):
                        est_n.set_params(**{f"{last_step_name}__n_estimators": n})
                    elif hasattr(last_estimator, "num_boost_round"):
                        est_n.set_params(**{f"{last_step_name}__num_boost_round": n})
                    else:
                        raise RuntimeError("Pipeline final estimator does not support iteration-wise refit for R2 fallback.")
                else:
                    raise RuntimeError("Estimator does not support iteration-wise refit for R2 fallback.")

                est_n.fit(X_subtrain, y_subtrain)
                temp_train_r2.append(r2_score(y_subtrain, est_n.predict(X_subtrain)))
                temp_valid_r2.append(r2_score(y_valid, est_n.predict(X_valid)))

            fallback_train_r2 = np.asarray(temp_train_r2, dtype=float)
            fallback_valid_r2 = np.asarray(temp_valid_r2, dtype=float)
        except Exception as exc:
            fallback_train_r2 = None
            fallback_valid_r2 = None
            print(f"[Warning] Failed to build fallback R2 curves: {exc}")

    loss_bundle = aggregated[loss_metric]
    rounds = loss_bundle["rounds"]
    loss_name_lower = str(loss_metric).lower()
    train_loss = np.asarray(loss_bundle["train_mean"], dtype=float)
    valid_loss = np.asarray(loss_bundle["valid_mean"], dtype=float)
    if "l2" in loss_name_lower or "mse" in loss_name_lower:
        train_loss = np.sqrt(np.maximum(train_loss, 0.0))
        valid_loss = np.sqrt(np.maximum(valid_loss, 0.0))

    # Use the shared panel geometry.
    fig = plt.figure(figsize=FIGURE2_FIGSIZE, dpi=DPI)
    ax1 = fig.add_axes([BOX_LEFT, BOX_BOTTOM, BOX_WIDTH, BOX_HEIGHT])
    ax2 = ax1.twinx()

    # Keep both y-axes visible.
    ax2.set_zorder(1)
    ax1.set_zorder(2)
    ax1.patch.set_alpha(0.0)

    handles = []
    labels = []

    if score_metric is not None:
        score_bundle = aggregated[score_metric]
        score_rounds = score_bundle["rounds"]
        train_score_vals = score_bundle["train_mean"]
        valid_score_vals = score_bundle["valid_mean"]
    elif fallback_train_r2 is not None and fallback_valid_r2 is not None and len(fallback_train_r2) > 0:
        score_rounds = rounds
        train_score_vals = fallback_train_r2
        valid_score_vals = fallback_valid_r2
    else:
        score_rounds = None
        train_score_vals = None
        valid_score_vals = None

    if score_rounds is not None:
        line1 = ax1.plot(
            score_rounds,
            train_score_vals,
            color=COLOR_SCORE_TRAIN,
            linewidth=LINEWIDTH,
            linestyle="-",
            label="Train score",
            zorder=3,
        )[0]
        line2 = ax1.plot(
            score_rounds,
            valid_score_vals,
            color=COLOR_SCORE_VALID,
            linewidth=LINEWIDTH,
            linestyle="-",
            label="Validation score",
            zorder=3,
        )[0]
        handles.extend([line1, line2])
        labels.extend([line1.get_label(), line2.get_label()])

    line3 = ax2.plot(
        rounds,
        train_loss,
        color=COLOR_LOSS_TRAIN,
        linewidth=LINEWIDTH,
        linestyle="--",
        label="Train loss",
        zorder=3,
    )[0]
    line4 = ax2.plot(
        rounds,
        valid_loss,
        color=COLOR_LOSS_VALID,
        linewidth=LINEWIDTH,
        linestyle="--",
        label="Validation loss",
        zorder=3,
    )[0]
    handles.extend([line3, line4])
    labels.extend([line3.get_label(), line4.get_label()])

    ax1.set_xlabel("Training Iteration", fontsize=AXIS_LABEL_FONTSIZE, fontweight=LABEL_FONTWEIGHT)
    ax1.set_ylabel("Score", fontsize=AXIS_LABEL_FONTSIZE, fontweight=LABEL_FONTWEIGHT)
    ax2.set_ylabel("Loss", fontsize=AXIS_LABEL_FONTSIZE, fontweight=LABEL_FONTWEIGHT)

    style_axis(ax1, grid=True)
    style_axis(ax2, grid=False)

    # Put the legend at the exact center of the main plotting area.
    add_shap_center_legend(ax1, handles=handles, labels=labels, ncol=1)

    save_figure(fig, output_dir, f"Training_Score_Loss_Curves_FixedSplit_{display_model_name}")

def generate_all_diagnostic_plots(
    best_model,
    X_train,
    y_train,
    output_dir="./Plots",
    model_name="LGBM",
    fixed_val_size=0.2,
    fixed_val_random_state=26,
    fixed_val_n_strata_bins=5,
):
    ensure_dir(output_dir)

    display_model_name = _display_name(
        _normalize_model_name(model_name, best_model)
    )

    print("=" * 70)
    print(f"Generating training diagnostics for {display_model_name} ...")
    print("=" * 70)

    plot_training_score_and_loss_curves(
        best_model=best_model,
        X_train=X_train,
        y_train=y_train,
        output_dir=output_dir,
        model_name=display_model_name,
        val_size=fixed_val_size,
        random_state=fixed_val_random_state,
        n_strata_bins=fixed_val_n_strata_bins,
    )

    print("Training diagnostics finished.")


# -*- coding: utf-8 -*-

# ============================================================================
# Module: Imports and dependencies
# ============================================================================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm

try:
    import shap
except Exception:
    shap = None
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
except Exception:
    sm_lowess = None


# ============================================================================
# Module: Global constants and style settings
# ============================================================================
FIG_DPI = 600
GLOBAL_FONT_FAMILY = "Times New Roman"
SAVE_FORMATS = ("png", "pdf", "svg")
BLUE = "#406DA8"
RED = "#BC5D59"
MID_GREY = "#6B7280"
DARK_GREY = "#374151"
RDBU_CMAP = LinearSegmentedColormap.from_list(
    "academic_red_blue",
    [BLUE, "#F9EFEE", RED],
    N=256,
)

# Shared figure-panel geometry.
BOX_LEFT = 0.25
BOX_BOTTOM = 0.22
BOX_WIDTH = 0.58
BOX_HEIGHT = 0.62

# Shared frame for Figure 1 prediction plots
FIG1_FRAME_LEFT = 0.115
FIG1_FRAME_BOTTOM = 0.08
FIG1_FRAME_WIDTH = 0.845
FIG1_FRAME_HEIGHT = 0.90

# Line + residual split inside the same frame
FIG1_RESIDUAL_HEIGHT = 0.18
FIG1_LINE_RESIDUAL_GAP = 0.015

def format_feature_label(name: str) -> str:
    mapping = {
        "CaO": r"CaO",
        "MgO": r"MgO",
        "SiO2": r"SiO$_2$",
        "MnO": r"MnO",
        "Fe2O3": r"Fe$_2$O$_3$",
        "CO2 concentration": r"CO$_2$ concentration",
        "CO2 partial pressure": r"CO$_2$ partial pressure",
        "Temperature": r"Temperature",
        "Time": r"Time",
        "Particle size": r"Particle size",
        "W/S": r"W/S",
    }
    return mapping.get(str(name), str(name))


plt.rcParams["font.family"] = GLOBAL_FONT_FAMILY
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = FIG_DPI
plt.rcParams["savefig.dpi"] = FIG_DPI

MODEL_DISPLAY_NAMES = {
    "Dummy": "Dummy",
    "Linear": "Linear",
    "Ridge": "Ridge",
    "Lasso": "Lasso",
    "DT": "DT",
    "RF": "RF",
    "GB": "GB",
    "SVR": "SVR",
    "KNN": "KNN",
    "MLP": "MLP",
    "XGB": "XGB",
    "LGBM": "LGBM",
}
FIG1_MAX_MODELS = 12


# ============================================================================
# Module: File, dataframe, and style utilities
# ============================================================================
def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, save_stem: Union[str, Path], formats: Sequence[str] = SAVE_FORMATS) -> None:
    save_stem = Path(save_stem)
    ensure_dir(save_stem.parent)
    for fmt in formats:
        fig.savefig(save_stem.with_suffix(f".{fmt}"), dpi=FIG_DPI)
    plt.close(fig)


def to_dataframe(X: Union[pd.DataFrame, np.ndarray], feature_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy().reset_index(drop=True)
    X = np.asarray(X)
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=list(feature_names))


def apply_academic_style(ax: plt.Axes, grid: bool = True) -> None:
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, color="#7A7A7A")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
PANEL_GRID_FIGSIZE = (18.5, 16.4)
PANEL_GRID_ADJUST = dict(
    left=0.075,
    right=0.965,
    bottom=0.070,
    top=0.960,
    wspace=0.28,
    hspace=0.34,
)
PANEL_2D_FIGSIZE = (18.5, 16.4)
PANEL_2D_ADJUST = dict(
    left=0.075,
    right=0.950,
    bottom=0.070,
    top=0.960,
    wspace=0.34,
    hspace=0.34,
)
PANEL_AXIS_LABEL_FONTSIZE = 22
PANEL_TICK_FONTSIZE = 20
PANEL_LEGEND_FONTSIZE = 20
PANEL_MARKER_LABEL_FONTSIZE = 18
PANEL_LABEL_FONTSIZE = 22
PANEL_LABEL_X = -0.13
PANEL_LABEL_Y = 1.06
PANEL_COLORBAR_TICK_FONTSIZE = 20


def add_panel_label(
        ax: plt.Axes,
        label: str,
        x: float = PANEL_LABEL_X,
        y: float = PANEL_LABEL_Y,
        fontsize: int = PANEL_LABEL_FONTSIZE,
) -> None:
    """Add a consistent subplot label, such as (a), (b), or (c)."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        family=GLOBAL_FONT_FAMILY,
        clip_on=False,
        zorder=50,
    )


def add_panel_labels(axes, n_panels: int) -> None:
    """Add sequential labels to the requested panels."""
    for index, ax in enumerate(list(axes)[:n_panels]):
        add_panel_label(ax, f"({chr(97 + index)})")

def apply_panel_axis_format(
        ax: plt.Axes,
        xlabel: str,
        ylabel: str,
        grid: bool = True,
        square: bool = False,
) -> None:
    """Apply the shared panel-axis format used by Figures 3, 10, 11, 12, and 13."""
    ax.set_xlabel(
        xlabel,
        fontsize=PANEL_AXIS_LABEL_FONTSIZE,
        fontweight="bold",
        family=GLOBAL_FONT_FAMILY,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=PANEL_AXIS_LABEL_FONTSIZE,
        fontweight="bold",
        family=GLOBAL_FONT_FAMILY,
    )
    ax.tick_params(axis="both", labelsize=PANEL_TICK_FONTSIZE, width=1.0, length=5.0)
    apply_academic_style(ax, grid=grid)

    if square:
        ax.set_box_aspect(1)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_family(GLOBAL_FONT_FAMILY)


def style_panel_legend(legend) -> None:
    """Apply the shared legend text style."""
    for txt in legend.get_texts():
        txt.set_fontweight("normal")
        txt.set_family(GLOBAL_FONT_FAMILY)


def style_panel_colorbar(cbar) -> None:
    """Apply the shared colorbar tick and border style."""
    cbar.ax.tick_params(
        labelsize=PANEL_COLORBAR_TICK_FONTSIZE,
        width=1.0,
        length=5.0,
    )
    cbar.outline.set_linewidth(1.5)

    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_family(GLOBAL_FONT_FAMILY)


def apply_prediction_axes_style(
        ax: plt.Axes,
        grid: bool = True,
        tick_labelsize: int = 16,
        tick_bold: bool = True,
) -> None:
    apply_academic_style(ax, grid=grid)
    ax.tick_params(axis="both", labelsize=tick_labelsize, width=1.0, length=5)

    if tick_bold:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")


# ============================================================================
# Module: Numeric helper functions
# ============================================================================
def red_blue_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if np.isclose(vmin, vmax):
        return np.full_like(values, 0.5, dtype=float)
    return (values - vmin) / (vmax - vmin)


def lowess_smooth(x: np.ndarray, y: np.ndarray, frac: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    if len(x_sorted) < 5:
        return x_sorted, y_sorted
    if sm_lowess is not None:
        smoothed = sm_lowess(endog=y_sorted, exog=x_sorted, frac=frac, return_sorted=True)
        return smoothed[:, 0], smoothed[:, 1]
    window = max(5, int(np.ceil(frac * len(x_sorted))))
    if window % 2 == 0:
        window += 1
    y_roll = pd.Series(y_sorted).rolling(window=window, center=True, min_periods=max(3, window // 4)).mean()
    y_fill = y_roll.to_numpy()
    mask = np.isnan(y_fill)
    if mask.any():
        y_fill[mask] = np.interp(x_sorted[mask], x_sorted[~mask], y_fill[~mask])
    return x_sorted, y_fill


def normalize_shap_values(shap_values: Any) -> np.ndarray:
    if hasattr(shap_values, "values"):
        arr = np.asarray(shap_values.values)
    else:
        arr = np.asarray(shap_values)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(float)


def normalize_shap_interaction_values(inter_values: Any) -> np.ndarray:
    arr = np.asarray(inter_values)
    if arr.ndim == 4:
        arr = arr[..., 0]
    return arr.astype(float)


def unwrap_estimator(model: Any) -> Tuple[Any, Optional[Any], bool]:
    if isinstance(model, Pipeline):
        final_estimator = model.steps[-1][1]
        return final_estimator, model, True
    return model, None, False


def extract_native_estimator(model: Any):
    final_estimator, _, _ = unwrap_estimator(model)
    if hasattr(final_estimator, "model_"):
        final_estimator = final_estimator.model_
    return final_estimator


def sample_dataframe(X: pd.DataFrame, max_samples: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    if max_samples is None or len(X) <= max_samples:
        return X.copy().reset_index(drop=True)
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    idx = np.sort(idx)
    return X.iloc[idx].reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def quantile_edges(values: np.ndarray, bins: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    q = np.linspace(0, 1, bins + 1)
    edges = np.quantile(values, q)
    edges = np.unique(edges)
    if len(edges) < 3:
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        edges = np.linspace(vmin, vmax, 3)
    return edges


# ============================================================================
# Module: SHAP ranking and cache utilities
# ============================================================================
@dataclass
class ShapRankingBundle:
    feature_names: List[str]
    main_effect_df: pd.DataFrame
    interaction_pair_df: pd.DataFrame
    interaction_feature_df: pd.DataFrame
    top6_main_features: List[str]
    top6_interaction_pairs: List[Tuple[str, str]]
    top6_interaction_features: List[str]
    strongest_main_feature: str
    top9_partners_for_strongest_main: List[str]


def compute_shap_rankings(
        shap_values: Any,
        shap_interaction_values: Any,
        feature_names: Sequence[str],
        top_main_n: int = 6,
        top_pair_n: int = 6,
        top_inter_feature_n: int = 6,
        top_partner_n: int = 9,
) -> ShapRankingBundle:
    shap_arr = shap_values
    inter_arr = shap_interaction_values
    feature_names = list(feature_names)

    mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)
    main_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    mean_abs_inter = np.mean(np.abs(inter_arr), axis=0)
    pair_rows = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            pair_rows.append({
                "feature_a": feature_names[i],
                "feature_b": feature_names[j],
                "mean_abs_interaction": float(mean_abs_inter[i, j]),
            })
    pair_df = pd.DataFrame(pair_rows).sort_values("mean_abs_interaction", ascending=False).reset_index(drop=True)

    interaction_feature_strength = []
    for i, feat in enumerate(feature_names):
        score = float(np.sum(mean_abs_inter[i, :]) - mean_abs_inter[i, i])
        interaction_feature_strength.append({"feature": feat, "interaction_strength_sum": score})
    inter_feature_df = (
        pd.DataFrame(interaction_feature_strength)
        .sort_values("interaction_strength_sum", ascending=False)
        .reset_index(drop=True)
    )

    strongest_main_feature = main_df.iloc[0]["feature"]
    partner_df = pair_df[
        (pair_df["feature_a"] == strongest_main_feature) |
        (pair_df["feature_b"] == strongest_main_feature)
        ].copy()
    partner_df["partner"] = np.where(
        partner_df["feature_a"] == strongest_main_feature,
        partner_df["feature_b"],
        partner_df["feature_a"],
    )

    return ShapRankingBundle(
        feature_names=feature_names,
        main_effect_df=main_df,
        interaction_pair_df=pair_df,
        interaction_feature_df=inter_feature_df,
        top6_main_features=main_df["feature"].head(top_main_n).tolist(),
        top6_interaction_pairs=list(
            pair_df.loc[: top_pair_n - 1, ["feature_a", "feature_b"]].itertuples(index=False, name=None)
        ),
        top6_interaction_features=inter_feature_df["feature"].head(top_inter_feature_n).tolist(),
        strongest_main_feature=strongest_main_feature,
        top9_partners_for_strongest_main=partner_df["partner"].head(top_partner_n).tolist(),
    )


def save_ranking_tables(bundle: ShapRankingBundle, save_dir: Union[str, Path]) -> None:
    save_dir = Path(save_dir)
    ensure_dir(save_dir)
    bundle.main_effect_df.to_csv(save_dir / "ranking_main_effect.csv", index=False, encoding="utf-8-sig")
    bundle.interaction_pair_df.to_csv(save_dir / "ranking_interaction_pairs.csv", index=False, encoding="utf-8-sig")
    bundle.interaction_feature_df.to_csv(save_dir / "ranking_interaction_features.csv", index=False,
                                         encoding="utf-8-sig")

def save_plot_source_bundle(
        save_stem: Union[str, Path],
        figure_tag: str,
        tables: Dict[str, pd.DataFrame],
        cache: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save numerical source data directly into the current figure folder.

    Outputs
    -------
    <figure_tag>_<table_name>.csv
    <figure_tag>_source_data.xlsx
    <figure_tag>_plot_cache.joblib
    """
    output_dir = Path(save_stem).parent
    ensure_dir(output_dir)

    normalized_tables: Dict[str, pd.DataFrame] = {}

    for table_name, table in tables.items():
        if table is None:
            continue

        if not isinstance(table, pd.DataFrame):
            table = pd.DataFrame(table)

        table = table.copy()
        normalized_tables[table_name] = table

        safe_name = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_"
            for ch in table_name
        )

        table.to_csv(
            output_dir / f"{figure_tag}_{safe_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    excel_path = output_dir / f"{figure_tag}_source_data.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for table_name, table in normalized_tables.items():
            sheet_name = table_name[:31]
            table.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

    cache_payload = {
        "figure_tag": figure_tag,
        "tables": normalized_tables,
        "cache": cache if cache is not None else {},
    }

    joblib.dump(
        cache_payload,
        output_dir / f"{figure_tag}_plot_cache.joblib",
        compress=3,
    )

    print(f"[SAVE] {figure_tag} source data generated.")


BASELINE_MODELS_FOR_GROUP = [
    "Dummy",
    "Linear",
    "Ridge",
    "Lasso",
]

NON_BASELINE_MODELS_FOR_GROUP = [
    "DT",
    "GB",
    "KNN",
    "MLP",
    "RF",
    "SVR",
    "XGB",
    "LGBM",
]

PRED_AXIS_MIN = -20
PRED_AXIS_MAX = 600
PRED_AXIS_TICKS = np.arange(0, 600, 100)

LINE_Y_MIN = -20
LINE_Y_MAX = 600
LINE_Y_TICKS = np.arange(0, 600, 100)

# Residual y-axis for baseline models
BASELINE_RESIDUAL_Y_MIN = -250
BASELINE_RESIDUAL_Y_MAX = 250
BASELINE_RESIDUAL_Y_TICKS = np.arange(-250, 251, 250)

# Residual y-axis for non-baseline models
NON_BASELINE_RESIDUAL_Y_MIN = -150
NON_BASELINE_RESIDUAL_Y_MAX = 150
NON_BASELINE_RESIDUAL_Y_TICKS = np.arange(-150, 151, 150)

LINE_X_TICK_STEP = 20

def get_group_edge_visibility(model_name: str) -> tuple[bool, bool]:
    keep_x_ticks = {
        "DT", "KNN", "MLP", "SVR",
        "Ridge", "Lasso",
    }

    keep_y_ticks = {
        "LGBM", "DT",
        "Dummy", "Lasso",
    }

    if model_name in BASELINE_MODELS_FOR_GROUP or model_name in NON_BASELINE_MODELS_FOR_GROUP:
        show_x = model_name in keep_x_ticks
        show_y = model_name in keep_y_ticks
        return show_x, show_y

    return True, True




def apply_group_edge_ticks(
    ax: plt.Axes,
    model_name: str,
    allow_x: bool = True,
    allow_y: bool = True,
) -> None:
    show_x, show_y = get_group_edge_visibility(model_name)

    show_x = show_x and allow_x
    show_y = show_y and allow_y

    ax.tick_params(
        axis="x",
        labelbottom=show_x,
        bottom=show_x,
    )
    ax.tick_params(
        axis="y",
        labelleft=show_y,
        left=show_y,
    )


def apply_common_prediction_axis(ax: plt.Axes) -> None:
    ax.set_xlim(PRED_AXIS_MIN, PRED_AXIS_MAX)
    ax.set_ylim(PRED_AXIS_MIN, PRED_AXIS_MAX)
    ax.set_xticks(PRED_AXIS_TICKS)
    ax.set_yticks(PRED_AXIS_TICKS)


def apply_common_line_axis(ax: plt.Axes, n_points: int) -> None:
    xmax = max(n_points - 1, 1)

    xticks = np.arange(0, xmax + 1, LINE_X_TICK_STEP)

    pad = max(5, 0.03 * xmax)

    ax.set_xlim(-pad, xmax + pad)
    ax.set_xticks(xticks)


def apply_common_residual_axis(ax: plt.Axes, model_name: str) -> None:
    if model_name in BASELINE_MODELS_FOR_GROUP:
        ax.set_ylim(BASELINE_RESIDUAL_Y_MIN, BASELINE_RESIDUAL_Y_MAX)
        ax.set_yticks(BASELINE_RESIDUAL_Y_TICKS)

    elif model_name in NON_BASELINE_MODELS_FOR_GROUP:
        ax.set_ylim(NON_BASELINE_RESIDUAL_Y_MIN, NON_BASELINE_RESIDUAL_Y_MAX)
        ax.set_yticks(NON_BASELINE_RESIDUAL_Y_TICKS)

    else:
        ax.set_ylim(BASELINE_RESIDUAL_Y_MIN, BASELINE_RESIDUAL_Y_MAX)
        ax.set_yticks(BASELINE_RESIDUAL_Y_TICKS)


# ============================================================================
# Figure 1: Prediction validation plot
# ============================================================================
def plot_prediction_validation_figure(y_true_train, y_pred_train, y_true_test, y_pred_test, model_name: str,
                                      save_stem: str) -> None:
    y_true_train = np.asarray(y_true_train, dtype=float).ravel()
    y_pred_train = np.asarray(y_pred_train, dtype=float).ravel()
    y_true_test = np.asarray(y_true_test, dtype=float).ravel()
    y_pred_test = np.asarray(y_pred_test, dtype=float).ravel()

    train_metrics = compute_metrics(y_true_train, y_pred_train)
    test_metrics = compute_metrics(y_true_test, y_pred_test)
    x_all = np.concatenate([y_true_train, y_true_test])
    y_all = np.concatenate([y_pred_train, y_pred_test])

    lower = PRED_AXIS_MIN
    upper = PRED_AXIS_MAX

    line_x = np.linspace(lower, upper, 300)
    fit_y = np.polyval(np.polyfit(x_all, y_all, deg=1), line_x)

    fig = plt.figure(figsize=(5.95, 5.27), dpi=FIG_DPI)

    # Keep only the main panel and enlarge its drawing area.
    ax_main = fig.add_axes([
        FIG1_FRAME_LEFT,
        FIG1_FRAME_BOTTOM,
        FIG1_FRAME_WIDTH,
        FIG1_FRAME_HEIGHT,
    ])

    ax_main.scatter(
        y_true_train, y_pred_train,
        s=42, color=BLUE, alpha=0.45,
        edgecolors="none", label="Train data", zorder=3
    )
    ax_main.scatter(
        y_true_test, y_pred_test,
        s=52, color=RED, alpha=0.75,
        edgecolors="none", label="Test data", zorder=4
    )

    ax_main.plot(
        line_x, line_x,
        linestyle="--", color="black", linewidth=1.5,
        label="y = x", zorder=2
    )
    ax_main.plot(
        line_x, fit_y,
        linestyle="-", color="black", linewidth=1.5,
        label="Fitted line", zorder=2
    )

    apply_common_prediction_axis(ax_main)



    ax_main.set_xlabel("")
    ax_main.set_ylabel("")

    ax_main.text(
        0.5, 0.96, MODEL_DISPLAY_NAMES.get(model_name, model_name),
        transform=ax_main.transAxes, ha="center", va="top",
        fontsize=22, fontweight="bold"
    )

    metrics_text = (
        r"Train $\it{R^2}$ = " + f"{train_metrics['R2']:.3f}" + "\n"
        r"Train RMSE = " + f"{train_metrics['RMSE']:.3f}" + "\n"
        r"Test $\it{R^2}$ = " + f"{test_metrics['R2']:.3f}" + "\n"
        r"Test RMSE = " + f"{test_metrics['RMSE']:.3f}"
    )

    ax_main.text(
        0.05, 0.90, metrics_text,
        transform=ax_main.transAxes,
        ha="left", va="top",
        fontsize=16, color=RED, fontweight="bold"
    )

    apply_prediction_axes_style(ax_main, tick_labelsize=20)
    apply_group_edge_ticks(ax_main, model_name, allow_x=True, allow_y=True)

    legend = ax_main.legend(
        loc="lower right",
        frameon=False,
        fontsize=16,
        markerscale=1.35,
        labelspacing=0.85,
        handletextpad=0.7,
        borderaxespad=0.6,
    )

    for txt in legend.get_texts():
        txt.set_fontsize(16)
        txt.set_fontweight("normal")
        txt.set_family(GLOBAL_FONT_FAMILY)

    save_figure(fig, save_stem)


# ============================================================================
# Figure 1: Prediction line and residual plot
# ============================================================================
def plot_prediction_line_figure(y_true_train, y_pred_train, y_true_test, y_pred_test, model_name: str,
                                save_stem: str) -> None:
    y_true_test = np.asarray(y_true_test, dtype=float).ravel()
    y_pred_test = np.asarray(y_pred_test, dtype=float).ravel()

    idx_test = np.arange(1, len(y_true_test) + 1)
    res_test = y_pred_test - y_true_test
    test_metrics = compute_metrics(y_true_test, y_pred_test)

    fig = plt.figure(figsize=(5.95, 5.27), dpi=FIG_DPI)

    # Keep the combined main and residual panels within BOX_HEIGHT.
    line_height = FIG1_FRAME_HEIGHT - FIG1_RESIDUAL_HEIGHT - FIG1_LINE_RESIDUAL_GAP

    ax = fig.add_axes([
        FIG1_FRAME_LEFT,
        FIG1_FRAME_BOTTOM + FIG1_RESIDUAL_HEIGHT + FIG1_LINE_RESIDUAL_GAP,
        FIG1_FRAME_WIDTH,
        line_height,
    ])

    ax_res = fig.add_axes([
        FIG1_FRAME_LEFT,
        FIG1_FRAME_BOTTOM,
        FIG1_FRAME_WIDTH,
        FIG1_RESIDUAL_HEIGHT,
    ], sharex=ax)

    ax.set_ylim(LINE_Y_MIN, LINE_Y_MAX)
    ax.set_yticks(LINE_Y_TICKS)
    apply_common_line_axis(ax, len(y_true_test))

    ax.plot(idx_test, y_true_test, color=RED, linewidth=1.5, marker="o", markersize=6, label="Actual", zorder=3)
    ax.plot(idx_test, y_pred_test, color=BLUE, linewidth=1.5, linestyle="--", marker="^", markersize=6,
            label="Predicted", zorder=3)

    ax.set_ylabel("")

    apply_prediction_axes_style(ax, tick_labelsize=20, tick_bold=True)
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    apply_group_edge_ticks(ax, model_name, allow_x=False, allow_y=True)

    legend = ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=20, borderpad=0.3, columnspacing=0.8)
    legend.get_frame().set_linewidth(1.0)

    # Keep the legend text for Actual and Predicted at normal weight.
    for txt in legend.get_texts():
        txt.set_fontweight("normal")

    ax.text(0.98, 0.92, MODEL_DISPLAY_NAMES.get(model_name, model_name),
            transform=ax.transAxes, ha="right", va="center", fontsize=22, fontweight="bold")

    # Residual panel.
    ax_res.scatter(idx_test, res_test, s=40, color=RED, alpha=0.75, edgecolors="none", zorder=4)
    ax_res.axhline(0.0, color="black", linewidth=1.2)

    ax_res.set_xlabel("")
    ax_res.set_ylabel("")
    apply_common_residual_axis(ax_res, model_name)
    apply_common_line_axis(ax_res, len(y_true_test))

    ax_res.text(0.98, 0.88, r"$\it{MAE}$ = " + f"{test_metrics['MAE']:.2f}",
                transform=ax_res.transAxes, ha="right", va="top", fontsize=20, color="black", fontweight="bold",  zorder=10)

    apply_prediction_axes_style(ax_res, tick_labelsize=20, tick_bold=True)
    apply_group_edge_ticks(ax_res, model_name, allow_x=True, allow_y=True)

    save_figure(fig, save_stem)


def load_predictions_file(pred_path: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_df = pd.read_csv(pred_path)

    required_cols = {"dataset", "y_true", "y_pred"}
    missing_cols = required_cols - set(pred_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in {pred_path}: {sorted(missing_cols)}")

    train_df = pred_df[pred_df["dataset"].astype(str).str.lower() == "train"].copy()
    test_df = pred_df[pred_df["dataset"].astype(str).str.lower() == "test"].copy()

    if train_df.empty:
        raise ValueError(f"No train rows found in {pred_path}")
    if test_df.empty:
        raise ValueError(f"No test rows found in {pred_path}")

    return train_df, test_df


def build_shap_cache(model: Any, X_train_df: pd.DataFrame, X_test_df: pd.DataFrame, random_state: int = 26) -> Dict[
    str, Any]:
    if shap is None:
        raise ImportError("shap is not installed.")

    native_model = extract_native_estimator(model)
    tree_explainer = shap.TreeExplainer(native_model)

    X_explain = X_train_df
    X_eval_df = sample_dataframe(X_explain, max_samples=min(550, len(X_explain)), random_state=random_state)
    X_inter_df = sample_dataframe(X_explain, max_samples=min(320, len(X_explain)), random_state=random_state)

    shap_values_eval = tree_explainer.shap_values(X_eval_df)
    if isinstance(shap_values_eval, list):
        shap_values_eval = shap_values_eval[0]
    shap_values_eval = normalize_shap_values(shap_values_eval)

    shap_interaction_values = tree_explainer.shap_interaction_values(X_inter_df)
    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[0]
    shap_interaction_values = normalize_shap_interaction_values(shap_interaction_values)

    expected_value = tree_explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.asarray(expected_value).ravel()[0])
    else:
        expected_value = float(expected_value)

    return {
        "X_eval_df": X_eval_df,
        "X_inter_df": X_inter_df,
        "shap_values_eval": shap_values_eval,
        "shap_interaction_values": shap_interaction_values,
        "expected_value": expected_value,
    }


def get_cache_path(save_dir: Union[str, Path], model_name: str, cache_tag: str = "train") -> Path:
    return Path(save_dir) / f"{model_name}_figure_cache_{cache_tag}.joblib"


def build_and_save_shap_cache(
        model,
        X_train_df,
        X_test_df,
        model_name,
        save_dir,
        random_state=26,
        cache_tag="train",
):
    ensure_dir(save_dir)
    cache_path = get_cache_path(save_dir, model_name, cache_tag=cache_tag)

    if cache_path.exists():
        return joblib.load(cache_path)

    cache = build_shap_cache(
        model=model,
        X_train_df=X_train_df,
        X_test_df=X_test_df,
        random_state=random_state,
    )
    joblib.dump(cache, cache_path)
    return cache


def prepare_interpretation_context(results_dir: str, model_name: str):
    results_root = Path(results_dir)
    model_dir = results_root / model_name
    interpret_dir = model_dir / "interpretability"
    ensure_dir(interpret_dir)

    model_path = model_dir / f"model_{model_name}.pkl"
    X_train_path = results_root / "X_train_raw.csv"
    X_test_path = results_root / "X_test_raw.csv"
    y_train_path = results_root / "y_train.csv"
    y_test_path = results_root / "y_test.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not X_train_path.exists() or not X_test_path.exists():
        raise FileNotFoundError("X_train_raw.csv or X_test_raw.csv is missing in Results root.")
    if not y_train_path.exists() or not y_test_path.exists():
        raise FileNotFoundError("y_train.csv or y_test.csv is missing in Results root.")

    model = joblib.load(model_path)
    X_train_df = pd.read_csv(X_train_path)
    X_test_df = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0]
    y_test = pd.read_csv(y_test_path).iloc[:, 0]

    cache = build_and_save_shap_cache(
        model=model,
        X_train_df=X_train_df,
        X_test_df=X_test_df,
        model_name=model_name,
        save_dir=interpret_dir,
        random_state=26,
        cache_tag="train",
    )
    ranking_bundle = compute_shap_rankings(
        shap_values=cache["shap_values_eval"],
        shap_interaction_values=cache["shap_interaction_values"],
        feature_names=list(X_eval_or_test_columns(cache["X_eval_df"], X_test_df)),
        top_main_n=6,
        top_pair_n=6,
        top_inter_feature_n=6,
        top_partner_n=9,
    )
    save_ranking_tables(ranking_bundle, interpret_dir / "00_rankings")

    return {
        "results_root": results_root,
        "model_dir": model_dir,
        "interpret_dir": interpret_dir,
        "model": model,
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "y_train": y_train,
        "y_test": y_test,
        "X_eval_df": cache["X_eval_df"],
        "X_inter_df": cache["X_inter_df"],
        "shap_values_eval": cache["shap_values_eval"],
        "interaction_values": cache["shap_interaction_values"],
        "expected_value": cache["expected_value"],
        "ranking_bundle": ranking_bundle,
    }


def X_eval_or_test_columns(X_eval_df: pd.DataFrame, X_test_df: pd.DataFrame) -> Sequence[str]:
    if hasattr(X_eval_df, "columns"):
        return list(X_eval_df.columns)
    return list(X_test_df.columns)


# ============================================================================
# Figure 2: SHAP global contribution
# ============================================================================
def plot_figure2_shap_global_contribution(shap_values: Any, X: pd.DataFrame, save_stem: Union[str, Path]) -> None:
    shap_arr = shap_values

    features = list(X.columns)
    mean_abs = np.mean(np.abs(shap_arr), axis=0)
    order = np.argsort(mean_abs)[::-1]

    features_ranked = [features[i] for i in order]
    features_ranked_fmt = [format_feature_label(f) for f in features_ranked]
    mean_abs_ranked = mean_abs[order]
    pct_ranked = mean_abs_ranked / mean_abs_ranked.sum() * 100.0

    # ================================================================
    # Figure 2 source data
    # ================================================================
    importance_source = pd.DataFrame({
        "Rank": np.arange(1, len(features_ranked) + 1),
        "Feature": features_ranked,
        "Mean_absolute_SHAP": mean_abs_ranked,
        "Importance_percentage": pct_ranked,
        "Cumulative_importance_percentage":
            np.cumsum(pct_ranked),
    })

    sample_source = pd.concat(
        [
            pd.DataFrame({
                "Sample_ID": np.arange(1, len(X) + 1)
            }),
            X.reset_index(drop=True).add_prefix(
                "Feature_value__"
            ),
            pd.DataFrame(
                shap_arr,
                columns=[
                    f"SHAP_value__{feature}"
                    for feature in features
                ]
            ),
        ],
        axis=1,
    )

    save_plot_source_bundle(
        save_stem=save_stem,
        figure_tag="Figure2",
        tables={
            "SHAP_importance": importance_source,
            "SHAP_values_by_sample": sample_source,
        },
        cache={
            "feature_names": features,
            "feature_order": order,
            "X": X.copy(),
            "shap_values": np.asarray(shap_arr),
            "mean_absolute_SHAP": mean_abs,
            "importance_percentage":
                mean_abs / mean_abs.sum() * 100.0,
            "scatter_random_state": 26,
        },
    )
    # ================================================================
    # Export SHAP feature-importance percentage table
    # ================================================================
    importance_table = pd.DataFrame({
        "Rank": np.arange(1, len(features_ranked) + 1),
        "Feature": features_ranked,
        "Mean absolute SHAP value": mean_abs_ranked,
        "Importance (%)": pct_ranked,
        "Cumulative importance (%)": np.cumsum(pct_ranked),
    })

    # Set numerical precision for the exported table
    importance_table["Mean absolute SHAP value"] = (
        importance_table["Mean absolute SHAP value"].round(4)
    )
    importance_table["Importance (%)"] = (
        importance_table["Importance (%)"].round(2)
    )
    importance_table["Cumulative importance (%)"] = (
        importance_table["Cumulative importance (%)"].round(2)
    )

    # Export the table beside the figure.
    table_dir = Path(save_stem).parent
    ensure_dir(table_dir)

    print(
        "[SAVE] SHAP importance percentage table:",
        table_dir / "Figure2_SHAP_Feature_Importance_Percentage.xlsx"
    )
    shap_sorted = shap_arr[:, order]
    shap_min = float(np.nanmin(shap_sorted))
    shap_max = float(np.nanmax(shap_sorted))
    shap_pad = 0.06 * (shap_max - shap_min + 1e-12)
    shap_xlim = (shap_min - shap_pad, shap_max + shap_pad)

    top_max = float(np.nanmax(mean_abs_ranked)) * 1.14

    fig = plt.figure(figsize=(8.5, 6.5), dpi=FIG_DPI)

    # Use the shared global axes box to keep the geometry consistent with Figures 1 and 4.
    ax = fig.add_axes([BOX_LEFT, BOX_BOTTOM, BOX_WIDTH, BOX_HEIGHT])
    ax_top = ax.twiny()

    # Keep the colorbar close to the main panel.
    cax = fig.add_axes([BOX_LEFT + BOX_WIDTH, BOX_BOTTOM, 0.02, BOX_HEIGHT])

    y = np.arange(len(features_ranked))[::-1]

    ax_top.set_zorder(0)
    ax.set_zorder(2)
    ax.patch.set_alpha(0.0)

    ax_top.barh(y, mean_abs_ranked, height=0.58, color="#b6d4e9", edgecolor="none", alpha=0.65, zorder=0)

    pct_x = shap_xlim[0] + 0.01 * (shap_xlim[1] - shap_xlim[0])

    rng = np.random.default_rng(26)
    for row, feat_idx in enumerate(order):
        y0 = len(order) - 1 - row
        sv = shap_arr[:, feat_idx]
        fv = X.iloc[:, feat_idx].to_numpy(dtype=float)
        jitter = rng.normal(0.0, 0.10, size=len(sv))

        ax.scatter(sv, np.full_like(sv, y0, dtype=float) + jitter, c=red_blue_norm(fv),
                   cmap=RDBU_CMAP, s=22, alpha=1.0, edgecolors="none", zorder=3)

    ax.set_yticks(y)
    # Use a smaller y tick label size to avoid crowding long feature names.
    ax.set_yticklabels(features_ranked_fmt, fontsize=16, fontweight="bold")

    ax.set_xlim(*shap_xlim)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=18, fontweight="bold")

    ax_top.set_xlim(0, top_max)
    ax_top.set_xlabel("Mean absolute SHAP value", fontsize=18, fontweight="bold")

    ax.axvline(0.0, color=MID_GREY, linestyle="--", linewidth=1.2, zorder=2)
    apply_academic_style(ax, grid=True)
    ax_top.grid(False)

    ax.spines["right"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    ax.set_ylim(-0.8, len(features_ranked) - 0.2)
    ax_top.set_ylim(ax.get_ylim())

    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=RDBU_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["", ""])
    cbar.ax.tick_params(length=0)

    cbar.ax.text(1.48, 0.50, "Feature value", transform=cbar.ax.transAxes, rotation=90, va="center", ha="left",
                 fontsize=18, fontweight="bold")
    cbar.ax.text(1.48, 1.00, "High", transform=cbar.ax.transAxes, va="center", ha="left", fontsize=16,
                 fontweight="bold")
    cbar.ax.text(1.48, 0.00, "Low", transform=cbar.ax.transAxes, va="center", ha="left", fontsize=16, fontweight="bold")

    ax.tick_params(axis="x", labelsize=16, width=1.0, length=5)
    ax.tick_params(axis="y", labelsize=16, width=1.0, length=5)
    ax_top.tick_params(axis="x", labelsize=16, width=1.0, length=5)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels() + ax_top.get_xticklabels():
        lbl.set_fontweight("bold")

    save_figure(fig, save_stem)


# ============================================================================
# Module: Shared dependence-plot helpers
# ============================================================================
def get_feature_axis_label(feature: str) -> str:
    try:
        from data_statistics import FEATURE_CONFIG
    except Exception:
        FEATURE_CONFIG = []

    for item in FEATURE_CONFIG:
        if item["key"] == feature:
            display = item.get("display", feature)
            unit = item.get("unit", "")
            if unit and unit != "-":
                return f"{display} ({unit})"
            return display

    return format_feature_label(feature)


def find_all_zero_crossings(x_smooth: np.ndarray, y_smooth: np.ndarray) -> List[float]:
    x_smooth = np.asarray(x_smooth, dtype=float)
    y_smooth = np.asarray(y_smooth, dtype=float)

    valid = np.isfinite(x_smooth) & np.isfinite(y_smooth)
    x_smooth = x_smooth[valid]
    y_smooth = y_smooth[valid]

    crossings = []
    if len(x_smooth) < 2:
        return crossings

    for i in range(len(x_smooth) - 1):
        x1, x2 = x_smooth[i], x_smooth[i + 1]
        y1, y2 = y_smooth[i], y_smooth[i + 1]

        if np.isclose(y1, 0.0):
            crossings.append(float(x1))

        if y1 * y2 < 0:
            xc = x1 - y1 * (x2 - x1) / (y2 - y1)
            crossings.append(float(xc))

    if np.isclose(y_smooth[-1], 0.0):
        crossings.append(float(x_smooth[-1]))

    out = []
    for v in crossings:
        if not out or abs(v - out[-1]) > 1e-8:
            out.append(v)
    return out


def shade_promoting_inhibiting_regions(
        ax: plt.Axes,
        x_smooth: np.ndarray,
        y_smooth: np.ndarray,
        xmin_plot: float,
        xmax_plot: float,
        ymin_plot: float,
        ymax_plot: float,
        pos_color: str = RED,
        neg_color: str = BLUE,
        alpha: float = 0.08,
) -> List[float]:
    crossings = find_all_zero_crossings(x_smooth, y_smooth)

    bounds = [float(xmin_plot)] + crossings + [float(xmax_plot)]

    x_left = float(xmin_plot)
    x_right = float(xmax_plot)
    x_span = x_right - x_left + 1e-12

    for left, right in zip(bounds[:-1], bounds[1:]):
        if right <= left:
            continue

        x_mid = 0.5 * (left + right)

        y_mid = float(np.interp(x_mid, x_smooth, y_smooth))

        xmin_frac = (left - x_left) / x_span
        xmax_frac = (right - x_left) / x_span

        xmin_frac = np.clip(xmin_frac, 0, 1)
        xmax_frac = np.clip(xmax_frac, 0, 1)

        if y_mid >= 0:
            ax.axhspan(
                0, ymax_plot,
                xmin=xmin_frac, xmax=xmax_frac,
                facecolor=pos_color,
                alpha=alpha,
                zorder=0
            )
        else:
            ax.axhspan(
                ymin_plot, 0,
                xmin=xmin_frac, xmax=xmax_frac,
                facecolor=neg_color,
                alpha=alpha,
                zorder=0
            )

    return crossings


def build_trend_band(x: np.ndarray, y: np.ndarray, x_smooth: np.ndarray, y_smooth: np.ndarray, frac: float = 0.20):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_smooth = np.asarray(x_smooth, dtype=float)
    y_smooth = np.asarray(y_smooth, dtype=float)

    if len(x_smooth) < 3:
        return y_smooth, y_smooth

    y_fit_on_raw = np.interp(x, x_smooth, y_smooth)
    resid = y - y_fit_on_raw

    band = np.nanstd(resid)
    if not np.isfinite(band):
        band = 0.0

    band = max(band * 0.35, 0.05 * (np.nanmax(y) - np.nanmin(y) + 1e-12))

    y_lower = y_smooth - band
    y_upper = y_smooth + band
    return y_lower, y_upper


def add_crossing_label(
        ax: plt.Axes,
        xc: float,
        text: str,
        pos_code: str,
        xmin_plot: float,
        xmax_plot: float,
        ymin_plot: float,
        ymax_plot: float,
        color: str = RED,
        fontsize: int = 16,
):
    xspan = xmax_plot - xmin_plot
    yspan = ymax_plot - ymin_plot

    dx = 0.015 * xspan
    dy = 0.055 * yspan

    if pos_code == "upper_left":
        xt = xc - dx
        yt = 0 + dy
        ha = "right"
        va = "bottom"
    elif pos_code == "upper_right":
        xt = xc + dx
        yt = 0 + dy
        ha = "left"
        va = "bottom"
    elif pos_code == "lower_left":
        xt = xc - dx
        yt = 0 - dy
        ha = "right"
        va = "top"
    elif pos_code == "lower_left_far":
        xt = xc - 0.055 * xspan
        yt = 0 - 0.130 * yspan
        ha = "right"
        va = "top"
    elif pos_code == "lower_right":
        xt = xc + dx
        yt = 0 - dy
        ha = "left"
        va = "top"
    else:
        xt = xc + dx
        yt = 0 + dy
        ha = "left"
        va = "bottom"

    ax.text(
        xt, yt, text,
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        ha=ha,
        va=va,
        zorder=5
    )


# ============================================================================
# Figure 3: SHAP dependence grid
# ============================================================================
def plot_figure3_shap_dependence_grid(
        shap_values: Any,
        X: pd.DataFrame,
        ranking_bundle: ShapRankingBundle,
        save_stem: Union[str, Path]
) -> None:
    shap_arr = shap_values
    features = ranking_bundle.main_effect_df["feature"].tolist()[:11]

    # Collect the exact numerical values used to draw and export Figure 3.
    point_frames: List[pd.DataFrame] = []
    trend_frames: List[pd.DataFrame] = []
    crossing_rows: List[Dict[str, Any]] = []

    fig, axes = plt.subplots(4, 3, figsize=PANEL_GRID_FIGSIZE, dpi=FIG_DPI)
    axes = axes.ravel()
    fig.subplots_adjust(**PANEL_GRID_ADJUST)
    add_panel_labels(axes, n_panels=len(features))

    label_position_map = {
        0: ["upper_left"],
        1: ["lower_right"],
        2: ["lower_right"],
        3: ["upper_left"],
        4: ["lower_right", "lower_left"],
        5: ["upper_right"],
        6: ["lower_right"],
        7: ["lower_right", "lower_left_far", "upper_right", "upper_right"],
        8: ["lower_right", "lower_left"],
        9: ["lower_left", "upper_left", "lower_left"],
        10: ["lower_left", "upper_left"],
    }

    for idx, feat in enumerate(features):
        ax = axes[idx]
        feat_idx = X.columns.get_loc(feat)

        x_all = X[feat].to_numpy(dtype=float)
        y_all = shap_arr[:, feat_idx]

        valid = np.isfinite(x_all) & np.isfinite(y_all)
        valid_sample_ids = (
                np.flatnonzero(valid) + 1
        )

        x = x_all[valid]
        y = y_all[valid]

        point_frames.append(
            pd.DataFrame({
                "Feature": feat,
                "Sample_ID": valid_sample_ids,
                "Feature_value": x,
                "SHAP_value": y,
            })
        )

        if len(x) < 5:
            ax.axis("off")
            continue

        ax.scatter(
            x, y,
            color=RED,
            s=22,
            alpha=0.85,
            edgecolors="none",
            zorder=3
        )

        x_smooth, y_smooth = lowess_smooth(x, y, frac=0.28)
        y_lower, y_upper = build_trend_band(x, y, x_smooth, y_smooth)

        ax.fill_between(
            x_smooth, y_lower, y_upper,
            color=BLUE,
            alpha=0.28,
            zorder=2
        )

        ax.plot(
            x_smooth, y_smooth,
            color=BLUE,
            linewidth=2.2,
            zorder=4
        )

        xmin, xmax = np.nanmin(x), np.nanmax(x)
        ymin, ymax = np.nanmin(y), np.nanmax(y)

        xpad = 0.05 * (xmax - xmin + 1e-12)
        ypad = 0.08 * (ymax - ymin + 1e-12)
        xmin_plot = xmin - xpad
        xmax_plot = xmax + xpad
        ymin_plot = ymin - ypad
        ymax_plot = ymax + ypad
        ax.set_xlim(xmin_plot, xmax_plot)
        ax.set_ylim(ymin_plot, ymax_plot)

        ax.axhline(0, color=MID_GREY, linestyle="--", linewidth=1.2, zorder=1)

        crossings = shade_promoting_inhibiting_regions(
            ax=ax,
            x_smooth=x_smooth,
            y_smooth=y_smooth,
            xmin_plot=xmin_plot,
            xmax_plot=xmax_plot,
            ymin_plot=ymin_plot,
            ymax_plot=ymax_plot,
            pos_color="#D65E5E",
            neg_color="#1f77b4",
            alpha=0.08,
        )

        trend_frames.append(
            pd.DataFrame({
                "Feature": feat,
                "Trend_index":
                    np.arange(1, len(x_smooth) + 1),
                "Feature_value": x_smooth,
                "LOWESS_SHAP": y_smooth,
                "Trend_lower": y_lower,
                "Trend_upper": y_upper,
            })
        )

        for crossing_index, crossing_value in enumerate(
                crossings,
                start=1
        ):
            crossing_rows.append({
                "Feature": feat,
                "Crossing_index": crossing_index,
                "Zero_crossing_feature_value":
                    float(crossing_value),
            })

        label_positions = label_position_map.get(idx, [])

        for j, xc in enumerate(crossings):
            ax.axvline(
                xc,
                color=RED,
                linestyle="--",
                linewidth=1.5,
                zorder=1
            )

            pos_code = label_positions[j] if j < len(label_positions) else "upper_right"

            add_crossing_label(
                ax=ax,
                xc=xc,
                text=f"{xc:.2f}",
                pos_code=pos_code,
                xmin_plot=xmin_plot,
                xmax_plot=xmax_plot,
                ymin_plot=ymin_plot,
                ymax_plot=ymax_plot,
                color=BLUE,
                fontsize=PANEL_MARKER_LABEL_FONTSIZE,
            )

        apply_panel_axis_format(
            ax,
            xlabel=get_feature_axis_label(feat),
            ylabel="SHAP value",
            grid=True,
        )

    legend_ax = axes[11]
    legend_ax.axis("off")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=RED, markersize=7, label="Samples"),
        Line2D([0], [0], color=BLUE, lw=1.5, label="Trend line"),
        Patch(facecolor=BLUE, alpha=0.28, label="Trend range"),
        Patch(facecolor="#D65E5E", alpha=0.12, label="Promoting region"),
        Patch(facecolor="#1f77b4", alpha=0.12, label="Inhibiting region"),
    ]

    legend = legend_ax.legend(
        handles=legend_handles,
        loc="center",
        fontsize=PANEL_LEGEND_FONTSIZE,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#D9D9D9",
        facecolor="white",
        labelspacing=1.0,
        handlelength=2.2,
        handletextpad=0.8,
        borderpad=1.1
    )
    style_panel_legend(legend)
    point_table = pd.concat(
        point_frames,
        ignore_index=True
    )

    trend_table = pd.concat(
        trend_frames,
        ignore_index=True
    )

    crossing_table = pd.DataFrame(
        crossing_rows,
        columns=[
            "Feature",
            "Crossing_index",
            "Zero_crossing_feature_value",
        ],
    )

    save_plot_source_bundle(
        save_stem=save_stem,
        figure_tag="Figure3",
        tables={
            "SHAP_dependence_points": point_table,
            "LOWESS_trend": trend_table,
            "Zero_crossings": crossing_table,
        },
        cache={
            "selected_features": features,
            "lowess_fraction": 0.28,
            "point_data": point_table,
            "trend_data": trend_table,
            "crossings": crossing_table,
        },
    )
    save_figure(fig, save_stem)


# ============================================================================
# Figure 4: Main and interaction effects
# ============================================================================
def plot_figure4_main_vs_interaction_bar(
        shap_values: Any,
        shap_interaction_values: Any,
        X: pd.DataFrame,
        ranking_bundle: ShapRankingBundle,
        save_stem: Union[str, Path]
) -> None:
    shap_arr = shap_values
    inter_arr = shap_interaction_values

    feature_names = ranking_bundle.main_effect_df["feature"].tolist()
    mean_abs_main = np.mean(np.abs(shap_arr), axis=0)
    mean_abs_inter = np.mean(np.abs(inter_arr), axis=0)
    inter_sum = mean_abs_inter.sum(axis=1) - np.diag(mean_abs_inter)
    order = [X.columns.get_loc(f) for f in feature_names]
    main_values = mean_abs_main[order]
    inter_values = inter_sum[order]
    total_values = main_values + inter_values

    figure4_source = pd.DataFrame({
        "Rank": np.arange(1, len(feature_names) + 1),
        "Feature": feature_names,
        "Mean_absolute_main_SHAP": main_values,
        "Sum_mean_absolute_interaction":
            inter_values,
        "Combined_magnitude": total_values,
        "Main_effect_percentage": np.divide(
            main_values,
            total_values,
            out=np.zeros_like(main_values),
            where=total_values != 0,
        ) * 100.0,
        "Interaction_effect_percentage": np.divide(
            inter_values,
            total_values,
            out=np.zeros_like(inter_values),
            where=total_values != 0,
        ) * 100.0,
    })

    save_plot_source_bundle(
        save_stem=save_stem,
        figure_tag="Figure4",
        tables={
            "Main_and_interaction_effects":
                figure4_source,
        },
        cache={
            "feature_names": feature_names,
            "main_values": main_values,
            "interaction_values": inter_values,
            "mean_absolute_interaction_matrix":
                mean_abs_inter,
        },
    )
    x = np.arange(len(feature_names))
    width = 0.38

    fig = plt.figure(figsize=(8.5, 6.5), dpi=FIG_DPI)

    ax = fig.add_axes([BOX_LEFT, BOX_BOTTOM, BOX_WIDTH, BOX_HEIGHT])

    bars1 = ax.bar(x - width / 2, main_values, width=width, color=BLUE, alpha=0.85, edgecolor="white", linewidth=1.5,
                   label="Main effect(Mean |SHAP|)", zorder=3)
    bars2 = ax.bar(x + width / 2, inter_values, width=width, color=RED, alpha=0.80, edgecolor="white", linewidth=1.5,
                   label="Interaction effect(sum over others)", zorder=3)

    xticklabels = [format_feature_label(f) for f in feature_names]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=35, ha="right", fontsize=16)

    ax.set_ylabel("Magnitude (Mean |SHAP|)", fontsize=18, fontweight="bold")

    ax.tick_params(axis="y", labelsize=16, width=1.0, length=5)
    ax.tick_params(axis="x", labelsize=16, width=1.0, length=5)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    apply_academic_style(ax, grid=True)
    ax.axhline(0, color=DARK_GREY, linewidth=1.0, zorder=10, clip_on=False)

    legend = ax.legend(loc="upper right", frameon=False, fontsize=16)
    for txt in legend.get_texts():
        txt.set_fontweight("normal")

    y_max = max(np.max(main_values), np.max(inter_values))

    ax.set_ylim(0, y_max * 1.25)

    save_figure(fig, save_stem)


# ============================================================================
# Figure 5: SHAP interaction matrix
# ============================================================================
def plot_figure5_interaction_matrix(
        shap_interaction_values: Any,
        X_inter: pd.DataFrame,
        ranking_bundle: ShapRankingBundle,
        save_stem: Union[str, Path]
) -> None:
    inter_arr = normalize_shap_interaction_values(shap_interaction_values)
    X_inter = to_dataframe(X_inter)

    top_features = ranking_bundle.top6_interaction_features[:6]
    n = len(top_features)

    idxs = [X_inter.columns.get_loc(f) for f in top_features]
    mean_abs_inter = np.mean(np.abs(inter_arr), axis=0)
    sub_mat = mean_abs_inter[np.ix_(idxs, idxs)]
    # Mean absolute interaction matrix
    interaction_matrix_df = pd.DataFrame(
        sub_mat,
        index=top_features,
        columns=top_features,
    ).reset_index().rename(
        columns={"index": "Feature"}
    )

    # Sample-level interaction values
    selected_tensor = (
        inter_arr[:, idxs, :][:, :, idxs]
    )

    interaction_long_frames = []

    for i, feature_a in enumerate(top_features):
        for j in range(i, len(top_features)):
            feature_b = top_features[j]

            interaction_long_frames.append(
                pd.DataFrame({
                    "Sample_ID":
                        np.arange(1, len(X_inter) + 1),
                    "Feature_A": feature_a,
                    "Feature_B": feature_b,
                    "Effect_type":
                        "Main_effect_diagonal"
                        if i == j
                        else "Pairwise_interaction",
                    "Feature_A_value":
                        X_inter[feature_a].to_numpy(),
                    "Feature_B_value":
                        X_inter[feature_b].to_numpy(),
                    "SHAP_interaction_value":
                        selected_tensor[:, i, j],
                })
            )

    interaction_long_df = pd.concat(
        interaction_long_frames,
        ignore_index=True,
    )

    save_plot_source_bundle(
        save_stem=save_stem,
        figure_tag="Figure5",
        tables={
            "Mean_absolute_interaction_matrix":
                interaction_matrix_df,
            "Sample_interaction_values":
                interaction_long_df,
        },
        cache={
            "selected_features": top_features,
            "selected_feature_indices": idxs,
            "X_selected":
                X_inter[top_features].copy(),
            "interaction_tensor":
                selected_tensor,
            "mean_absolute_interaction_matrix":
                sub_mat,
            "jitter_random_state": 26,
        },
    )
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    off_diag_vals = sub_mat[mask]

    heat_vmax = float(np.nanmax(off_diag_vals))
    heat_norm = Normalize(vmin=-heat_vmax, vmax=heat_vmax)

    fig, axes = plt.subplots(n, n, figsize=(15.8, 15.2), dpi=FIG_DPI)
    plt.subplots_adjust(left=0.07, right=0.86, bottom=0.10, top=0.88, wspace=0.08, hspace=0.08)

    axis_label_fontsize = 22
    tick_fontsize = 20
    marker_label_fontsize = 18

    rng = np.random.default_rng(26)

    def _jitter_from_x(xvals: np.ndarray, scale: float = 0.28) -> np.ndarray:
        xvals = np.asarray(xvals, dtype=float)
        if len(xvals) < 5: return rng.normal(0.0, 0.06, size=len(xvals))
        try:
            kde = gaussian_kde(xvals)
            dens = kde(xvals)
            dens = dens / (np.nanmax(dens) + 1e-12)
            width = 0.04 + scale * dens
            return rng.uniform(-1.0, 1.0, size=len(xvals)) * width
        except Exception:
            return rng.normal(0.0, 0.12, size=len(xvals))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ii, jj = idxs[i], idxs[j]

            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color(DARK_GREY)

            ax.set_facecolor("white")

            if i == j:
                x = inter_arr[:, ii, ii].astype(float)
                c_raw = X_inter.iloc[:, ii].to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(c_raw)
                x, c_raw = x[valid], c_raw[valid]
                y = _jitter_from_x(x, scale=0.32)

                fv_min, fv_max = float(np.nanmin(c_raw)), float(np.nanmax(c_raw))
                if np.isclose(fv_min, fv_max): fv_max = fv_min + 1.0
                scatter_norm = Normalize(vmin=fv_min, vmax=fv_max)

                ax.scatter(x, y, c=c_raw, cmap=RDBU_CMAP, norm=scatter_norm, s=22, edgecolors="none", alpha=0.85,
                           zorder=3)
                ax.axvline(0.0, color=MID_GREY, linestyle="--", linewidth=1.2, zorder=1)
                apply_academic_style(ax, grid=True)
                ax.set_yticks([])
                ax.tick_params(axis="y", left=False, labelleft=False)

                if i == 0:
                    ax.xaxis.set_ticks_position("top")
                    ax.xaxis.set_label_position("top")
                    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False,
                                   labelsize=tick_fontsize,
                                   width=1.0, length=5.0)
                    for lbl in ax.get_xticklabels():
                        lbl.set_fontweight("bold")
                        lbl.set_family(GLOBAL_FONT_FAMILY)
                else:
                    ax.tick_params(axis="x", top=False, bottom=False, labeltop=False, labelbottom=False)
                continue

            if i > j:
                value = float(sub_mat[i, j])
                rgba = RDBU_CMAP(heat_norm(value))
                ax.set_facecolor(rgba)

                r, g, b, _ = rgba
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                txt_color = "white" if luminance < 0.53 else "black"

                ax.text(
                    0.5, 0.5, f"{value:.2f}",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    family=GLOBAL_FONT_FAMILY,
                    color=txt_color,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                continue

            x = inter_arr[:, ii, jj].astype(float)
            c_raw = X_inter.iloc[:, ii].to_numpy(dtype=float)

            valid = np.isfinite(x) & np.isfinite(c_raw)
            x, c_raw = x[valid], c_raw[valid]
            y = _jitter_from_x(x, scale=0.30)

            fv_min, fv_max = float(np.nanmin(c_raw)), float(np.nanmax(c_raw))
            if np.isclose(fv_min, fv_max): fv_max = fv_min + 1.0
            scatter_norm = Normalize(vmin=fv_min, vmax=fv_max)

            ax.scatter(x, y, c=c_raw, cmap=RDBU_CMAP, norm=scatter_norm, s=22, edgecolors="none", alpha=0.95, zorder=3)
            ax.axvline(0.0, color=MID_GREY, linestyle="--", linewidth=1.2, zorder=1)
            apply_academic_style(ax, grid=True)
            ax.set_yticks([])
            ax.tick_params(axis="y", left=False, labelleft=False)

            if i == 0:
                ax.xaxis.set_ticks_position("top")
                ax.xaxis.set_label_position("top")
                ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, labelsize=18,
                               width=1.0, length=5)
                for lbl in ax.get_xticklabels():
                    lbl.set_fontweight("bold")
                    lbl.set_family(GLOBAL_FONT_FAMILY)
            else:
                ax.tick_params(axis="x", top=False, bottom=False, labeltop=False, labelbottom=False)

    for i in range(n):
        axes[i, 0].set_ylabel(
            format_feature_label(top_features[i]),
            fontsize=axis_label_fontsize, fontweight="bold", family=GLOBAL_FONT_FAMILY,
            rotation=90, labelpad=8,
        )

        if top_features[i] == "Particle size":
            axes[i, 0].yaxis.set_label_coords(-0.07, 0.32)

    for j in range(n):
        axes[-1, j].set_xlabel(
            format_feature_label(top_features[j]),
            fontsize=axis_label_fontsize, fontweight="bold", family=GLOBAL_FONT_FAMILY,
            rotation=0, ha="center", labelpad=8,
        )

        if top_features[j] == "Particle size":
            axes[-1, j].xaxis.set_label_coords(0.62, -0.07)

    cax = fig.add_axes([0.87, 0.29, 0.012, 0.47])
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=RDBU_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([])
    cbar.ax.tick_params(length=0)
    cbar.outline.set_linewidth(1.0)

    cbar.set_label(
        "Raw feature value / Mean |Interaction|",
        fontsize=20,
        fontweight="bold",
        family=GLOBAL_FONT_FAMILY,
        rotation=90,
        labelpad=30
    )

    cbar.ax.text(1.45, 1.00, "High", transform=cbar.ax.transAxes, ha="left", va="center", fontsize=20,
                 fontweight="bold", family=GLOBAL_FONT_FAMILY)
    cbar.ax.text(1.45, 0.00, "Low", transform=cbar.ax.transAxes, ha="left", va="center", fontsize=20, fontweight="bold",
                 family=GLOBAL_FONT_FAMILY)


    fig.text(
        0.47, 0.925,
        "SHAP interaction value",
        ha="center", va="center",
        fontsize=22, fontweight="bold", family=GLOBAL_FONT_FAMILY
    )

    save_figure(fig, save_stem)


def get_top_interaction_partners_for_feature(
        ranking_bundle: ShapRankingBundle,
        target_feature: str,
        top_n: int = 6,
) -> List[str]:
    pair_df = ranking_bundle.interaction_pair_df.copy()

    sub = pair_df[
        (pair_df["feature_a"] == target_feature) |
        (pair_df["feature_b"] == target_feature)
        ].copy()

    if sub.empty:
        return []

    sub["partner"] = np.where(
        sub["feature_a"] == target_feature,
        sub["feature_b"],
        sub["feature_a"], )

    sub = sub.drop_duplicates(subset=["partner"], keep="first")
    return sub["partner"].head(top_n).tolist()


# ============================================================================
# Figure 6: Core pairwise interaction scatter plots
# ============================================================================
def plot_figure6_core_pairwise_scatter(
        shap_interaction_values: Any,
        X_inter: pd.DataFrame,
        ranking_bundle: ShapRankingBundle,
        save_stem: Union[str, Path],
        main_feature: str = "Temperature",
        top_n: int = 6,
) -> None:
    inter_arr = normalize_shap_interaction_values(shap_interaction_values)
    X_inter = to_dataframe(X_inter)

    if main_feature not in X_inter.columns:
        raise ValueError(f"Main feature '{main_feature}' not found in X_inter columns.")

    partner_features = get_top_interaction_partners_for_feature(
        ranking_bundle=ranking_bundle,
        target_feature=main_feature,
        top_n=top_n,
    )

    if len(partner_features) == 0:
        raise ValueError(f"No interaction partners found for main feature '{main_feature}'.")

    i_main = X_inter.columns.get_loc(main_feature)

    fig, axes = plt.subplots(
        2, 3,
        figsize=(16.5, 10.0),
        dpi=FIG_DPI
    )
    axes_flat = axes.ravel()

    for idx, (ax, partner) in enumerate(zip(axes_flat, partner_features)):
        j = X_inter.columns.get_loc(partner)

        x = X_inter[main_feature].to_numpy(dtype=float)
        y = inter_arr[:, i_main, j]
        color_values = X_inter[partner].to_numpy(dtype=float)

        norm = Normalize(vmin=np.nanmin(color_values), vmax=np.nanmax(color_values))

        sc = ax.scatter(
            x, y,
            c=color_values,
            cmap=RDBU_CMAP,
            norm=norm,
            s=20,
            alpha=1.0,
            edgecolors="none")

        ax.axhline(0.0, color=MID_GREY, linestyle="--", linewidth=1.0)

        ax.set_title(
            f"{format_feature_label(main_feature)} × {format_feature_label(partner)}",
            fontsize=16,
            fontweight="bold",
            family=GLOBAL_FONT_FAMILY,
            pad=8)

        ax.set_xlabel(
            format_feature_label(main_feature),
            fontsize=16,
            fontweight="bold",
            family=GLOBAL_FONT_FAMILY,
            labelpad=6)

        ax.set_ylabel(
            "SHAP interaction value",
            fontsize=16,
            fontweight="bold",
            family=GLOBAL_FONT_FAMILY,
            labelpad=4)

        col = idx % 3
        if col == 0:
            ax.yaxis.set_label_coords(-0.11, 0.5)
        elif col == 1:
            ax.yaxis.set_label_coords(-0.12, 0.5)
        else:
            ax.yaxis.set_label_coords(-0.12, 0.5)

        ax.tick_params(axis="both", labelsize=16, width=1.0, length=4)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
            lbl.set_family(GLOBAL_FONT_FAMILY)

        apply_academic_style(ax, grid=True)

        cbar = fig.colorbar(
            sc,
            ax=ax,
            fraction=0.050,
            pad=0
        )
        cbar.ax.tick_params(labelsize=16, width=0.8, length=3)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_fontweight("bold")
            lbl.set_family(GLOBAL_FONT_FAMILY)

        cbar.set_label(
            format_feature_label(partner),
            fontsize=16,
            fontweight="bold",
            family=GLOBAL_FONT_FAMILY,
            rotation=270,
            labelpad=16
        )

    for k in range(len(partner_features), len(axes_flat)):
        axes_flat[k].axis("off")

    fig.suptitle(
        "Figure 6. Pairwise Interaction Scatter: Temperature vs Top 6 Partner Features",
        fontsize=18,
        fontweight="bold",
        family=GLOBAL_FONT_FAMILY,
        y=0.98
    )

    # Adjust spacing to avoid vertical crowding.
    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        bottom=0.08,
        top=0.90,
        wspace=0.36,
        hspace=0.24
    )
    fig.align_ylabels(axes)
    save_figure(fig, save_stem)


# ============================================================================
# Figure 7: Feature interaction network
# ============================================================================
def plot_figure7_feature_interaction_network(
        shap_values: Any,
        shap_interaction_values: Any,
        X: pd.DataFrame,
        save_stem: Union[str, Path]
) -> None:
    shap_arr = shap_values
    inter_arr = shap_interaction_values
    feature_names = list(X.columns)

    main_strength = np.mean(np.abs(shap_arr), axis=0)
    inter_strength = np.mean(np.abs(inter_arr), axis=0)

    # ---- Sort by importance: start at rightmost, then counterclockwise ----
    sorted_idx = np.argsort(-main_strength)
    sorted_features = [feature_names[i] for i in sorted_idx]

    G = nx.Graph()
    for i in sorted_idx:
        G.add_node(feature_names[i], weight=float(main_strength[i]))

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            G.add_edge(feature_names[i], feature_names[j], weight=float(inter_strength[i, j]))

    # ---- Strict circular layout ----
    n = len(sorted_features)
    radius = 0.90
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {
        feat: (radius * np.cos(theta), radius * np.sin(theta))
        for feat, theta in zip(sorted_features, angles)
    }

    # ---- Figure and axes ----
    fig, ax = plt.subplots(figsize=(8.8, 8.2), dpi=FIG_DPI)
    ax.set_axis_off()
    ax.set_aspect("equal")

    # ---- Deeper blue/red colormaps ----
    node_cmap = LinearSegmentedColormap.from_list(
        "node_blue_map",
        ["#D9E8F5", "#2C7FB8"]
    )
    edge_cmap = LinearSegmentedColormap.from_list(
        "edge_red_map",
        ["#F4D7DB", "#E79AA0", "#D85C5C"]
    )

    # ---- Node sizes and colors ----
    node_weights = np.array([G.nodes[n]["weight"] for n in sorted_features], dtype=float)
    nw_min = float(node_weights.min())
    nw_max = float(node_weights.max())
    nw_span = max(nw_max - nw_min, 1e-12)

    # Slightly reduce the blue node size.
    node_sizes = 100 + 1700 * (node_weights - nw_min) / nw_span
    node_norm = Normalize(vmin=nw_min, vmax=nw_max)

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=sorted_features,
        ax=ax,
        node_size=node_sizes,
        node_color=node_weights,
        cmap=node_cmap,
        edgecolors=DARK_GREY,
        linewidths=1.0,
    )

    # ---- Edge widths and colors ----
    edge_list = [(u, v) for u, v in G.edges()]
    edge_weights = np.array([G[u][v]["weight"] for u, v in edge_list], dtype=float)

    ew_min = float(edge_weights.min())
    ew_max = float(edge_weights.max())
    ew_span = max(ew_max - ew_min, 1e-12)

    # Slightly increase edge width to avoid overly thin links.
    edge_widths = 1.4 + 7.0 * (edge_weights - ew_min) / ew_span

    edge_norm = PowerNorm(gamma=0.75, vmin=ew_min, vmax=ew_max)
    edge_colors = edge_cmap(edge_norm(edge_weights))
    edge_alphas = 0.20 + 0.72 * (edge_weights - ew_min) / ew_span

    for (u, v), w, c, a in zip(edge_list, edge_widths, edge_colors, edge_alphas):
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=[(u, v)],
            width=float(w),
            edge_color=[c],
            alpha=float(a),
        )

    # ---- External labels ----
    # Move labels closer to the circle.
    label_radius = 1.02
    for feat, theta in zip(sorted_features, angles):
        x = label_radius * np.cos(theta)
        y = label_radius * np.sin(theta)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        if cos_t > 0.2:
            ha = "left"
        elif cos_t < -0.2:
            ha = "right"
        else:
            ha = "center"

        if sin_t > 0.2:
            va = "bottom"
        elif sin_t < -0.2:
            va = "top"
        else:
            va = "center"

        ax.text(
            x,
            y,
            format_feature_label(feat),
            fontsize=12,
            ha=ha,
            va=va,
        )

    # ---- Layout margins ----
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)

    # ---- Title ----
    fig.suptitle(
        "Figure 7. Feature Influence and Interaction Network",
        fontsize=12,
        y=0.97
    )

    # ---- Bottom horizontal colorbars ----
    node_sm = ScalarMappable(norm=node_norm, cmap=node_cmap)
    node_sm.set_array([])
    edge_sm = ScalarMappable(norm=edge_norm, cmap=edge_cmap)
    edge_sm.set_array([])

    cax1 = fig.add_axes([0.18, 0.08, 0.26, 0.020])
    cax2 = fig.add_axes([0.56, 0.08, 0.26, 0.020])

    cbar1 = fig.colorbar(node_sm, cax=cax1, orientation="horizontal")
    cbar1.set_label("Importance", fontsize=12)
    cbar1.ax.tick_params(labelsize=12)

    cbar2 = fig.colorbar(edge_sm, cax=cax2, orientation="horizontal")
    cbar2.set_label("Interaction Intensity", fontsize=12)
    cbar2.ax.tick_params(labelsize=12)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.16)
    save_figure(fig, save_stem)

# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from ydata_profiling import ProfileReport
    HAS_YDATA_PROFILING = True
except Exception:
    ProfileReport = None
    HAS_YDATA_PROFILING = False


# ============================================================================
# Global style
# ============================================================================
FIG_DPI = 600
SAVE_FORMATS = ("png", "pdf", "svg")
GLOBAL_FONT_FAMILY = "Times New Roman"

plt.rcParams["font.family"] = GLOBAL_FONT_FAMILY
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
# Project paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DATASET_FILE = str(DATA_DIR / "SS-ML.xlsx")
OUT_DIR = str(OUTPUT_ROOT / "Data_Statistics")

# ============================================================================
# Figure layout settings
# ============================================================================
RAINCLOUD_FIG_WIDTH = 18.5
RAINCLOUD_PANEL_HEIGHT = 4.8

HEATMAP_FIGSIZE = (12, 10)

PANEL_GRID_ADJUST = dict(
    left=0.075,
    right=0.965,
    bottom=0.015,
    top=0.965,
    wspace=0.28,
    hspace=0.15,
)


PANEL_AXIS_LABEL_FONTSIZE = 22
PANEL_TICK_FONTSIZE = 20
PANEL_LABEL_FONTSIZE = 22
PANEL_LABEL_X = -0.13
PANEL_LABEL_Y = 1.04

SPINE_WIDTH = 1.5
GRID_LINEWIDTH = 0.6
GRID_ALPHA = 0.35
TICK_WIDTH = 1.0
TICK_LENGTH = 5.0


# ============================================================================
# Raincloud style settings
# ============================================================================
MAIN_COLOR = "#406DA8"
EDGE_COLOR = "#7298C6"
POINT_FACE = "#7298C6"
POINT_EDGE = "#406DA8"

VIOLIN_ALPHA = 0.45
RAINCLOUD_BOX_WIDTH = 0.05
POINT_SIZE = 40
POINT_ALPHA = 0.4

POINT_LOC = -0.18
POINT_SCALE = 0.05
POINT_LEFT_MIN = -0.38
POINT_LEFT_MAX = -0.03

X_LEFT_LIMIT = -0.40
X_RIGHT_LIMIT = 0.60


# ============================================================================
# Heatmap style settings
# ============================================================================
CBAR_SHRINK = 1
CBAR_ASPECT = 28
CBAR_PAD = 0.02
CBAR_TICK_FONTSIZE = 18


HEATMAP_XTICK_FONTSIZE = 18
HEATMAP_YTICK_FONTSIZE = 18


# ============================================================================
# Feature configuration
# ============================================================================
FEATURE_CONFIG = [
    {"key": "CaO", "display": "CaO", "unit": "%", "aliases": ["cao"]},
    {"key": "MgO", "display": "MgO", "unit": "%", "aliases": ["mgo"]},
    {"key": "SiO2", "display": "SiO$_2$", "unit": "%", "aliases": ["sio2"]},
    {"key": "Fe2O3", "display": "Fe$_2$O$_3$", "unit": "%", "aliases": ["fe2o3"]},
    {"key": "MnO", "display": "MnO", "unit": "%", "aliases": ["mno"]},
    {
        "key": "Particle size",
        "display": "Particle size",
        "unit": "μm",
        "aliases": ["particlesize", "particle_size", "particle size"],
    },
    {
        "key": "Temperature",
        "display": "Temperature",
        "unit": "°C",
        "aliases": ["temperature", "temp"],
    },
    {
        "key": "CO2 partial pressure",
        "display": "CO$_2$ partial pressure",
        "unit": "MPa",
        "aliases": ["co2partialpressure", "co2_partial_pressure", "co2 partial pressure"],
    },
    {
        "key": "CO2 concentration",
        "display": "CO$_2$ concentration",
        "unit": "%",
        "aliases": ["co2concentration", "co2_concentration", "co2 concentration"],
    },
    {
        "key": "W/S",
        "display": "W/S",
        "unit": "-",
        "aliases": ["w/s", "ws", "water/solid", "watertosolid", "liquid/solid", "l/s"],
    },
    {
        "key": "Time",
        "display": "Time",
        "unit": "h",
        "aliases": ["time", "carbonationtime", "reactiontime"],
    },
]

TARGET_CONFIG = {
    "key": "CO2 sequestration",
    "display": "CS",
    "unit": "g-CO$_2$/kg-DSS",
}


# ============================================================================
# File utilities
# ============================================================================
def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(
    fig: plt.Figure,
    save_stem: Union[str, Path],
    formats: Sequence[str] = SAVE_FORMATS,
    bbox_inches: Optional[str] = None,
) -> None:
    save_stem = Path(save_stem)
    ensure_dir(save_stem.parent)

    for fmt in formats:
        fig.savefig(
            save_stem.with_suffix(f".{fmt}"),
            dpi=FIG_DPI,
            bbox_inches=bbox_inches,
        )

    plt.close(fig)


# ============================================================================
# Data loading and column handling
# ============================================================================
def read_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=0)
    return df.copy()


def normalize_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def get_display_name(feature: str) -> str:
    if feature == TARGET_CONFIG["key"]:
        return TARGET_CONFIG["display"]

    for item in FEATURE_CONFIG:
        if item["key"] == feature:
            return item["display"]

    return feature


def get_display_column_map(df: pd.DataFrame) -> Dict[str, str]:
    return {col: get_display_name(col) for col in df.columns}


def get_feature_unit_map() -> Dict[str, str]:
    unit_map = {item["key"]: item["unit"] for item in FEATURE_CONFIG}
    unit_map[TARGET_CONFIG["key"]] = TARGET_CONFIG["unit"]
    return unit_map


def get_ylabel(feature: str) -> str:
    unit_map = get_feature_unit_map()
    display_name = get_display_name(feature)
    unit = unit_map.get(feature, "")

    if unit and unit != "-":
        return f"{display_name} ({unit})"

    return display_name


def build_standardized_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.shape[1] < 2:
        raise ValueError("The dataset must contain at least one feature column and one target column.")

    out = pd.DataFrame()

    target_col_raw = df_raw.columns[-1]
    feature_source_cols = list(df_raw.columns[:-1])

    normalized_to_raw = {
        normalize_name(col): col
        for col in feature_source_cols
    }

    used_source_cols = set()

    for item in FEATURE_CONFIG:
        matched_col = None
        candidate_names = [item["key"]] + item.get("aliases", [])

        for candidate in candidate_names:
            candidate_norm = normalize_name(candidate)
            if candidate_norm in normalized_to_raw:
                matched_col = normalized_to_raw[candidate_norm]
                break

        if matched_col is not None:
            out[item["key"]] = pd.to_numeric(df_raw[matched_col], errors="coerce")
            used_source_cols.add(matched_col)

    out[TARGET_CONFIG["key"]] = pd.to_numeric(df_raw[target_col_raw], errors="coerce")

    unknown_feature_cols = [col for col in feature_source_cols if col not in used_source_cols]
    if unknown_feature_cols:
        print("[Info] Ignored columns not defined in FEATURE_CONFIG:")
        for col in unknown_feature_cols:
            print(f"  - {col}")

    if out[TARGET_CONFIG["key"]].notna().sum() == 0:
        raise ValueError(
            f"The last column '{target_col_raw}' was used as the target, "
            "but it cannot be converted to numeric values."
        )

    feature_keys = [item["key"] for item in FEATURE_CONFIG if item["key"] in out.columns]
    if len(feature_keys) == 0:
        raise ValueError(
            "No feature columns were recognized. Please check FEATURE_CONFIG and the dataset column names."
        )

    return out


def get_feature_keys(df: Optional[pd.DataFrame] = None) -> List[str]:
    ordered_keys = [item["key"] for item in FEATURE_CONFIG]

    if df is None:
        return ordered_keys

    return [key for key in ordered_keys if key in df.columns]


def get_feature_target(df: pd.DataFrame):
    feature_cols = get_feature_keys(df)
    target_col = TARGET_CONFIG["key"]
    return feature_cols, target_col


# ============================================================================
# Panel label helpers
# ============================================================================
def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = PANEL_LABEL_X,
    y: float = PANEL_LABEL_Y,
    fontsize: int = PANEL_LABEL_FONTSIZE,
) -> None:
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


def add_panel_labels(
    axes: Sequence[plt.Axes],
    n_panels: int,
    x: float = PANEL_LABEL_X,
    y: float = PANEL_LABEL_Y,
    fontsize: int = PANEL_LABEL_FONTSIZE,
) -> None:
    for i, ax in enumerate(list(axes)[:n_panels]):
        add_panel_label(
            ax=ax,
            label=f"({chr(97 + i)})",
            x=x,
            y=y,
            fontsize=fontsize,
        )


def apply_raincloud_axis_style(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(
        ylabel,
        fontsize=PANEL_AXIS_LABEL_FONTSIZE,
        fontweight="bold",
        family=GLOBAL_FONT_FAMILY,
    )

    ax.tick_params(
        axis="y",
        labelsize=PANEL_TICK_FONTSIZE,
        length=TICK_LENGTH,
        width=TICK_WIDTH,
    )

    ax.tick_params(
        axis="x",
        length=0,
        width=0,
        labelbottom=False,
        bottom=False,
    )

    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_family(GLOBAL_FONT_FAMILY)

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("black")

    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
        color="#7A7A7A",
        zorder=0,
    )


# ============================================================================
# Raincloud plot
# ============================================================================
def _half_violin(ax: plt.Axes, data: pd.DataFrame, feature: str, color: str = MAIN_COLOR) -> None:
    values = pd.to_numeric(data[feature], errors="coerce").dropna().values
    n = len(values)

    if n == 0:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            fontsize=PANEL_TICK_FONTSIZE,
            fontweight="bold",
            family=GLOBAL_FONT_FAMILY,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        apply_raincloud_axis_style(ax, get_ylabel(feature))
        ax.set_xlim(X_LEFT_LIMIT, X_RIGHT_LIMIT)
        return

    plot_df = pd.DataFrame({
        "group": [""] * n,
        feature: values,
    })

    sns.violinplot(
        data=plot_df,
        x="group",
        y=feature,
        ax=ax,
        color=color,
        inner=None,
        linewidth=1.2,
        cut=0,
        width=1.05,
        saturation=1,
        zorder=1,
    )

    for coll in ax.collections:
        try:
            paths = coll.get_paths()
            if not paths:
                continue

            for path in paths:
                vertices = path.vertices
                vertices[:, 0] = np.clip(vertices[:, 0], 0.0, np.inf)

            coll.set_alpha(VIOLIN_ALPHA)
            coll.set_edgecolor(EDGE_COLOR)
            coll.set_linewidth(1.0)
        except Exception:
            pass

    sns.boxplot(
        data=plot_df,
        x="group",
        y=feature,
        ax=ax,
        width=RAINCLOUD_BOX_WIDTH,
        notch=True,
        showcaps=True,
        showfliers=False,
        boxprops={
            "facecolor": MAIN_COLOR,
            "edgecolor": EDGE_COLOR,
            "linewidth": 1.5,
            "alpha": 1.0,
        },
        whiskerprops={
            "color": EDGE_COLOR,
            "linewidth": 1.5,
        },
        capprops={
            "color": EDGE_COLOR,
            "linewidth": 1.5,
        },
        medianprops={
            "color": "black",
            "linewidth": 1.6,
        },
        zorder=3,
    )

    rng = np.random.default_rng(2026)
    x_points = rng.normal(loc=POINT_LOC, scale=POINT_SCALE, size=n)
    x_points = np.clip(x_points, POINT_LEFT_MIN, POINT_LEFT_MAX)

    ax.scatter(
        x_points,
        values,
        s=POINT_SIZE,
        facecolors=POINT_FACE,
        edgecolors=POINT_EDGE,
        linewidths=1.0,
        alpha=POINT_ALPHA,
        zorder=4,
    )

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.set_xlim(X_LEFT_LIMIT, X_RIGHT_LIMIT)

    apply_raincloud_axis_style(ax, get_ylabel(feature))


def plot_raincloud_grid(
    df: pd.DataFrame,
    columns_to_plot: List[str],
    save_stem: Union[str, Path],
    ncols: int = 3,
) -> None:
    n = len(columns_to_plot)
    if n == 0:
        raise ValueError("No columns were provided for the raincloud plot.")

    nrows = (n + ncols - 1) // ncols

    raincloud_figsize = (
        RAINCLOUD_FIG_WIDTH,
        RAINCLOUD_PANEL_HEIGHT * nrows,
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=raincloud_figsize,
        dpi=FIG_DPI,
    )

    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        _half_violin(ax, df[[col]], col, color=MAIN_COLOR)
        ax.set_title("")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    add_panel_labels(axes, n_panels=n)

    for ax in axes[:n]:
        ax.yaxis.set_label_coords(-0.11, 0.5)

    fig.subplots_adjust(**PANEL_GRID_ADJUST)

    save_figure(fig, save_stem, bbox_inches=None)


# ============================================================================
# Correlation heatmap
# ============================================================================
def plot_corr_heatmap(
    corr_df: pd.DataFrame,
    save_stem: Union[str, Path],
    cmap=None,
    show_values: bool = False,
) -> None:
    display_cols = [get_display_name(c) for c in corr_df.columns]

    corr_df = corr_df.copy()
    corr_df.columns = display_cols
    corr_df.index = display_cols

    if cmap is None:
        cmap = "coolwarm"

    fig = plt.figure(figsize=HEATMAP_FIGSIZE, dpi=FIG_DPI)

    ax = sns.heatmap(
        corr_df,
        annot=show_values,
        fmt=".2f",
        cmap=cmap,
        square=True,
        linewidths=1.5,
        cbar=True,
        annot_kws={
            "fontsize": 13,
            "fontweight": "bold",
            "family": GLOBAL_FONT_FAMILY,
            "color": "black",
        },
        cbar_kws={
            "shrink": CBAR_SHRINK,
            "aspect": CBAR_ASPECT,
            "pad": CBAR_PAD,
        },
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

    plt.xticks(
        fontsize=HEATMAP_XTICK_FONTSIZE,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        fontsize=HEATMAP_YTICK_FONTSIZE,
        rotation=0,
    )

    plt.tight_layout()

    save_figure(fig, save_stem, bbox_inches="tight")


# ============================================================================
# Summary table
# ============================================================================
def build_feature_target_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols, target_col = get_feature_target(df)
    cols = feature_cols + [target_col]

    summary_rows = []

    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()

        summary_rows.append({
            "variable": col,
            "display_name": get_display_name(col),
            "range_min": values.min() if len(values) else np.nan,
            "range_max": values.max() if len(values) else np.nan,
            "median": values.median() if len(values) else np.nan,
            "std": values.std() if len(values) else np.nan,
        })

    return pd.DataFrame(summary_rows)


# ============================================================================
# Profiling report
# ============================================================================
def generate_ydata_profiling_report(
    df: pd.DataFrame,
    save_path: Union[str, Path],
    title: str = "YData Profiling Report",
) -> None:
    if not HAS_YDATA_PROFILING:
        print("[Warning] ydata-profiling is not installed. Skip profiling report.")
        return

    df_for_profile = df.rename(columns=get_display_column_map(df)).copy()

    profile = ProfileReport(
        df_for_profile,
        title=title,
        explorative=True,
    )

    profile.to_file(str(save_path))


# ============================================================================
# Main workflow
# ============================================================================
def main(
    dataset_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> None:
    dataset_file = dataset_file or DATASET_FILE
    out_dir = Path(out_dir or OUT_DIR)

    ensure_dir(out_dir)

    df_raw = read_dataset(dataset_file)
    df = build_standardized_dataframe(df_raw)

    feature_cols, target_col = get_feature_target(df)
    distribution_cols = feature_cols + [target_col]

    full_numeric_df = df[distribution_cols].select_dtypes(include="number").copy()

    raincloud_stem = out_dir / "raincloud_features"
    pearson_stem = out_dir / "pearson_correlation"
    spearman_stem = out_dir / "spearman_correlation"
    pearson_annot_stem = out_dir / "pearson_correlation_annot"
    spearman_annot_stem = out_dir / "spearman_correlation_annot"

    data_copy_path = out_dir / "data_used_for_statistics.csv"
    summary_table_path = out_dir / "feature_target_summary_table.csv"
    profiling_report_path = out_dir / "ydata_profiling_report.html"

    plot_raincloud_grid(
        df=df,
        columns_to_plot=distribution_cols,
        save_stem=raincloud_stem,
        ncols=3,
    )

    pearson_corr = full_numeric_df.corr(method="pearson")
    spearman_corr = full_numeric_df.corr(method="spearman")

    plot_corr_heatmap(
        corr_df=pearson_corr,
        save_stem=pearson_stem,
        cmap="coolwarm",
        show_values=False,
    )

    plot_corr_heatmap(
        corr_df=spearman_corr,
        save_stem=spearman_stem,
        cmap="coolwarm",
        show_values=False,
    )

    plot_corr_heatmap(
        corr_df=pearson_corr,
        save_stem=pearson_annot_stem,
        cmap="coolwarm",
        show_values=True,
    )

    plot_corr_heatmap(
        corr_df=spearman_corr,
        save_stem=spearman_annot_stem,
        cmap="coolwarm",
        show_values=True,
    )

    df.to_csv(data_copy_path, index=False, encoding="utf-8-sig")

    summary_table_df = build_feature_target_summary_table(df)
    summary_table_df.to_csv(summary_table_path, index=False, encoding="utf-8-sig")

    generate_ydata_profiling_report(
        df=df,
        save_path=profiling_report_path,
        title="Raw Data Profiling Report",
    )

    print("=" * 80)
    print("Data statistics finished.")
    print(f"Dataset: {Path(dataset_file).name}")
    print(f"Feature count used: {len(feature_cols)}")
    print("Feature order actually used:")

    for i, key in enumerate(feature_cols, start=1):
        print(f"  {i:02d}. key={key}  display={get_display_name(key)}")

    print(f"Target used: {target_col} -> {get_display_name(target_col)}")

    print("Saved figures:")
    for stem in [
        raincloud_stem,
        pearson_stem,
        spearman_stem,
        pearson_annot_stem,
        spearman_annot_stem,
    ]:
        for fmt in SAVE_FORMATS:
            print(f"  - {stem.with_suffix(f'.{fmt}')}")

    print("Saved tables and reports:")
    print(f"  - {data_copy_path}")
    print(f"  - {summary_table_path}")
    print(f"  - {profiling_report_path}")


if __name__ == "__main__":
    main()

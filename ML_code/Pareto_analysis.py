from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from openpyxl.utils.cell import column_index_from_string


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DATASET_FILE = str(DATA_DIR / "SS-ML.xlsx")
OUT_DIR = str(OUTPUT_ROOT / "Pareto_LCA_integrated")

SHEET_NAME = 0
HEADER_ROW = 0

# -----------------------------------------------------------------------------
# 1.1 Pareto objectives
# -----------------------------------------------------------------------------
COST_EXCEL_COL = "BL"
CENET_EXCEL_COL = "AX"
CO2_UPTAKE_EXCEL_COL = "L"

WS_EXCEL_COL = "J"
TIME_EXCEL_COL = "K"

COST_KEY = "Ctotal"
CENET_KEY = "CEnet"
UPTAKE_KEY = "CO2_uptake"
WS_KEY = "W/S"
TIME_KEY = "Time"

COST_DISPLAY = r"$\mathbf{Cost}$"
COST_UNIT = "USD/t-DSS"
CENET_DISPLAY = r"$\mathbf{CE}_{\mathbf{net}}$"
CENET_UNIT = r"kg-$\mathbf{CO}_{\mathbf{2}}$ eq/t-DSS"
UPTAKE_DISPLAY = r"$\mathbf{CO}_{\mathbf{2}}$ sequestration"
UPTAKE_UNIT = r"kg-$\mathbf{CO}_{\mathbf{2}}$/t-DSS"

# Optional projection line for the 2D Pareto plot.
DRAW_PARETO_PROJECTION_LINE = False

# Inset axis limits.
INSET_X_RANGE = (0.0, 100.0)
INSET_Y_RANGE = (-120.0, 200.0)

# -----------------------------------------------------------------------------
# 1.2 Eighteen environmental-impact totals
# Fill one Excel column letter for each impact.
# Keep the names in the same order as your LCA method/output table.
# -----------------------------------------------------------------------------
IMPACT_TOTAL_EXCEL_COLS: Dict[str, Optional[str]] = {
    "Climate change": None,
    "Fine particulate matter formation": None,
    "Fossil resource scarcity": None,
    "Freshwater ecotoxicity": None,
    "Freshwater eutrophication": None,
    "Human carcinogenic toxicity": None,
    "Human non-carcinogenic toxicity": None,
    "Ionizing radiation": None,
    "Land use": None,
    "Marine ecotoxicity": None,
    "Marine eutrophication": None,
    "Mineral resource scarcity": None,
    "Ozone formation, human health": None,
    "Ozone formation, terrestrial ecosystems": None,
    "Stratospheric ozone depletion": None,
    "Terrestrial acidification": None,
    "Terrestrial ecotoxicity": None,
    "Water consumption": None,
}
# -----------------------------------------------------------------------------
# 1.3 Stage-specific environmental impacts
# -----------------------------------------------------------------------------
STAGE_NAMES = [
    "S1 CO$_2$ supply",
    "S2 Grinding",
    "S3 Slurry preparation\n& heated carbonation",
    "S4 Filtration",
    "S5 Drying",
]

STAGE_IMPACT_EXCEL_COLS: Dict[str, Dict[str, Optional[str]]] = {
    stage: {impact: None for impact in IMPACT_TOTAL_EXCEL_COLS}
    for stage in STAGE_NAMES
}

# -----------------------------------------------------------------------------
# 1.4 Stage costs for K-solution cost sensitivity
# Fill five stage-cost columns if available.
# -----------------------------------------------------------------------------
STAGE_COST_EXCEL_COLS: Dict[str, Optional[str]] = {
    "S1 CO$_2$ supply": None,
    "S2 Grinding": None,
    "S3 Slurry preparation\n& heated carbonation": None,
    "S4 Filtration": None,
    "S5 Drying": None,
}

# Climate-impact category used in the K-stage climate tornado plot.
CLIMATE_IMPACT_NAME = "Climate change"

# One-at-a-time perturbation for stage inventories and uptake.
SENSITIVITY_FRACTION = 0.20
INCLUDE_UPTAKE_IN_CLIMATE_SENSITIVITY = True

# A step of 0.05 yields 228 mixed-weight combinations.
WEIGHT_GRID_STEP = 0.05


# -----------------------------------------------------------------------------
# 1.5 Optional feature columns retained in exported representative-solution table
# -----------------------------------------------------------------------------
FEATURE_CONFIG = [
    {"key": "CaO", "aliases": ["cao"]},
    {"key": "W/S", "aliases": ["w/s", "ws"]},
    {"key": "Time", "aliases": ["time"]},
]


# =============================================================================
# 2. GLOBAL PLOTTING STYLE
# =============================================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.unicode_minus"] = False

ALL_POINT_COLOR = "#406DA8"
MIDDLE_COLOR = "#CEC7D3"
CMAP_HIGH_COLOR = "#BC5D59"
PARETO_POINT_COLOR = "#0057B8"
REPRESENTATIVE_POINT_COLOR = "#FF0000"

CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "custom_blue_purple_red",
    [ALL_POINT_COLOR, MIDDLE_COLOR, CMAP_HIGH_COLOR],
    N=256,
)

ALL_POINT_SIZE_3D = 58
PARETO_POINT_SIZE_3D = 80
REPRESENTATIVE_POINT_SIZE_3D = 250
ALL_POINT_SIZE_2D = 58
PARETO_POINT_SIZE_2D = 80
REPRESENTATIVE_POINT_SIZE_2D = 250

ALL_MARKER = "o"
PARETO_MARKER = "o"
REPRESENTATIVE_MARKER = "*"
POINT_EDGE_COLOR = "white"
POINT_EDGE_WIDTH = 0.45
PARETO_EDGE_WIDTH = 0.75
REPRESENTATIVE_EDGE_WIDTH = 0.85

LINE_WIDTH = 1.5
TICK_LENGTH = 5.0
TICK_WIDTH = 1.0
GRID_COLOR = "#7A7A7A"
GRID_LINEWIDTH = 0.6
GRID_ALPHA = 0.35
FIG_DPI = 600
SAVE_FORMATS = ("png", "pdf", "svg")

# 2D Pareto colorbar geometry.
# Match the colorbar height to the 2D axis.
PARETO_2D_CBAR_WIDTH_RATIO = 0.025
PARETO_2D_CBAR_PAD_RATIO = 0.020


# =============================================================================
# 3. BASIC UTILITIES
# =============================================================================
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(
    fig: plt.Figure,
    stem: str | Path,
    tight: bool = False,
    pad_inches: float = 0.02,
) -> None:
    stem = Path(stem)
    ensure_dir(stem.parent)
    kwargs = {"dpi": FIG_DPI}
    if tight:
        kwargs["bbox_inches"] = "tight"
        kwargs["pad_inches"] = pad_inches
    for ext in SAVE_FORMATS:
        fig.savefig(stem.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)


def normalize_name(name: object) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\n", "")
        .replace("\r", "")
    )


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.replace("\u200b", "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
    )
    return out


def get_series_by_excel_col(dataframe: pd.DataFrame, excel_col: str) -> pd.Series:
    col_index = column_index_from_string(excel_col) - 1
    if col_index < 0 or col_index >= dataframe.shape[1]:
        raise ValueError(f"Excel column {excel_col} is outside the worksheet range.")
    return dataframe.iloc[:, col_index]


def get_label(display: str, unit: str) -> str:
    return f"{display} ({unit})" if unit and unit != "-" else display


def build_standardized_feature_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    feature_df = pd.DataFrame(index=df_raw.index)
    normalized_to_raw = {normalize_name(col): col for col in df_raw.columns}

    for item in FEATURE_CONFIG:
        for candidate in [item["key"]] + item.get("aliases", []):
            key = normalize_name(candidate)
            if key in normalized_to_raw:
                matched_col = normalized_to_raw[key]
                feature_df[item["key"]] = pd.to_numeric(df_raw[matched_col], errors="coerce")
                break
    return feature_df


def configured(mapping: Mapping[str, Optional[str]]) -> bool:
    return bool(mapping) and all(value not in (None, "") for value in mapping.values())


def configured_nested(mapping: Mapping[str, Mapping[str, Optional[str]]]) -> bool:
    return bool(mapping) and all(configured(inner) for inner in mapping.values())


def extract_excel_columns(
    df_raw: pd.DataFrame,
    mapping: Mapping[str, str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=df_raw.index)
    for name, col_letter in mapping.items():
        out[name] = pd.to_numeric(get_series_by_excel_col(df_raw, col_letter), errors="coerce")
    return out


def set_bold_ticklabels_2d(ax: plt.Axes, tick_fontsize: int = 16) -> None:
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
    )
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontfamily("Times New Roman")


def set_bold_ticklabels_3d(ax, tick_fontsize: int = 16) -> None:
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
    )
    ax.zaxis.set_tick_params(
        labelsize=tick_fontsize,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
    )
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        for tick in axis.get_ticklabels():
            tick.set_fontweight("bold")
            tick.set_fontfamily("Times New Roman")


def apply_2d_style(ax: plt.Axes, grid: bool = True) -> None:
    if grid:
        ax.grid(
            True,
            linestyle="--",
            linewidth=GRID_LINEWIDTH,
            alpha=GRID_ALPHA,
            color=GRID_COLOR,
        )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    set_bold_ticklabels_2d(ax, 16)


def style_3d_axis(ax) -> None:
    ax.grid(True)
    ax.xaxis.line.set_linewidth(LINE_WIDTH)
    ax.yaxis.line.set_linewidth(LINE_WIDTH)
    ax.zaxis.line.set_linewidth(LINE_WIDTH)
    ax.xaxis.line.set_color("black")
    ax.yaxis.line.set_color("black")
    ax.zaxis.line.set_color("black")

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 1))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"].update(
            {
                "color": (0.48, 0.48, 0.48, GRID_ALPHA),
                "linewidth": GRID_LINEWIDTH,
            }
        )


def add_left_z_frame_line_3d(ax) -> None:
    x_left = ax.get_xlim()[0]
    y_front = ax.get_ylim()[0]
    z_low, z_high = ax.get_zlim()
    ax.plot(
        [x_left, x_left],
        [y_front, y_front],
        [z_low, z_high],
        color=(0.48, 0.48, 0.48, GRID_ALPHA),
        linewidth=GRID_LINEWIDTH,
        zorder=0,
    )


def add_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    norm: Normalize,
    label: str,
    pad: float = 0.02,
    shrink: float = 0.58,
    fraction: float = 0.026,
    aspect: int = 34,
    label_mode: str = "title",
    outline_linewidth: float = 1.0,
):
    sm = ScalarMappable(norm=norm, cmap=CUSTOM_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        pad=pad,
        shrink=shrink,
        fraction=fraction,
        aspect=aspect,
    )

    if label_mode == "vertical":
        cbar.ax.set_title("")
        cbar.ax.yaxis.set_label_position("right")
        cbar.ax.set_ylabel(
            label,
            fontsize=18,
            fontweight="bold",
            rotation=90,
            labelpad=18,
            fontfamily="Times New Roman",
        )
    else:
        cbar.set_label("")
        cbar.ax.set_title(
            label,
            fontsize=18,
            fontweight="bold",
            pad=8,
            fontfamily="Times New Roman",
        )

    cbar.ax.tick_params(labelsize=14, width=TICK_WIDTH, length=TICK_LENGTH)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontfamily("Times New Roman")

    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(outline_linewidth)
    return cbar


# =============================================================================
# 4. PARETO ANALYSIS
# =============================================================================
def get_3d_objective_matrix(
    dataframe: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    minimize_x: bool = True,
    minimize_y: bool = True,
    maximize_z: bool = True,
) -> np.ndarray:
    values = dataframe[[x_col, y_col, z_col]].to_numpy(dtype=float).copy()
    if not minimize_x:
        values[:, 0] = -values[:, 0]
    if not minimize_y:
        values[:, 1] = -values[:, 1]
    if maximize_z:
        values[:, 2] = -values[:, 2]
    return values


def normalize_objective_matrix(objective_matrix: np.ndarray) -> np.ndarray:
    obj_min = np.nanmin(objective_matrix, axis=0)
    obj_max = np.nanmax(objective_matrix, axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0
    return (objective_matrix - obj_min) / obj_range


def pareto_frontier_3d(
    dataframe: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    minimize_x: bool = True,
    minimize_y: bool = True,
    maximize_z: bool = True,
    eps: float = 1e-9,
) -> np.ndarray:
    objective_values = get_3d_objective_matrix(
        dataframe,
        x_col,
        y_col,
        z_col,
        minimize_x=minimize_x,
        minimize_y=minimize_y,
        maximize_z=maximize_z,
    )

    n = objective_values.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        current = objective_values[i]
        dominated_by_others = (
            np.all(objective_values <= current + eps, axis=1)
            & np.any(objective_values < current - eps, axis=1)
        )
        dominated_by_others[i] = False
        if np.any(dominated_by_others):
            is_pareto[i] = False
    return is_pareto


def normalized_weights(weights: Sequence[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    if arr.shape != (3,):
        raise ValueError("Pareto weights must contain exactly three values.")
    if np.any(arr < 0) or np.isclose(arr.sum(), 0.0):
        raise ValueError("Pareto weights must be non-negative and not all zero.")
    return arr / arr.sum()


def compromise_local_index(
    pareto_data: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    weights: Sequence[float] = (1.0, 1.0, 1.0),
) -> int:
    objective_values = get_3d_objective_matrix(
        pareto_data,
        x_col,
        y_col,
        z_col,
        minimize_x=True,
        minimize_y=True,
        maximize_z=True,
    )
    normalized_values = normalize_objective_matrix(objective_values)
    w = normalized_weights(weights)
    distances = np.sqrt(np.sum(w * normalized_values**2, axis=1))
    return int(np.argmin(distances))


def select_four_representative_solutions(
    pareto_data: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    weights: Sequence[float] = (1.0, 1.0, 1.0),
) -> pd.DataFrame:
    if pareto_data.empty:
        return pd.DataFrame()

    # U: Pareto solution with the maximum CO2 sequestration.
    idx_u = pareto_data[z_col].idxmax()

    # E: Pareto solution with the minimum net carbon emission.
    idx_e = pareto_data[y_col].idxmin()

    # C: Pareto solution with the minimum process cost.
    idx_c = pareto_data[x_col].idxmin()

    # K: Compromise solution closest to the normalized ideal point.
    idx_k = pareto_data.index[
        compromise_local_index(
            pareto_data,
            x_col,
            y_col,
            z_col,
            weights,
        )
    ]

    scheme_info = [
        ("U", "Maximum CO$_2$ sequestration", idx_u),
        ("E", "Minimum net carbon emission", idx_e),
        ("C", "Minimum cost", idx_c),
        ("K", "Balanced solution", idx_k),
    ]

    rows = []
    for label, role, idx in scheme_info:
        row = pareto_data.loc[idx].copy()
        row["Scenario_Label"] = label
        row["Scenario_Role"] = role
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def generate_simplex_weight_grid(step: float) -> List[Tuple[float, float, float]]:
    if step <= 0 or step > 1:
        raise ValueError("WEIGHT_GRID_STEP must be in (0, 1].")

    n = int(round(1.0 / step))
    weights: List[Tuple[float, float, float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            w = np.array([i, j, k], dtype=float) / n
            # Exclude vertices with two zero weights; retain all mixed preferences.
            if np.count_nonzero(w) >= 2:
                weights.append((float(w[0]), float(w[1]), float(w[2])))
    return weights


def run_weight_grid_sensitivity(
    pareto_data: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    representative_df: pd.DataFrame,
    out_dir: str,
) -> pd.DataFrame:
    records = []
    for weights in generate_simplex_weight_grid(WEIGHT_GRID_STEP):
        local_idx = compromise_local_index(
            pareto_data,
            x_col,
            y_col,
            z_col,
            weights,
        )
        row = pareto_data.iloc[local_idx]
        records.append(
            {
                "W_Cost": weights[0],
                "W_CEnet": weights[1],
                "W_Uptake": weights[2],
                "Selected_Original_Excel_Row": row["Original_row_in_excel"],
                "Cost": row[x_col],
                "CEnet": row[y_col],
                "CO2_uptake": row[z_col],
            }
        )

    sensitivity_df = pd.DataFrame(records)
    sensitivity_df.to_csv(
        os.path.join(out_dir, "pareto_weight_grid_sensitivity.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    counts = (
        sensitivity_df.groupby("Selected_Original_Excel_Row", as_index=False)
        .size()
        .rename(columns={"size": "Selection_count"})
        .sort_values("Selection_count", ascending=False)
        .reset_index(drop=True)
    )
    counts["Selection_frequency_%"] = counts["Selection_count"] / len(sensitivity_df) * 100.0
    counts.to_csv(
        os.path.join(out_dir, "pareto_weight_selection_frequency.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    plot_weight_selection_frequency(
        counts,
        representative_df,
        out_dir,
    )
    return sensitivity_df


# =============================================================================
# 5. PARETO PLOTS
# =============================================================================
def make_base_legend_handles(include_schemes: bool = False) -> List[Line2D]:
    handles = [
        Line2D(
            [0], [0],
            marker=PARETO_MARKER,
            linestyle="None",
            markerfacecolor=PARETO_POINT_COLOR,
            markeredgecolor=POINT_EDGE_COLOR,
            markeredgewidth=PARETO_EDGE_WIDTH,
            markersize=11,
            label="Pareto samples",
        ),
        Line2D(
            [0], [0],
            marker=REPRESENTATIVE_MARKER,
            linestyle="None",
            markerfacecolor=REPRESENTATIVE_POINT_COLOR,
            markeredgecolor=POINT_EDGE_COLOR,
            markeredgewidth=REPRESENTATIVE_EDGE_WIDTH,
            markersize=13,
            label="Representative samples",
        ),
    ]

    if include_schemes:
        scheme_labels = [
            ("U", "Maximum CO$_2$ sequestration"),
            ("E", "Minimum net carbon emission"),
            ("C", "Minimum cost"),
            ("K", "Balanced solution"),
        ]
        for label, text in scheme_labels:
            handles.append(
                Line2D(
                    [0], [0],
                    marker=rf"$\mathbf{{{label}}}$",
                    linestyle="None",
                    markerfacecolor="black",
                    markeredgecolor="none",
                    markersize=12,
                    label=text,
                )
            )
    return handles


def add_normal_legend(
    ax,
    handles: Sequence[Line2D],
    fontsize: int = 14,
    loc: str = "upper left",
    bbox_to_anchor: Tuple[float, float] = (0.0, 0.98),
):
    legend = ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=fontsize,
        frameon=True,
        facecolor="white",
        edgecolor="#BBBBBB",
        framealpha=0.95,
        handlelength=1.2,
        borderpad=0.45,
        labelspacing=0.4,
        prop={"family": "Times New Roman", "size": fontsize},
    )
    for text in legend.get_texts():
        text.set_fontfamily("Times New Roman")
    return legend


def plot_pareto_3d(
    analysis_data: pd.DataFrame,
    pareto_data: pd.DataFrame,
    representative_df: pd.DataFrame,
    out_dir: str,
) -> None:
    cost_label = get_label(COST_DISPLAY, COST_UNIT)
    ce_label = get_label(CENET_DISPLAY, CENET_UNIT)
    uptake_label = get_label(UPTAKE_DISPLAY, UPTAKE_UNIT)

    uptake_min = float(analysis_data[UPTAKE_KEY].min())
    uptake_max = float(analysis_data[UPTAKE_KEY].max())
    uptake_pad = 0.05 * (uptake_max - uptake_min) if uptake_max > uptake_min else 1.0
    uptake_norm = Normalize(uptake_min - uptake_pad, uptake_max + uptake_pad)

    fig = plt.figure(figsize=(12.2, 8.8), dpi=FIG_DPI)
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    ax.set_proj_type("ortho")

    ax.scatter(
        analysis_data[CENET_KEY],
        analysis_data[COST_KEY],
        analysis_data[UPTAKE_KEY],
        c=analysis_data[UPTAKE_KEY],
        cmap=CUSTOM_CMAP,
        norm=uptake_norm,
        s=ALL_POINT_SIZE_3D,
        marker=ALL_MARKER,
        alpha=1.0,
        edgecolors=POINT_EDGE_COLOR,
        linewidths=POINT_EDGE_WIDTH,
        depthshade=False,
        zorder=1,
    )
    ax.scatter(
        pareto_data[CENET_KEY],
        pareto_data[COST_KEY],
        pareto_data[UPTAKE_KEY],
        color=PARETO_POINT_COLOR,
        s=PARETO_POINT_SIZE_3D,
        marker=PARETO_MARKER,
        alpha=1.0,
        edgecolors=POINT_EDGE_COLOR,
        linewidths=PARETO_EDGE_WIDTH,
        depthshade=False,
        zorder=3,
    )
    ax.scatter(
        representative_df[CENET_KEY],
        representative_df[COST_KEY],
        representative_df[UPTAKE_KEY],
        color=REPRESENTATIVE_POINT_COLOR,
        s=REPRESENTATIVE_POINT_SIZE_3D,
        marker=REPRESENTATIVE_MARKER,
        alpha=1.0,
        edgecolors=POINT_EDGE_COLOR,
        linewidths=REPRESENTATIVE_EDGE_WIDTH,
        depthshade=False,
        zorder=5,
    )

    x_min, x_max = analysis_data[CENET_KEY].min(), analysis_data[CENET_KEY].max()
    y_min, y_max = analysis_data[COST_KEY].min(), analysis_data[COST_KEY].max()
    z_min, z_max = analysis_data[UPTAKE_KEY].min(), analysis_data[UPTAKE_KEY].max()

    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    z_pad = 0.05 * (z_max - z_min) if z_max > z_min else 1.0

    ax.set_xlim(x_max + x_pad, x_min - x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_zlim(z_min - z_pad, z_max + z_pad)

    ax.set_xlabel(ce_label, fontsize=18, fontweight="bold", labelpad=18)
    ax.set_ylabel(cost_label, fontsize=18, fontweight="bold", labelpad=18)
    ax.set_zlabel("")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.view_init(elev=24, azim=-58)

    style_3d_axis(ax)
    add_left_z_frame_line_3d(ax)
    set_bold_ticklabels_3d(ax, 16)
    add_normal_legend(
        ax,
        make_base_legend_handles(include_schemes=False),
        fontsize=16,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
    )
    add_colorbar(
        fig,
        ax,
        uptake_norm,
        uptake_label,
        pad=0.04,
        shrink=0.58,
        fraction=0.026,
        aspect=34,
    )

    fig.subplots_adjust(left=0.02, right=0.84, bottom=0.02, top=0.97)
    save_figure(fig, Path(out_dir) / "pareto_3d_scatter", tight=True)


def plot_pareto_2d_projection(
    analysis_data: pd.DataFrame,
    pareto_data: pd.DataFrame,
    representative_df: pd.DataFrame,
    out_dir: str,
) -> None:
    cost_label = get_label(COST_DISPLAY, COST_UNIT)
    ce_label = get_label(CENET_DISPLAY, CENET_UNIT)
    uptake_label = get_label(UPTAKE_DISPLAY, UPTAKE_UNIT)

    uptake_min = float(analysis_data[UPTAKE_KEY].min())
    uptake_max = float(analysis_data[UPTAKE_KEY].max())
    uptake_pad = 0.05 * (uptake_max - uptake_min) if uptake_max > uptake_min else 1.0
    uptake_norm = Normalize(uptake_min - uptake_pad, uptake_max + uptake_pad)

    pareto_sorted = pareto_data.sort_values(COST_KEY).reset_index(drop=True)

    fig, ax2 = plt.subplots(figsize=(8.5, 6.5), dpi=FIG_DPI)
    ax2.scatter(
        analysis_data[COST_KEY],
        analysis_data[CENET_KEY],
        c=analysis_data[UPTAKE_KEY],
        cmap=CUSTOM_CMAP,
        norm=uptake_norm,
        s=ALL_POINT_SIZE_2D,
        marker=ALL_MARKER,
        alpha=1.0,
        edgecolors=POINT_EDGE_COLOR,
        linewidths=POINT_EDGE_WIDTH,
        zorder=1,
    )

    if DRAW_PARETO_PROJECTION_LINE and len(pareto_sorted) >= 2:
        ax2.plot(
            pareto_sorted[COST_KEY],
            pareto_sorted[CENET_KEY],
            color=PARETO_POINT_COLOR,
            linestyle="--",
            linewidth=1.8,
            zorder=2,
        )

    ax2.scatter(
        pareto_sorted[COST_KEY],
        pareto_sorted[CENET_KEY],
        color=PARETO_POINT_COLOR,
        s=PARETO_POINT_SIZE_2D,
        marker=PARETO_MARKER,
        alpha=1.0,
        edgecolors=POINT_EDGE_COLOR,
        linewidths=PARETO_EDGE_WIDTH,
        zorder=3,
    )
    ax2.scatter(
        representative_df[COST_KEY],
        representative_df[CENET_KEY],
        color=REPRESENTATIVE_POINT_COLOR,
        s=REPRESENTATIVE_POINT_SIZE_2D,
        marker=REPRESENTATIVE_MARKER,
        alpha=1.0,
        edgecolors=POINT_EDGE_COLOR,
        linewidths=REPRESENTATIVE_EDGE_WIDTH,
        zorder=5,
    )

    x_min, x_max = analysis_data[COST_KEY].min(), analysis_data[COST_KEY].max()
    y_min, y_max = analysis_data[CENET_KEY].min(), analysis_data[CENET_KEY].max()
    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    ax2.set_xlim(x_min - x_pad, x_max + x_pad)
    ax2.set_ylim(y_min - y_pad, y_max + y_pad)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax2.set_xlabel(cost_label, fontsize=18, fontweight="bold")
    ax2.set_ylabel(ce_label, fontsize=18, fontweight="bold")
    apply_2d_style(ax2, grid=True)

    legend = ax2.legend(
        handles=make_base_legend_handles(include_schemes=True),
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=16,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.8,
        labelspacing=0.4,
        borderaxespad=0.0,
        prop={"family": "Times New Roman", "size": 16},
    )
    for text in legend.get_texts():
        text.set_fontweight("normal")
        text.set_fontfamily("Times New Roman")

    axins = ax2.inset_axes([0.525, 0.06, 0.45, 0.48])
    axins.scatter(
        analysis_data[COST_KEY],
        analysis_data[CENET_KEY],
        c=analysis_data[UPTAKE_KEY],
        cmap=CUSTOM_CMAP,
        norm=uptake_norm,
        s=ALL_POINT_SIZE_2D * 0.7,
        marker=ALL_MARKER,
        alpha=1.0,
        edgecolors=POINT_EDGE_COLOR,
        linewidths=POINT_EDGE_WIDTH,
        zorder=1,
    )
    if DRAW_PARETO_PROJECTION_LINE and len(pareto_sorted) >= 2:
        axins.plot(
            pareto_sorted[COST_KEY],
            pareto_sorted[CENET_KEY],
            color=PARETO_POINT_COLOR,
            linestyle="--",
            linewidth=1.6,
            zorder=2,
        )
    axins.scatter(
        pareto_sorted[COST_KEY],
        pareto_sorted[CENET_KEY],
        color=PARETO_POINT_COLOR,
        s=PARETO_POINT_SIZE_2D * 0.8,
        marker=PARETO_MARKER,
        alpha=1.0,
        edgecolors=POINT_EDGE_COLOR,
        linewidths=PARETO_EDGE_WIDTH,
        zorder=3,
    )
    axins.scatter(
        representative_df[COST_KEY],
        representative_df[CENET_KEY],
        color=REPRESENTATIVE_POINT_COLOR,
        s=REPRESENTATIVE_POINT_SIZE_2D * 0.95,
        marker=REPRESENTATIVE_MARKER,
        alpha=1.0,
        edgecolors=POINT_EDGE_COLOR,
        linewidths=REPRESENTATIVE_EDGE_WIDTH,
        zorder=5,
    )

    label_offsets = {
        "U": (10, 0),
        "E": (10,-6),
        "C": (-18, -8),
        "K": (-18, 0),
    }
    for _, row in representative_df.iterrows():
        label = str(row["Scenario_Label"])
        dx, dy = label_offsets.get(label, (4, 10))
        axins.annotate(
            label,
            xy=(row[COST_KEY], row[CENET_KEY]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=16,
            fontweight="bold",
            fontfamily="Times New Roman",
            color="black",
            ha="left",
            va="center",
            zorder=6,
        )

    axins.set_xlim(*INSET_X_RANGE)
    axins.set_ylim(*INSET_Y_RANGE)
    axins.grid(True, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA, color=GRID_COLOR)
    set_bold_ticklabels_2d(axins, 12)
    for spine in axins.spines.values():
        spine.set_linewidth(1.2)

    rect = Rectangle(
        (INSET_X_RANGE[0], INSET_Y_RANGE[0]),
        INSET_X_RANGE[1] - INSET_X_RANGE[0],
        INSET_Y_RANGE[1] - INSET_Y_RANGE[0],
        fill=True,
        facecolor=MIDDLE_COLOR,
        edgecolor="none",
        alpha=0.65,
        zorder=0,
    )
    ax2.add_patch(rect)
    ax2.annotate(
        "",
        xy=(0.525, 0.30),
        xycoords="axes fraction",
        xytext=(INSET_X_RANGE[1], np.mean(INSET_Y_RANGE)),
        textcoords="data",
        arrowprops=dict(
            arrowstyle="->",
            color=GRID_COLOR,
            lw=1.5,
            ls="dashed",
        ),
        zorder=4,
    )

    # Align the colorbar with the 2D axis.
    fig.subplots_adjust(left=0.105, right=0.850, bottom=0.125, top=0.98)

    main_pos = ax2.get_position()
    cbar_width = main_pos.width * PARETO_2D_CBAR_WIDTH_RATIO
    cbar_pad = main_pos.width * PARETO_2D_CBAR_PAD_RATIO

    cax = fig.add_axes(
        [
            main_pos.x1 + cbar_pad,
            main_pos.y0,
            cbar_width,
            main_pos.height,
        ]
    )

    sm = ScalarMappable(norm=uptake_norm, cmap=CUSTOM_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)

    cbar.ax.set_title("")
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.set_ylabel(
        uptake_label,
        fontsize=18,
        fontweight="bold",
        rotation=90,
        labelpad=12,
        fontfamily="Times New Roman",
    )
    cbar.ax.tick_params(labelsize=16, width=1.0, length=5.0)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontfamily("Times New Roman")

    # Use a 1.0-point colorbar outline.
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1.0)

    save_figure(
        fig,
        Path(out_dir) / "pareto_2d_inset_plot",
        tight=True,
        pad_inches=0.01,
    )


# =============================================================================
# 6. ENVIRONMENTAL-IMPACT HEATMAPS
# =============================================================================
def rowwise_maxabs_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    denom = np.nanmax(np.abs(arr), axis=1, keepdims=True)
    denom[~np.isfinite(denom) | np.isclose(denom, 0.0)] = 1.0
    return arr / denom


def rowwise_signed_contribution_percent(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    denom = np.nansum(np.abs(arr), axis=1, keepdims=True)
    denom[~np.isfinite(denom) | np.isclose(denom, 0.0)] = 1.0
    return arr / denom * 100.0


def plot_four_solution_impact_heatmap(
    df_raw: pd.DataFrame,
    representative_df: pd.DataFrame,
    out_dir: str,
) -> None:
    if not configured(IMPACT_TOTAL_EXCEL_COLS):
        print("[SKIP] Fill IMPACT_TOTAL_EXCEL_COLS to generate the 18-impact × 4-solution heatmap.")
        return

    impact_table_all = extract_excel_columns(
        df_raw,
        {k: v for k, v in IMPACT_TOTAL_EXCEL_COLS.items() if v is not None},
    )

    raw_rows = []
    for _, scheme in representative_df.iterrows():
        excel_row = int(scheme["Original_row_in_excel"])
        zero_based_row = excel_row - HEADER_ROW - 2
        if zero_based_row not in impact_table_all.index:
            raise IndexError(f"Representative Excel row {excel_row} is outside the LCA table.")
        row = impact_table_all.loc[zero_based_row].copy()
        row.name = str(scheme["Scenario_Label"])
        raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows).T
    raw_df = raw_df[["U", "E", "C", "K"]]
    raw_df.to_csv(
        os.path.join(out_dir, "four_solution_18impact_raw.csv"),
        encoding="utf-8-sig",
    )

    norm_df = pd.DataFrame(
        rowwise_maxabs_normalize(raw_df.to_numpy()),
        index=raw_df.index,
        columns=raw_df.columns,
    )
    norm_df.to_csv(
        os.path.join(out_dir, "four_solution_18impact_row_normalized.csv"),
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(8.5, 10.0), dpi=FIG_DPI)
    im = ax.imshow(
        norm_df.to_numpy(),
        aspect="auto",
        cmap=CUSTOM_CMAP,
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
    )

    ax.set_xticks(np.arange(norm_df.shape[1]))
    ax.set_xticklabels(norm_df.columns, fontsize=18, fontweight="bold")
    ax.set_yticks(np.arange(norm_df.shape[0]))
    ax.set_yticklabels(norm_df.index, fontsize=13, fontweight="bold")
    ax.set_xlabel("Representative solution", fontsize=18, fontweight="bold")
    ax.set_ylabel("Environmental impact category", fontsize=18, fontweight="bold")

    for i in range(norm_df.shape[0]):
        for j in range(norm_df.shape[1]):
            value = norm_df.iat[i, j]
            raw = raw_df.iat[i, j]
            text_color = "white" if abs(value) >= 0.60 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=text_color,
            )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis="both", length=5, width=1.0)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035, aspect=32)
    cbar.set_label(
        "Row-wise normalized impact",
        fontsize=16,
        fontweight="bold",
        labelpad=12,
    )
    cbar.ax.tick_params(labelsize=14, width=1.0, length=5.0)
    cbar.outline.set_linewidth(1.0)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    fig.subplots_adjust(left=0.38, right=0.90, bottom=0.10, top=0.98)
    save_figure(fig, Path(out_dir) / "four_solution_18impact_heatmap", tight=False)


def plot_k_solution_stage_heatmap(
    df_raw: pd.DataFrame,
    representative_df: pd.DataFrame,
    out_dir: str,
) -> None:
    if not configured_nested(STAGE_IMPACT_EXCEL_COLS):
        print("[SKIP] Fill STAGE_IMPACT_EXCEL_COLS to generate the K-solution 18-impact × 5-stage heatmap.")
        return

    k_row = representative_df.loc[representative_df["Scenario_Label"] == "K"].iloc[0]
    excel_row = int(k_row["Original_row_in_excel"])
    zero_based_row = excel_row - HEADER_ROW - 2

    stage_raw = pd.DataFrame(index=list(IMPACT_TOTAL_EXCEL_COLS.keys()), columns=STAGE_NAMES, dtype=float)
    for stage in STAGE_NAMES:
        for impact in IMPACT_TOTAL_EXCEL_COLS:
            col_letter = STAGE_IMPACT_EXCEL_COLS[stage][impact]
            series = pd.to_numeric(get_series_by_excel_col(df_raw, col_letter), errors="coerce")
            stage_raw.loc[impact, stage] = series.loc[zero_based_row]

    stage_raw.to_csv(
        os.path.join(out_dir, "K_solution_18impact_5stage_raw.csv"),
        encoding="utf-8-sig",
    )

    contribution_df = pd.DataFrame(
        rowwise_signed_contribution_percent(stage_raw.to_numpy()),
        index=stage_raw.index,
        columns=stage_raw.columns,
    )
    contribution_df.to_csv(
        os.path.join(out_dir, "K_solution_18impact_5stage_contribution_percent.csv"),
        encoding="utf-8-sig",
    )

    vmax = max(100.0, float(np.nanmax(np.abs(contribution_df.to_numpy()))))
    fig, ax = plt.subplots(figsize=(12.0, 10.0), dpi=FIG_DPI)
    im = ax.imshow(
        contribution_df.to_numpy(),
        aspect="auto",
        cmap=CUSTOM_CMAP,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
    )

    ax.set_xticks(np.arange(contribution_df.shape[1]))
    ax.set_xticklabels(contribution_df.columns, fontsize=14, fontweight="bold")
    ax.set_yticks(np.arange(contribution_df.shape[0]))
    ax.set_yticklabels(contribution_df.index, fontsize=13, fontweight="bold")
    ax.set_xlabel("Life-cycle stage", fontsize=18, fontweight="bold")
    ax.set_ylabel("Environmental impact category", fontsize=18, fontweight="bold")

    for i in range(contribution_df.shape[0]):
        for j in range(contribution_df.shape[1]):
            value = contribution_df.iat[i, j]
            text_color = "white" if abs(value) >= 55 else "black"
            ax.text(
                j,
                i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color=text_color,
            )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis="both", length=5, width=1.0)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035, aspect=32)
    cbar.set_label(
        "Signed stage contribution (%)",
        fontsize=16,
        fontweight="bold",
        labelpad=12,
    )
    cbar.ax.tick_params(labelsize=14, width=1.0, length=5.0)
    cbar.outline.set_linewidth(1.0)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    fig.subplots_adjust(left=0.32, right=0.91, bottom=0.15, top=0.98)
    save_figure(fig, Path(out_dir) / "K_solution_18impact_5stage_heatmap", tight=False)


# =============================================================================
# 7. SENSITIVITY PLOTS
# =============================================================================
def plot_weight_selection_frequency(
    counts: pd.DataFrame,
    representative_df: pd.DataFrame,
    out_dir: str,
) -> None:
    """
    Plot selection frequencies under different decision-weight combinations.

    Representative samples are labelled directly as U, E, C, or K.
    Other Pareto samples are labelled as Sample + sample number.
    """

    plot_df = counts.head(12).copy()

    # Map Excel row numbers to representative labels.
    representative_map = {
        int(row["Original_row_in_excel"]): str(row["Scenario_Label"])
        for _, row in representative_df.iterrows()
    }

    def make_sample_label(excel_row) -> str:
        sample_number = int(excel_row)

        # Use U, E, C, or K for representative samples.
        if sample_number in representative_map:
            return representative_map[sample_number]

        # Label other samples by row number.
        return f"Sp {sample_number}"

    plot_df["Label"] = plot_df[
        "Selected_Original_Excel_Row"
    ].apply(make_sample_label)

    # Place the highest selection frequency at the top.
    plot_df = plot_df.sort_values(
        "Selection_frequency_%",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 6.5),
        dpi=FIG_DPI,
    )

    bars = ax.barh(
        plot_df["Label"],
        plot_df["Selection_frequency_%"],
        color="#5B84B1",
        edgecolor="black",
        linewidth=1.0,
        height=0.68,
    )

    # Rank samples by selection frequency.
    frequency_rank = (
        plot_df["Selection_frequency_%"]
        .rank(method="first", ascending=False)
        .astype(int)
        .to_numpy()
    )

    # Add percentage labels.
    for bar, value, rank in zip(
            bars,
            plot_df["Selection_frequency_%"],
            frequency_rank,
    ):
        y_center = bar.get_y() + bar.get_height() / 2

        if rank <= 3:
            ax.annotate(
                f"{value:.2f}%",
                xy=(value, y_center),
                xytext=(-5, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=14,
                fontweight="bold",
                fontfamily="Times New Roman",
                color="white",
                clip_on=False,
                zorder=5,
            )
        else:
            ax.annotate(
                f"{value:.2f}%",
                xy=(value, y_center),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=14,
                fontweight="bold",
                fontfamily="Times New Roman",
                color="black",
                clip_on=False,
                zorder=5,
            )

    ax.set_xlabel(
        "Selection frequency (%)",
        fontsize=18,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Selected Pareto solution",
        fontsize=18,
        fontweight="bold",
    )
    max_frequency = plot_df["Selection_frequency_%"].max()

    ax.set_xlim(
        0,
        max_frequency * 1.16,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    apply_2d_style(ax, grid=True)

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
        color=GRID_COLOR,
    )
    ax.grid(axis="y", visible=False)

    for tick in ax.get_yticklabels():
        tick.set_fontsize(16)
        tick.set_fontweight("bold")
        tick.set_fontfamily("Times New Roman")

    fig.subplots_adjust(
        left=0.25,
        right=0.96,
        bottom=0.16,
        top=0.96,
    )

    save_figure(
        fig,
        Path(out_dir) / "pareto_weight_selection_frequency",
        tight=False,
    )


def build_k_climate_sensitivity(
    df_raw: pd.DataFrame,
    representative_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    if not configured_nested(STAGE_IMPACT_EXCEL_COLS):
        return None
    if CLIMATE_IMPACT_NAME not in IMPACT_TOTAL_EXCEL_COLS:
        raise KeyError(f"CLIMATE_IMPACT_NAME={CLIMATE_IMPACT_NAME!r} is not in IMPACT_TOTAL_EXCEL_COLS.")

    k_row = representative_df.loc[representative_df["Scenario_Label"] == "K"].iloc[0]
    excel_row = int(k_row["Original_row_in_excel"])
    zero_based_row = excel_row - HEADER_ROW - 2
    baseline_cenet = float(k_row[CENET_KEY])

    records = []
    p = SENSITIVITY_FRACTION

    for stage in STAGE_NAMES:
        col_letter = STAGE_IMPACT_EXCEL_COLS[stage][CLIMATE_IMPACT_NAME]
        stage_series = pd.to_numeric(get_series_by_excel_col(df_raw, col_letter), errors="coerce")
        stage_value = float(stage_series.loc[zero_based_row])

        low = baseline_cenet - p * stage_value
        high = baseline_cenet + p * stage_value
        records.append(
            {
                "Parameter": stage.replace("\n", " "),
                "Low": low,
                "High": high,
                "Baseline": baseline_cenet,
            }
        )

    if INCLUDE_UPTAKE_IN_CLIMATE_SENSITIVITY:
        uptake = float(k_row[UPTAKE_KEY])
        # CE_net = process burden - uptake. Therefore higher uptake lowers CE_net.
        low_uptake = baseline_cenet + p * uptake   # uptake -20%
        high_uptake = baseline_cenet - p * uptake # uptake +20%
        records.append(
            {
                "Parameter": "CO$_2$ uptake",
                "Low": low_uptake,
                "High": high_uptake,
                "Baseline": baseline_cenet,
            }
        )

    return pd.DataFrame(records)


def build_k_cost_sensitivity(
    df_raw: pd.DataFrame,
    representative_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    if not configured(STAGE_COST_EXCEL_COLS):
        return None

    k_row = representative_df.loc[representative_df["Scenario_Label"] == "K"].iloc[0]
    excel_row = int(k_row["Original_row_in_excel"])
    zero_based_row = excel_row - HEADER_ROW - 2
    baseline_cost = float(k_row[COST_KEY])
    p = SENSITIVITY_FRACTION

    records = []
    for stage, col_letter in STAGE_COST_EXCEL_COLS.items():
        stage_series = pd.to_numeric(get_series_by_excel_col(df_raw, col_letter), errors="coerce")
        stage_cost = float(stage_series.loc[zero_based_row])
        records.append(
            {
                "Parameter": stage.replace("\n", " "),
                "Low": baseline_cost - p * stage_cost,
                "High": baseline_cost + p * stage_cost,
                "Baseline": baseline_cost,
            }
        )
    return pd.DataFrame(records)


def plot_tornado(
    sensitivity_df: pd.DataFrame,
    xlabel: str,
    save_stem: str | Path,
) -> None:
    df = sensitivity_df.copy()
    df["Range"] = (df["High"] - df["Low"]).abs()
    df = df.sort_values("Range", ascending=True).reset_index(drop=True)

    y = np.arange(len(df))
    baseline = float(df["Baseline"].iloc[0])
    left = np.minimum(df["Low"].to_numpy(), df["High"].to_numpy())
    right = np.maximum(df["Low"].to_numpy(), df["High"].to_numpy())

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=FIG_DPI)
    ax.barh(
        y,
        baseline - left,
        left=left,
        edgecolor="black",
        linewidth=1.0,
        label=f"Parameter −{int(SENSITIVITY_FRACTION * 100)}%",
    )
    ax.barh(
        y,
        right - baseline,
        left=baseline,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.65,
        label=f"Parameter +{int(SENSITIVITY_FRACTION * 100)}%",
    )

    ax.axvline(baseline, color="black", linewidth=1.5, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df["Parameter"], fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=18, fontweight="bold")
    ax.set_ylabel("Sensitivity parameter", fontsize=18, fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    apply_2d_style(ax, grid=True)

    legend = ax.legend(
        loc="best",
        fontsize=14,
        frameon=True,
        facecolor="white",
        edgecolor="#BBBBBB",
        framealpha=0.95,
    )
    for text in legend.get_texts():
        text.set_fontfamily("Times New Roman")

    fig.subplots_adjust(left=0.36, right=0.96, bottom=0.16, top=0.96)
    save_figure(fig, save_stem, tight=False)


# =============================================================================
# 8. DATA PREPARATION AND MAIN
# =============================================================================
def prepare_analysis_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    if CO2_UPTAKE_EXCEL_COL in (None, ""):
        raise ValueError(
            "Please fill CO2_UPTAKE_EXCEL_COL at the top of the script "
            "with the Excel column containing CO2 sequestration."
        )

    feature_df = build_standardized_feature_dataframe(df_raw)
    analysis = pd.DataFrame(index=df_raw.index)
    analysis["Original_row_in_excel"] = df_raw.index + HEADER_ROW + 2

    for col in feature_df.columns:
        analysis[col] = feature_df[col]

    analysis[COST_KEY] = pd.to_numeric(get_series_by_excel_col(df_raw, COST_EXCEL_COL), errors="coerce")
    analysis[CENET_KEY] = pd.to_numeric(get_series_by_excel_col(df_raw, CENET_EXCEL_COL), errors="coerce")
    analysis[UPTAKE_KEY] = pd.to_numeric(get_series_by_excel_col(df_raw, CO2_UPTAKE_EXCEL_COL), errors="coerce")
    analysis[WS_KEY] = pd.to_numeric(get_series_by_excel_col(df_raw, WS_EXCEL_COL), errors="coerce")
    analysis[TIME_KEY] = pd.to_numeric(get_series_by_excel_col(df_raw, TIME_EXCEL_COL), errors="coerce")

    analysis = analysis.dropna(
        subset=[COST_KEY, CENET_KEY, UPTAKE_KEY, WS_KEY, TIME_KEY]
    ).copy()
    analysis = analysis[(analysis[TIME_KEY] > 0) & ((analysis[WS_KEY]) > 1)].copy()

    if analysis.empty:
        raise ValueError("No valid rows remain after loading Pareto objectives.")
    return analysis


def main() -> None:
    ensure_dir(OUT_DIR)
    df_raw = clean_columns(
        pd.read_excel(
            DATASET_FILE,
            sheet_name=SHEET_NAME,
            header=HEADER_ROW,
        )
    )

    analysis_data = prepare_analysis_data(df_raw)
    analysis_data["Pareto_3D"] = pareto_frontier_3d(
        analysis_data,
        COST_KEY,
        CENET_KEY,
        UPTAKE_KEY,
        minimize_x=True,
        minimize_y=True,
        maximize_z=True,
    )
    pareto_data = analysis_data.loc[analysis_data["Pareto_3D"]].copy()
    if pareto_data.empty:
        raise ValueError("No Pareto solution was found.")

    representative_df = select_four_representative_solutions(
        pareto_data,
        COST_KEY,
        CENET_KEY,
        UPTAKE_KEY,
        weights=(1.0, 1.0, 1.0),
    )

    # Tables
    analysis_data.to_csv(
        os.path.join(OUT_DIR, "all_samples_with_pareto_flag.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pareto_data.sort_values([COST_KEY, CENET_KEY, UPTAKE_KEY]).to_csv(
        os.path.join(OUT_DIR, "pareto_samples_3d.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    representative_df.to_csv(
        os.path.join(OUT_DIR, "representative_solutions_U_E_C_K.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Pareto plots
    plot_pareto_3d(analysis_data, pareto_data, representative_df, OUT_DIR)
    plot_pareto_2d_projection(analysis_data, pareto_data, representative_df, OUT_DIR)

    # Four-solution and K-stage environmental plots
    plot_four_solution_impact_heatmap(df_raw, representative_df, OUT_DIR)
    plot_k_solution_stage_heatmap(df_raw, representative_df, OUT_DIR)

    # Decision-weight sensitivity
    run_weight_grid_sensitivity(
        pareto_data.reset_index(drop=True),
        COST_KEY,
        CENET_KEY,
        UPTAKE_KEY,
        representative_df,
        OUT_DIR,
    )

    # K-solution stage/inventory sensitivity
    climate_sensitivity = build_k_climate_sensitivity(df_raw, representative_df)
    if climate_sensitivity is not None:
        climate_sensitivity.to_csv(
            os.path.join(OUT_DIR, "K_solution_climate_sensitivity.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        plot_tornado(
            climate_sensitivity,
            get_label(CENET_DISPLAY, CENET_UNIT),
            Path(OUT_DIR) / "K_solution_climate_sensitivity_tornado",
        )
    else:
        print("[SKIP] Climate tornado plot requires STAGE_IMPACT_EXCEL_COLS.")

    cost_sensitivity = build_k_cost_sensitivity(df_raw, representative_df)
    if cost_sensitivity is not None:
        cost_sensitivity.to_csv(
            os.path.join(OUT_DIR, "K_solution_cost_sensitivity.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        plot_tornado(
            cost_sensitivity,
            get_label(COST_DISPLAY, COST_UNIT),
            Path(OUT_DIR) / "K_solution_cost_sensitivity_tornado",
        )
    else:
        print("[SKIP] Cost tornado plot requires STAGE_COST_EXCEL_COLS.")

    print("All configured analyses are complete.")
    print("Output directory: outputs/Pareto_LCA_integrated")


if __name__ == "__main__":
    main()

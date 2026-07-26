# -*- coding: utf-8 -*-

# ============================================================================
# Module: Imports and dependencies
# ============================================================================
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Polygon
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from figure_output1 import (
    FIG_DPI,
    BLUE,
    RED,
    MID_GREY,
    RDBU_CMAP,
    ShapRankingBundle,
    apply_academic_style,
    PANEL_GRID_FIGSIZE,
    PANEL_GRID_ADJUST,
    PANEL_2D_FIGSIZE,
    PANEL_2D_ADJUST,
    PANEL_LEGEND_FONTSIZE,
    PANEL_MARKER_LABEL_FONTSIZE,
    apply_panel_axis_format,
    style_panel_legend,
    style_panel_colorbar,
    add_panel_labels,
    quantile_edges,
    format_feature_label,
    get_feature_axis_label,
    save_figure,
    to_dataframe,
    shade_promoting_inhibiting_regions,
    add_crossing_label,
    GLOBAL_FONT_FAMILY,
    save_plot_source_bundle,
)

# ============================================================================
# Figure 8: SHAP heatmap
# ============================================================================
def plot_figure8_shap_heatmap(shap_values: Any, X: pd.DataFrame, model: Any, save_stem: Union[str, Path]) -> None:
    shap_arr = shap_values
    X = to_dataframe(X)
    preds = np.asarray(model.predict(X), dtype=float).ravel()
    order = np.argsort(preds)

    features = list(X.columns)
    mean_abs = np.mean(np.abs(shap_arr), axis=0)
    feature_order = np.argsort(mean_abs)

    heat = shap_arr[order, :][:, feature_order].T
    ordered_feature_names = [
        features[i] for i in feature_order
    ]

    n_features = len(ordered_feature_names)
    n_samples = len(order)

    ordered_feature_values = (
        X.iloc[order, feature_order]
        .to_numpy(dtype=float)
        .T
    )

    heatmap_long_df = pd.DataFrame({
        "Feature": np.repeat(
            ordered_feature_names,
            n_samples,
        ),
        "Feature_plot_order": np.repeat(
            np.arange(1, n_features + 1),
            n_samples,
        ),
        "Sample_plot_order": np.tile(
            np.arange(1, n_samples + 1),
            n_features,
        ),
        "Original_sample_ID": np.tile(
            order + 1,
            n_features,
        ),
        "Prediction": np.tile(
            preds[order],
            n_features,
        ),
        "Feature_value":
            ordered_feature_values.reshape(-1),
        "SHAP_value": heat.reshape(-1),
    })

    prediction_order_df = pd.DataFrame({
        "Sample_plot_order":
            np.arange(1, n_samples + 1),
        "Original_sample_ID": order + 1,
        "Prediction": preds[order],
    })

    vmax = np.nanpercentile(np.abs(heat), 99)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    save_plot_source_bundle(
        save_stem=save_stem,
        figure_tag="Figure8",
        tables={
            "Heatmap_values": heatmap_long_df,
            "Prediction_order": prediction_order_df,
        },
        cache={
            "sample_order": order,
            "feature_order": feature_order,
            "feature_names":
                ordered_feature_names,
            "heat_matrix": heat,
            "predictions": preds,
            "color_limit": vmax,
        },
    )
    formatted_features = [format_feature_label(features[i]) for i in feature_order]

    fig = plt.figure(figsize=(8.5, 6.5), dpi=FIG_DPI)

    axis_label_fontsize = 18
    tick_fontsize = 16
    cbar_label_fontsize = 18

    left = 0.25
    heat_bottom = 0.18
    heat_width = 0.58
    heat_height = 0.52

    top_gap = 0.02
    top_height = 0.16

    cbar_left = 0.845
    cbar_width = 0.02
    cbar_bottom = heat_bottom
    cbar_height = heat_height + top_gap + top_height

    ax_heat = fig.add_axes([left, heat_bottom, heat_width, heat_height])
    ax_top = fig.add_axes([left, heat_bottom + heat_height + top_gap, heat_width, top_height], sharex=ax_heat)
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

    ax_top.plot(np.arange(len(order)), preds[order], color=BLUE, linewidth=1.5)
    ax_top.set_ylabel(r"$f(x)$", fontsize=axis_label_fontsize, fontweight="bold", rotation=0, labelpad=18, va="center", y=0.42 )
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)

    y_min, y_max = preds[order].min(), preds[order].max()
    y_pad = (y_max - y_min) * 0.15 if y_max > y_min else 1.0
    ax_top.set_ylim(y_min - 0.1 * y_pad, y_max + y_pad)

    apply_academic_style(ax_top, grid=True)
    ax_top.tick_params(axis="y", length=5.0, width=1.0, labelsize=tick_fontsize)
    for lbl in ax_top.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_family(GLOBAL_FONT_FAMILY)


    im = ax_heat.imshow(
        heat,
        aspect="auto",
        cmap=RDBU_CMAP,
        vmin=-vmax,
        vmax=vmax,
        origin="lower"
    )

    ax_heat.set_yticks(np.arange(len(formatted_features)))
    ax_heat.set_yticklabels(formatted_features, fontsize=tick_fontsize, fontweight="bold")
    ax_heat.set_xlabel("Samples", fontsize=axis_label_fontsize, fontweight="bold")

    apply_academic_style(ax_heat, grid=False)
    ax_heat.tick_params(axis="x", length=5.0, width=1.0, labelsize=tick_fontsize)
    ax_heat.tick_params(axis="y", length=5.0, width=1.0, labelsize=tick_fontsize)

    for lbl in ax_heat.get_xticklabels() + ax_heat.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_family(GLOBAL_FONT_FAMILY)


    for ax in [ax_top, ax_heat]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("SHAP value", fontsize=cbar_label_fontsize, fontweight="bold")
    cbar.ax.tick_params(length=5.0, width=1.0, labelsize=tick_fontsize)
    cbar.outline.set_linewidth(1.0)

    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_family(GLOBAL_FONT_FAMILY)

    save_figure(fig, save_stem)


# ============================================================================
# Figure 9 helpers: Representative samples and waterfall panels
# ============================================================================
def select_quantile_representative_indices(values: np.ndarray, quantiles: Sequence[float]) -> List[int]:
    values = np.asarray(values, dtype=float).ravel()
    selected = []
    for q in quantiles:
        target = np.quantile(values, q)
        idx = int(np.argmin(np.abs(values - target)))
        selected.append(idx)
    out, seen = [], set()
    for idx in selected:
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    while len(out) < len(quantiles):
        for idx in np.argsort(values):
            if int(idx) not in seen:
                out.append(int(idx))
                seen.add(int(idx))
                break
    return out[: len(quantiles)]

def waterfall_single_sample(
        ax: plt.Axes,
        shap_row: np.ndarray,
        feature_values: pd.Series,
        feature_names: Sequence[str],
        expected_value: float,
        prediction_value: float,
        top_n: int = 8,
        label_sides: Optional[Sequence[str]] = None,
        panel_tag: Optional[str] = None,
        pred_loc: str = "bottom_left",
) -> None:
    """Draw one Figure 9 waterfall panel using the shared academic axis style."""

    tick_fontsize = 20
    inner_fontsize = 20
    colored_label_fontsize = 18
    colored_number_fontsize = 18
    legend_fontsize = 20

    shap_row = np.asarray(shap_row, dtype=float).ravel()
    vals_all = np.asarray(feature_values, dtype=object)

    order = np.argsort(np.abs(shap_row))[::-1][:top_n]
    contrib = shap_row[order]
    names = [feature_names[i] for i in order]
    vals = vals_all[order]
    plot_items = list(zip(names, vals, contrib))

    n_items = len(plot_items)

    row_gap = 2.15
    y_pos = np.arange(n_items) * row_gap
    y_pos = y_pos[::-1]

    cumulative = expected_value + np.cumsum([x[2] for x in plot_items])
    x_candidates = np.concatenate([[expected_value, prediction_value], cumulative])

    x_min = np.min(x_candidates)
    x_max = np.max(x_candidates)
    x_span = max(x_max - x_min, 1e-6)


    if panel_tag in ["(c)", "(d)"]:
        x_pad_left = max(x_span * 0.55, 20)
        x_pad_right = max(x_span * 0.15, 10)
    else:
        x_pad_left = max(x_span * 0.15, 10)
        x_pad_right = max(x_span * 0.55, 20)

    ax.set_xlim(x_min - x_pad_left, x_max + x_pad_right)
    ax.set_ylim(-1.4, y_pos.max() + 1.8)

    xrange_ = ax.get_xlim()[1] - ax.get_xlim()[0]


    if panel_tag in ["(c)", "(d)"]:
        ax.axvspan(
            ax.get_xlim()[0],
            expected_value,
            facecolor="#FFE4E1",
            alpha=0.30,
            zorder=-1
        )
        label_x = ax.get_xlim()[0] + 0.02 * xrange_
        label_ha = "left"
    else:
        ax.axvspan(
            expected_value,
            ax.get_xlim()[1],
            facecolor="#FFE4E1",
            alpha=0.30,
            zorder=-1
        )
        label_x = ax.get_xlim()[1] - 0.02 * xrange_
        label_ha = "right"

    running = expected_value
    bar_height = 1.72
    text_offset = xrange_ * 0.015

    white_bbox = dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="none",
        alpha=0.90
    )


    for yi, name, val, sv in zip(y_pos, names, vals, contrib):
        start = running
        end = running + sv
        width = abs(sv)
        color = RED if sv >= 0 else BLUE


        ax.plot(
            [start, start],
            [yi, y_pos.max() + 1.4 if yi == y_pos[0] else yi + row_gap],
            color=MID_GREY,
            linestyle="--",
            linewidth=1.0,
            zorder=1
        )

        hl = min(xrange_ * 0.035, width)
        hw = bar_height / 2

        if sv >= 0:
            rect_width = max(sv - hl, 0)
            ax.barh(
                yi,
                rect_width,
                left=start,
                height=bar_height,
                color=color,
                edgecolor="none",
                alpha=1.0,
                zorder=2
            )
            pts = [
                (end - hl, yi - hw),
                (end, yi),
                (end - hl, yi + hw),
            ]
            val_x = end + text_offset
            val_ha = "left"
        else:
            rect_width = max(abs(sv) - hl, 0)
            ax.barh(
                yi,
                rect_width,
                left=end + hl,
                height=bar_height,
                color=color,
                edgecolor="none",
                alpha=1.0,
                zorder=2
            )
            pts = [
                (end + hl, yi - hw),
                (end, yi),
                (end + hl, yi + hw),
            ]
            val_x = end - text_offset
            val_ha = "right"

        ax.add_patch(
            Polygon(
                pts,
                closed=True,
                facecolor=color,
                edgecolor="none",
                alpha=1.0,
                zorder=2
            )
        )


        label_text = f"{format_feature_label(name)}\n({val:.3g})"
        ax.text(
            label_x,
            yi,
            label_text,
            ha=label_ha,
            va="center",
            fontsize=colored_label_fontsize,
            color=color,
            fontweight="bold",
            family=GLOBAL_FONT_FAMILY,
            linespacing=1.2,
            zorder=20,
            clip_on=False
        )
        val_color = color
        val_bbox = white_bbox


        if panel_tag == "(c)" and yi == y_pos[0] and sv < 0:
            val_x = end + max(0.020 * xrange_, 0.1 * width)
            val_ha = "left"
            val_color = "white"
            val_bbox = None


        ax.text(
            val_x,
            yi,
            f"{sv:+.2f}",
            ha=val_ha,
            va="center",
            color=val_color,
            fontsize=colored_number_fontsize,
            fontweight="bold",
            family=GLOBAL_FONT_FAMILY,
            zorder=21,
            clip_on=False,
            bbox=val_bbox
        )

        running = end


    ax.plot(
        [running, running],
        [y_pos[-1], -1.4],
        color=MID_GREY,
        linestyle="--",
        linewidth=1.0,
        zorder=1
    )


    ax.axvline(
        prediction_value,
        color="black",
        linewidth=1.5,
        zorder=3
    )
    ax.axvline(
        expected_value,
        color="black",
        linestyle="--",
        linewidth=1.5,
        zorder=3
    )


    arrow_y = y_pos.max() + 1.45
    arrow_len = xrange_ * 0.015
    arrow_half_width = 0.25

    if prediction_value > expected_value:
        ax.plot(
            [expected_value, prediction_value - arrow_len],
            [arrow_y, arrow_y],
            color=MID_GREY,
            linestyle="--",
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(
            Polygon(
                [
                    (prediction_value - arrow_len, arrow_y - arrow_half_width),
                    (prediction_value, arrow_y),
                    (prediction_value - arrow_len, arrow_y + arrow_half_width),
                ],
                closed=True,
                facecolor="black",
                edgecolor="black",
                zorder=3
            )
        )
    elif prediction_value < expected_value:
        ax.plot(
            [expected_value, prediction_value + arrow_len],
            [arrow_y, arrow_y],
            color=MID_GREY,
            linestyle="--",
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(
            Polygon(
                [
                    (prediction_value + arrow_len, arrow_y - arrow_half_width),
                    (prediction_value, arrow_y),
                    (prediction_value + arrow_len, arrow_y + arrow_half_width),
                ],
                closed=True,
                facecolor="black",
                edgecolor="black",
                zorder=3
            )
        )


    trans = ax.get_xaxis_transform()

    if abs(expected_value - prediction_value) < x_span * 0.18:
        base_y, fx_y = 1.02, 1.08
    else:
        base_y, fx_y = 1.02, 1.02


    base_text_x = expected_value
    fx_text_x = prediction_value


    if panel_tag == "(c)":
        base_text_x = expected_value - 0.065 * xrange_
        fx_text_x = prediction_value + 0.025 * xrange_

    ax.text(
        base_text_x,
        base_y,
        f"Base = {expected_value:.2f}",
        transform=trans,
        ha="center",
        va="bottom",
        fontsize=inner_fontsize,
        fontweight="bold",
        family=GLOBAL_FONT_FAMILY,
        zorder=20,
        clip_on=False
    )

    ax.text(
        fx_text_x,
        fx_y,
        f"f(x) = {prediction_value:.2f}",
        transform=trans,
        ha="center",
        va="bottom",
        fontsize=inner_fontsize,
        fontweight="bold",
        family=GLOBAL_FONT_FAMILY,
        zorder=20,
        clip_on=False
    )


    if panel_tag is not None:
        ax.text(
            -0.01,
            1.01,
            panel_tag,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=inner_fontsize,
            fontweight="bold",
            family=GLOBAL_FONT_FAMILY,
            zorder=20,
            clip_on=False
        )


    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=RED,
            marker=">",
            markersize=10,
            linestyle="None",
            label="higher"
        ),
        mlines.Line2D(
            [],
            [],
            color=BLUE,
            marker="<",
            markersize=10,
            linestyle="None",
            label="lower"
        ),
    ]

    x_min_limit, x_max_limit = ax.get_xlim()
    rel_base_x = (expected_value - x_min_limit) / (x_max_limit - x_min_limit + 1e-12)

    bottom_y_offset = 0.025

    if panel_tag == "(a)":
        legend_loc = "lower right"
        legend_bbox = (rel_base_x - 0.010, bottom_y_offset)

    elif panel_tag == "(b)":

        legend_loc = "lower left"
        legend_bbox = (rel_base_x + 0.010, bottom_y_offset)

    elif panel_tag == "(c)":

        legend_loc = "lower right"
        legend_bbox = (rel_base_x - 0.010, bottom_y_offset)

    elif panel_tag == "(d)":
        legend_loc = "lower left"
        legend_bbox = (rel_base_x + 0.010, bottom_y_offset)

    else:
        legend_loc = "lower left"
        legend_bbox = (0.005, bottom_y_offset)

    legend = ax.legend(
        handles=legend_handles,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        fontsize=legend_fontsize,
        frameon=True,
        facecolor="white",
        framealpha=0.90,
        edgecolor="none",
        handletextpad=0.5,
        borderaxespad=0.0
    )

    for txt in legend.get_texts():
        txt.set_fontweight("normal")
        txt.set_family(GLOBAL_FONT_FAMILY)


    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(
        axis="both",
        labelsize=tick_fontsize,
        width=1.0,
        length=5.0
    )

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_family(GLOBAL_FONT_FAMILY)

    apply_academic_style(ax, grid=True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


# ============================================================================
# Figure 9: Typical force waterfall plots
# ============================================================================
def plot_figure9_typical_force_waterfalls(
        shap_values: Any,
        X_data: pd.DataFrame,
        model: Any,
        save_stem: Union[str, Path],
        expected_value: Optional[float] = None
) -> None:
    """Draw Figure 9 using the shared academic axis style."""

    shap_arr = shap_values
    X_data = to_dataframe(X_data)
    preds = np.asarray(model.predict(X_data), dtype=float).ravel()

    if expected_value is None:
        expected_value = float(np.mean(preds - shap_arr.sum(axis=1)))

    indices = select_quantile_representative_indices(
        preds,
        [0.125, 0.375, 0.625, 0.875]
    )
    waterfall_rows = []

    panel_tags_for_export = ["a", "b", "c", "d"]
    quantiles_for_export = [
        0.125, 0.375, 0.625, 0.875
    ]

    for panel_i, sample_idx in enumerate(indices):
        abs_order = np.argsort(
            np.abs(shap_arr[sample_idx])
        )[::-1]

        rank_lookup = {
            int(feature_idx): rank
            for rank, feature_idx in enumerate(
                abs_order,
                start=1,
            )
        }

        for feature_idx, feature_name in enumerate(
                X_data.columns
        ):
            shap_value = float(
                shap_arr[sample_idx, feature_idx]
            )

            waterfall_rows.append({
                "Panel":
                    panel_tags_for_export[panel_i],
                "Selection_quantile":
                    quantiles_for_export[panel_i],
                "Sample_ID": sample_idx + 1,
                "Prediction":
                    float(preds[sample_idx]),
                "Expected_value":
                    float(expected_value),
                "Feature": feature_name,
                "Feature_value":
                    X_data.iloc[
                        sample_idx,
                        feature_idx
                    ],
                "SHAP_value": shap_value,
                "Absolute_SHAP": abs(shap_value),
                "Absolute_SHAP_rank":
                    rank_lookup[feature_idx],
                "Displayed_in_top8":
                    rank_lookup[feature_idx] <= 8,
            })

    waterfall_source_df = pd.DataFrame(
        waterfall_rows
    )

    save_plot_source_bundle(
        save_stem=save_stem,
        figure_tag="Figure9",
        tables={
            "Waterfall_contributions":
                waterfall_source_df,
        },
        cache={
            "selected_sample_indices": indices,
            "expected_value":
                float(expected_value),
            "predictions": preds[indices],
            "X_selected":
                X_data.iloc[indices].copy(),
            "SHAP_selected":
                shap_arr[indices].copy(),
            "selection_quantiles":
                quantiles_for_export,
            "top_n": 8,
        },
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18.5, 13.5),
        dpi=FIG_DPI
    )
    axes = axes.ravel()

    feature_names = list(X_data.columns)

    per_panel_label_sides = [
        ["left", "right", "right", "right", "right", "right", "right", "right"],
        ["right", "right", "left", "left", "right", "left", "right", "left"],
        ["left", "left", "left", "left", "left", "left", "left", "left"],
        ["right", "left", "left", "left", "left", "left", "left", "left"],
    ]

    panel_tags = ["(a)", "(b)", "(c)", "(d)"]
    pred_locs = ["bottom_right", "bottom_right", "bottom_left", "bottom_left"]

    for panel_i, (ax, idx) in enumerate(zip(axes, indices)):
        waterfall_single_sample(
            ax=ax,
            shap_row=shap_arr[idx],
            feature_values=X_data.iloc[idx],
            feature_names=feature_names,
            expected_value=float(expected_value),
            prediction_value=float(preds[idx]),
            top_n=8,
            label_sides=per_panel_label_sides[panel_i],
            panel_tag=panel_tags[panel_i],
            pred_loc=pred_locs[panel_i],
        )

    fig.subplots_adjust(
        left=0.055,
        right=0.975,
        bottom=0.055,
        top=0.950,
        hspace=0.22,
        wspace=0.095
    )

    save_figure(fig, save_stem)


# ============================================================================
# Dependence analysis helpers: ALE computation
# ============================================================================
def compute_1d_ale_bootstrap(
        model: Any,
        X: pd.DataFrame,
        feature: str,
        bins: int = 20,
        n_bootstraps: int = 30,
        random_state: int = 42
) -> Dict[str, np.ndarray]:
    """Compute 1D ALE main effects with bootstrap 95% confidence intervals."""
    X = X.copy()
    x = X[feature].to_numpy(dtype=float)
    edges = quantile_edges(x, bins=bins)
    n_intervals = len(edges) - 1
    centers = (edges[:-1] + edges[1:]) / 2.0

    def _calculate_ale_single_pass(X_samp: pd.DataFrame) -> np.ndarray:
        x_s = X_samp[feature].to_numpy(dtype=float)
        effects = np.zeros(n_intervals, dtype=float)
        counts = np.zeros(n_intervals, dtype=int)

        for k in range(n_intervals):
            left, right = edges[k], edges[k + 1]
            if k == n_intervals - 1:
                mask = (x_s >= left) & (x_s <= right)
            else:
                mask = (x_s >= left) & (x_s < right)

            counts[k] = int(mask.sum())
            if counts[k] == 0:
                continue

            X_low = X_samp.loc[mask].copy()
            X_high = X_samp.loc[mask].copy()
            X_low[feature] = left
            X_high[feature] = right

            diff = np.asarray(model.predict(X_high) - model.predict(X_low), dtype=float).ravel()
            effects[k] = diff.mean()

        ale_values = np.cumsum(effects)
        weights = counts / max(counts.sum(), 1)
        ale_values_centered = ale_values - np.sum(ale_values * weights)
        return ale_values_centered


    main_ale = _calculate_ale_single_pass(X)


    rng = np.random.default_rng(random_state)
    n_samples = len(X)
    boot_ales = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X.iloc[idx].copy()
        boot_ales.append(_calculate_ale_single_pass(X_boot))

    boot_ales = np.array(boot_ales)


    ale_lower = np.nanpercentile(boot_ales, 2.5, axis=0)
    ale_upper = np.nanpercentile(boot_ales, 97.5, axis=0)


    mask_nan = np.isnan(ale_lower) | np.isnan(ale_upper)
    if mask_nan.any():
        ale_lower[mask_nan] = main_ale[mask_nan]
        ale_upper[mask_nan] = main_ale[mask_nan]

    return {
        "feature": feature,
        "edges": edges,
        "centers": centers,
        "ale": main_ale,
        "ale_lower": ale_lower,
        "ale_upper": ale_upper,
    }

def compute_2d_ale(model: Any, X: pd.DataFrame, feature_pair: Tuple[str, str], bins: int = 12) -> Dict[str, np.ndarray]:
    f1, f2 = feature_pair
    X = X.copy()
    x1 = X[f1].to_numpy(dtype=float)
    x2 = X[f2].to_numpy(dtype=float)

    edges1 = quantile_edges(x1, bins=bins)
    edges2 = quantile_edges(x2, bins=bins)

    n1 = len(edges1) - 1
    n2 = len(edges2) - 1

    delta = np.zeros((n2, n1), dtype=float)
    counts = np.zeros((n2, n1), dtype=int)

    for i in range(n1):
        l1, r1 = edges1[i], edges1[i + 1]
        if i == n1 - 1:
            mask1 = (x1 >= l1) & (x1 <= r1)
        else:
            mask1 = (x1 >= l1) & (x1 < r1)

        for j in range(n2):
            l2, r2 = edges2[j], edges2[j + 1]
            if j == n2 - 1:
                mask2 = (x2 >= l2) & (x2 <= r2)
            else:
                mask2 = (x2 >= l2) & (x2 < r2)

            mask = mask1 & mask2
            counts[j, i] = int(mask.sum())
            if counts[j, i] == 0:
                continue

            X00 = X.loc[mask].copy()
            X01 = X.loc[mask].copy()
            X10 = X.loc[mask].copy()
            X11 = X.loc[mask].copy()

            X00[f1], X00[f2] = l1, l2
            X01[f1], X01[f2] = l1, r2
            X10[f1], X10[f2] = r1, l2
            X11[f1], X11[f2] = r1, r2

            pred00 = np.asarray(model.predict(X00), dtype=float).ravel()
            pred01 = np.asarray(model.predict(X01), dtype=float).ravel()
            pred10 = np.asarray(model.predict(X10), dtype=float).ravel()
            pred11 = np.asarray(model.predict(X11), dtype=float).ravel()

            second_order = pred11 - pred10 - pred01 + pred00
            delta[j, i] = float(second_order.mean())

    ale2 = delta.cumsum(axis=0).cumsum(axis=1)
    if counts.sum() > 0:
        weighted_mean = np.sum(ale2 * counts) / np.sum(counts)
        ale2 = ale2 - weighted_mean

    centers1 = (edges1[:-1] + edges1[1:]) / 2.0
    centers2 = (edges2[:-1] + edges2[1:]) / 2.0
    Xg, Yg = np.meshgrid(centers1, centers2)

    return {
        "feature_pair": feature_pair,
        "edges1": edges1,
        "edges2": edges2,
        "centers1": centers1,
        "centers2": centers2,
        "Xg": Xg,
        "Yg": Yg,
        "ale": ale2,
        "counts": counts,
    }


# ============================================================================
# Figure 10: ICE and PDP grid
# ============================================================================
def plot_figure10_ice_pdp_grid(
        model: Any,
        X_ref: pd.DataFrame,
        ranking_bundle: ShapRankingBundle,
        save_stem: Union[str, Path],
        subsample_ice: int = 120,
        grid_resolution: int = 60,
) -> None:
    X_ref = to_dataframe(X_ref)
    features = ranking_bundle.main_effect_df["feature"].tolist()[:11]

    # ------------------------------------------------------------
    # ICE sampling
    # ------------------------------------------------------------
    rng = np.random.default_rng(26)

    if len(X_ref) > subsample_ice:
        ice_idx = np.sort(
            rng.choice(
                len(X_ref),
                size=subsample_ice,
                replace=False
            )
        )
        X_plot = X_ref.iloc[ice_idx].copy()
    else:
        ice_idx = np.arange(len(X_ref))
        X_plot = X_ref.copy()

    # Collect Figure 10 source data.
    pdp_source_frames = []
    ice_source_frames = []

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        4,
        3,
        figsize=PANEL_GRID_FIGSIZE,
        dpi=FIG_DPI
    )

    axes = axes.ravel()
    fig.subplots_adjust(**PANEL_GRID_ADJUST)
    add_panel_labels(
        axes,
        n_panels=len(features)
    )

    for ax, feat in zip(axes[:11], features):
        pd_result = partial_dependence(
            model,
            X_plot,
            features=[feat],
            kind="both",
            grid_resolution=grid_resolution
        )

        grid = np.asarray(
            pd_result["grid_values"][0],
            dtype=float
        )

        pdp = np.asarray(
            pd_result["average"][0],
            dtype=float
        )

        ice = np.asarray(
            pd_result["individual"][0],
            dtype=float
        )

        # ========================================================
        # Save PDP source values
        # ========================================================
        pdp_source_frames.append(
            pd.DataFrame({
                "Feature": feat,
                "Grid_index":
                    np.arange(1, len(grid) + 1),
                "Feature_grid_value": grid,
                "PDP_value": pdp,
            })
        )

        # ========================================================
        # Save every ICE curve
        # ========================================================
        n_ice_curves, n_grid_points = ice.shape

        ice_source_frames.append(
            pd.DataFrame({
                "Feature": feat,

                "ICE_curve_ID":
                    np.repeat(
                        np.arange(1, n_ice_curves + 1),
                        n_grid_points
                    ),

                "Source_row_index_0based":
                    np.repeat(
                        ice_idx,
                        n_grid_points
                    ),

                "Source_sample_ID_1based":
                    np.repeat(
                        ice_idx + 1,
                        n_grid_points
                    ),

                "Grid_index":
                    np.tile(
                        np.arange(1, n_grid_points + 1),
                        n_ice_curves
                    ),

                "Feature_grid_value":
                    np.tile(
                        grid,
                        n_ice_curves
                    ),

                "ICE_value":
                    ice.reshape(-1),
            })
        )

        for line in ice:
            ax.plot(
                grid,
                line,
                color=BLUE,
                linewidth=0.7,
                alpha=0.25,
                zorder=1
            )

        ax.plot(
            grid,
            pdp,
            color=RED,
            linewidth=2.5,
            zorder=2
        )

        ax.set_title("")

        apply_panel_axis_format(
            ax,
            xlabel=get_feature_axis_label(feat),
            ylabel="Partial dependence",
            grid=True,
        )

    legend_ax = axes[11]
    legend_ax.axis("off")

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=BLUE,
            linewidth=1.2,
            alpha=0.35,
            label="ICE"
        ),
        mlines.Line2D(
            [],
            [],
            color=RED,
            linewidth=2.5,
            label="PDP"
        ),
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

    # Export Figure 10 source data.
    pdp_source_df = pd.concat(
        pdp_source_frames,
        ignore_index=True
    )

    ice_source_df = pd.concat(
        ice_source_frames,
        ignore_index=True
    )

    selected_sample_df = X_plot.copy()
    selected_sample_df.insert(
        0,
        "Source_sample_ID_1based",
        ice_idx + 1
    )
    selected_sample_df.insert(
        0,
        "Source_row_index_0based",
        ice_idx
    )

    save_plot_source_bundle(
        save_stem=save_stem,
        figure_tag="Figure10",
        tables={
            "PDP_1D": pdp_source_df,
            "ICE_1D": ice_source_df,
            "ICE_selected_samples": selected_sample_df,
        },
        cache={
            "features": features,
            "grid_resolution": grid_resolution,
            "subsample_ice": subsample_ice,
            "random_state": 26,
            "source_row_indices": ice_idx,
            "X_plot": X_plot,
            "PDP_data": pdp_source_df,
            "ICE_data": ice_source_df,
        },
    )

    save_figure(fig, save_stem)


# ============================================================================
# Figure 11 helper: Single 2D PDP panel
# ============================================================================
def plot_2d_pdp_manual_single(
        ax: plt.Axes,
        model: Any,
        X_ref: pd.DataFrame,
        feature_pair: Tuple[str, str],
        plot_idx: int = 1,
        grid_resolution: int = 35,
):
    X_ref = to_dataframe(X_ref)

    f1, f2 = feature_pair
    i1 = X_ref.columns.get_loc(f1)
    i2 = X_ref.columns.get_loc(f2)

    pd_result = partial_dependence(
        model,
        X_ref,
        features=[(i1, i2)],
        kind="average",
        grid_resolution=grid_resolution,
    )

    grid_values_x = np.asarray(
        pd_result["grid_values"][0],
        dtype=float
    )

    grid_values_y = np.asarray(
        pd_result["grid_values"][1],
        dtype=float
    )

    average = np.asarray(
        pd_result["average"][0],
        dtype=float
    )

    if average.shape != (
            len(grid_values_y),
            len(grid_values_x)
    ):
        average = average.T

    Xg, Yg = np.meshgrid(
        grid_values_x,
        grid_values_y
    )

    q1, q2, q3 = np.quantile(
        average.ravel(),
        [0.25, 0.50, 0.75]
    )

    vmin_val = np.nanmin(average)
    vmax_val = np.nanmax(average)

    if np.isclose(vmin_val, vmax_val):
        vmax_val = vmin_val + 1.0

    pcm = ax.pcolormesh(
        Xg,
        Yg,
        average,
        shading="auto",
        cmap=RDBU_CMAP,
        norm=Normalize(
            vmin=vmin_val,
            vmax=vmax_val
        ),
    )

    ax.contour(
        Xg,
        Yg,
        average,
        levels=[q1, q2, q3],
        colors=[BLUE, "#374151", RED],
        linestyles=["--", "-", "--"],
        linewidths=[1.2, 1.5, 1.2],
    )

    max_idx = np.unravel_index(
        np.nanargmax(average),
        average.shape
    )

    min_idx = np.unravel_index(
        np.nanargmin(average),
        average.shape
    )

    BRIGHT_RED = "#D8090F"
    BRIGHT_BLUE = "#3BC6FA"

    ax.scatter(
        Xg[max_idx],
        Yg[max_idx],
        marker="*",
        s=130,
        color=BRIGHT_RED,
        edgecolors="black",
        linewidths=0.5,
        zorder=4
    )

    ax.scatter(
        Xg[min_idx],
        Yg[min_idx],
        marker="o",
        s=50,
        color=BRIGHT_BLUE,
        edgecolors="black",
        linewidths=0.5,
        zorder=4
    )

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=BLUE,
            linestyle="--",
            linewidth=1.2,
            label=f"Q1={q1:.1f}"
        ),
        mlines.Line2D(
            [],
            [],
            color="#374151",
            linestyle="-",
            linewidth=1.5,
            label=f"Med={q2:.1f}"
        ),
        mlines.Line2D(
            [],
            [],
            color=RED,
            linestyle="--",
            linewidth=1.2,
            label=f"Q3={q3:.1f}"
        ),
        mlines.Line2D(
            [],
            [],
            color=BRIGHT_RED,
            marker="*",
            linestyle="None",
            markersize=10,
            label="Max"
        ),
        mlines.Line2D(
            [],
            [],
            color=BRIGHT_BLUE,
            marker="o",
            linestyle="None",
            markersize=7,
            label="Min"
        ),
    ]

    loc_str = (
        "lower right"
        if plot_idx == 6
        else "upper left"
    )

    legend = ax.legend(
        handles=legend_handles,
        loc=loc_str,
        prop={
            "size": PANEL_LEGEND_FONTSIZE,
            "weight": "normal"
        },
        frameon=True,
        facecolor="white",
        framealpha=0.5,
        edgecolor="#D9D9D9",
        labelspacing=0.3
    )

    style_panel_legend(legend)

    apply_panel_axis_format(
        ax,
        xlabel=get_feature_axis_label(f1),
        ylabel=get_feature_axis_label(f2),
        grid=False,
        square=True,
    )

    source_result = {
        "feature_pair": feature_pair,
        "grid_x": grid_values_x,
        "grid_y": grid_values_y,
        "Xg": Xg,
        "Yg": Yg,
        "average": average,
        "Q1": float(q1),
        "Median": float(q2),
        "Q3": float(q3),
        "Maximum_feature_A": float(Xg[max_idx]),
        "Maximum_feature_B": float(Yg[max_idx]),
        "Maximum_PDP": float(average[max_idx]),
        "Minimum_feature_A": float(Xg[min_idx]),
        "Minimum_feature_B": float(Yg[min_idx]),
        "Minimum_PDP": float(average[min_idx]),
    }

    return pcm, source_result


# ============================================================================
# Figure 11: 2D PDP for top interaction pairs
# ============================================================================
def plot_figure11_pdp_2d_top9_pairs(
        model: Any,
        X_ref: pd.DataFrame,
        ranking_bundle: ShapRankingBundle,
        save_stem: Union[str, Path],
        grid_resolution: int = 35,
) -> None:
    X_ref = to_dataframe(X_ref)

    core_features = set(
        ranking_bundle.top6_interaction_features[:6]
    )

    pair_df = ranking_bundle.interaction_pair_df

    core_pairs_df = pair_df[
        pair_df["feature_a"].isin(core_features)
        & pair_df["feature_b"].isin(core_features)
    ]

    pairs = list(
        core_pairs_df.head(9)[
            ["feature_a", "feature_b"]
        ].itertuples(
            index=False,
            name=None
        )
    )

    # Collect source values for all nine panels
    pdp_2d_frames = []
    summary_rows = []
    pair_cache = {}

    fig, axes = plt.subplots(
        3,
        3,
        figsize=PANEL_2D_FIGSIZE,
        dpi=FIG_DPI
    )

    axes = axes.ravel()
    add_panel_labels(
        axes,
        n_panels=len(pairs)
    )

    for i, (ax, pair) in enumerate(
            zip(axes, pairs)
    ):
        pcm, source_result = (
            plot_2d_pdp_manual_single(
                ax=ax,
                model=model,
                X_ref=X_ref,
                feature_pair=pair,
                plot_idx=i + 1,
                grid_resolution=grid_resolution,
            )
        )

        feature_a, feature_b = pair

        Xg = source_result["Xg"]
        Yg = source_result["Yg"]
        average = source_result["average"]

        pdp_2d_frames.append(
            pd.DataFrame({
                "Panel_index": i + 1,
                "Feature_A": feature_a,
                "Feature_B": feature_b,
                "Feature_A_grid_value":
                    Xg.reshape(-1),
                "Feature_B_grid_value":
                    Yg.reshape(-1),
                "PDP_2D_value":
                    average.reshape(-1),
            })
        )

        summary_rows.append({
            "Panel_index": i + 1,
            "Feature_A": feature_a,
            "Feature_B": feature_b,
            "Q1": source_result["Q1"],
            "Median": source_result["Median"],
            "Q3": source_result["Q3"],
            "Maximum_feature_A":
                source_result["Maximum_feature_A"],
            "Maximum_feature_B":
                source_result["Maximum_feature_B"],
            "Maximum_PDP":
                source_result["Maximum_PDP"],
            "Minimum_feature_A":
                source_result["Minimum_feature_A"],
            "Minimum_feature_B":
                source_result["Minimum_feature_B"],
            "Minimum_PDP":
                source_result["Minimum_PDP"],
        })

        pair_cache[
            f"{feature_a}__{feature_b}"
        ] = source_result

        divider = make_axes_locatable(ax)

        cax = divider.append_axes(
            "right",
            size="5%",
            pad=0.08
        )

        cbar = fig.colorbar(
            pcm,
            cax=cax
        )

        cbar.locator = MaxNLocator(
            nbins=5
        )

        cbar.update_ticks()
        style_panel_colorbar(cbar)

    for j in range(len(pairs), len(axes)):
        axes[j].axis("off")

    pdp_2d_source_df = pd.concat(
        pdp_2d_frames,
        ignore_index=True
    )

    pdp_summary_df = pd.DataFrame(
        summary_rows
    )

    save_plot_source_bundle(
        save_stem=save_stem,
        figure_tag="Figure11",
        tables={
            "PDP_2D": pdp_2d_source_df,
            "PDP_2D_summary": pdp_summary_df,
        },
        cache={
            "pairs": pairs,
            "grid_resolution": grid_resolution,
            "pair_results": pair_cache,
            "X_reference": X_ref,
        },
    )

    fig.subplots_adjust(
        **PANEL_2D_ADJUST
    )

    save_figure(fig, save_stem)


# ============================================================================
# Figure 12: 1D ALE plots with confidence intervals
# ============================================================================
def plot_figure12_ale_1d_top11(
        model: Any,
        X_ref: pd.DataFrame,
        ranking_bundle: ShapRankingBundle,
        save_stem: Union[str, Path],
        bins: int = 20,
        n_bootstraps: int = 30,
) -> None:
    X_ref = to_dataframe(X_ref)

    features = (
        ranking_bundle.main_effect_df[
            "feature"
        ].tolist()[:11]
    )

    # Collect all ALE source values
    ale_source_frames = []
    crossing_rows = []
    ale_cache = {}

    fig, axes = plt.subplots(
        4,
        3,
        figsize=PANEL_GRID_FIGSIZE,
        dpi=FIG_DPI
    )

    axes = axes.ravel()
    fig.subplots_adjust(**PANEL_GRID_ADJUST)

    add_panel_labels(
        axes,
        n_panels=len(features)
    )

    label_position_map = {
        0: ["upper_left"],
        1: [
            "upper_left",
            "lower_left",
            "lower_right",
            "upper_right"
        ],
        2: [
            "lower_right",
            "lower_right",
            "lower_right"
        ],
        3: [
            "lower_right",
            "lower_right",
            "lower_right"
        ],
        4: [
            "upper_left",
            "lower_left",
            "lower_right",
            "upper_right",
            "upper_right"
        ],
        5: [
            "upper_right",
            "upper_right"
        ],
        6: [
            "lower_right",
            "lower_right",
            "lower_right"
        ],
        7: [
            "upper_right",
            "upper_right"
        ],
        8: [
            "lower_right",
            "lower_right",
            "lower_right"
        ],
        9: [
            "upper_right",
            "lower_right",
            "lower_left",
            "upper_left"
        ],
        10: [
            "upper_right",
            "upper_right"
        ],
    }

    for idx, feat in enumerate(features):
        ax = axes[idx]

        result = compute_1d_ale_bootstrap(
            model=model,
            X=X_ref,
            feature=feat,
            bins=bins,
            n_bootstraps=n_bootstraps,
            random_state=42,
        )

        edges = np.asarray(
            result["edges"],
            dtype=float
        )

        x_centers = np.asarray(
            result["centers"],
            dtype=float
        )

        y_main = np.asarray(
            result["ale"],
            dtype=float
        )

        y_lower = np.asarray(
            result["ale_lower"],
            dtype=float
        )

        y_upper = np.asarray(
            result["ale_upper"],
            dtype=float
        )

        # ========================================================
        # Count observations in every ALE interval
        # ========================================================
        feature_values = X_ref[
            feat
        ].to_numpy(dtype=float)

        bin_counts = []

        for bin_index in range(
                len(edges) - 1
        ):
            left_edge = edges[bin_index]
            right_edge = edges[bin_index + 1]

            if bin_index == len(edges) - 2:
                mask = (
                    (feature_values >= left_edge)
                    & (feature_values <= right_edge)
                )
            else:
                mask = (
                    (feature_values >= left_edge)
                    & (feature_values < right_edge)
                )

            bin_counts.append(
                int(mask.sum())
            )

        ale_feature_df = pd.DataFrame({
            "Feature": feat,
            "Bin_index":
                np.arange(
                    1,
                    len(x_centers) + 1
                ),
            "Bin_left_edge": edges[:-1],
            "Bin_right_edge": edges[1:],
            "Bin_center": x_centers,
            "Sample_count": bin_counts,
            "ALE_value": y_main,
            "ALE_95CI_lower": y_lower,
            "ALE_95CI_upper": y_upper,
        })

        ale_source_frames.append(
            ale_feature_df
        )

        ale_cache[feat] = {
            "edges": edges,
            "centers": x_centers,
            "ale": y_main,
            "ale_lower": y_lower,
            "ale_upper": y_upper,
            "sample_counts":
                np.asarray(bin_counts),
        }

        xmin_plot = x_centers.min()
        xmax_plot = x_centers.max()

        ymin_plot = y_lower.min()
        ymax_plot = y_upper.max()

        xpad = 0.05 * (
            xmax_plot - xmin_plot + 1e-12
        )

        ypad = 0.08 * (
            ymax_plot - ymin_plot + 1e-12
        )

        xmin_bound = xmin_plot - xpad
        xmax_bound = xmax_plot + xpad

        ymin_bound = ymin_plot - ypad
        ymax_bound = ymax_plot + ypad

        ax.set_xlim(
            xmin_bound,
            xmax_bound
        )

        ax.set_ylim(
            ymin_bound,
            ymax_bound
        )

        ax.fill_between(
            x_centers,
            y_lower,
            y_upper,
            color=BLUE,
            alpha=0.28,
            zorder=2
        )

        ax.plot(
            x_centers,
            y_main,
            color=BLUE,
            linewidth=2.2,
            zorder=4
        )

        ax.axhline(
            0.0,
            color=MID_GREY,
            linestyle="--",
            linewidth=1.0,
            zorder=1
        )

        crossings = (
            shade_promoting_inhibiting_regions(
                ax=ax,
                x_smooth=x_centers,
                y_smooth=y_main,
                xmin_plot=xmin_bound,
                xmax_plot=xmax_bound,
                ymin_plot=ymin_bound,
                ymax_plot=ymax_bound,
                pos_color="#D65E5E",
                neg_color="#1f77b4",
                alpha=0.08,
            )
        )

        for crossing_index, crossing_value in enumerate(
                crossings,
                start=1
        ):
            crossing_rows.append({
                "Feature": feat,
                "Crossing_index":
                    crossing_index,
                "Zero_crossing_feature_value":
                    float(crossing_value),
            })

        positions = label_position_map.get(
            idx,
            []
        )

        for j, xc in enumerate(crossings):
            ax.axvline(
                xc,
                color=RED,
                linestyle="--",
                linewidth=1.5,
                zorder=1
            )

            pos_code = (
                positions[j]
                if j < len(positions)
                else "upper_right"
            )

            add_crossing_label(
                ax=ax,
                xc=xc,
                text=f"{xc:.2f}",
                pos_code=pos_code,
                xmin_plot=xmin_bound,
                xmax_plot=xmax_bound,
                ymin_plot=ymin_bound,
                ymax_plot=ymax_bound,
                color=BLUE,
                fontsize=
                    PANEL_MARKER_LABEL_FONTSIZE,
            )

        apply_panel_axis_format(
            ax,
            xlabel=get_feature_axis_label(feat),
            ylabel="ALE value",
            grid=True,
        )

    legend_ax = axes[11]
    legend_ax.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BLUE,
            lw=2.2,
            label="ALE Main Effect"
        ),
        Patch(
            facecolor=BLUE,
            alpha=0.28,
            label="95% CI (Bootstrap)"
        ),
        Patch(
            facecolor="#D65E5E",
            alpha=0.12,
            label="Promoting region"
        ),
        Patch(
            facecolor="#1f77b4",
            alpha=0.12,
            label="Inhibiting region"
        ),
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

    # ============================================================
    # Export Figure 12 source data
    # ============================================================
    ale_source_df = pd.concat(
        ale_source_frames,
        ignore_index=True
    )

    crossing_df = pd.DataFrame(
        crossing_rows,
        columns=[
            "Feature",
            "Crossing_index",
            "Zero_crossing_feature_value",
        ]
    )

    save_plot_source_bundle(
        save_stem=save_stem,
        figure_tag="Figure12",
        tables={
            "ALE_1D": ale_source_df,
            "ALE_zero_crossings": crossing_df,
        },
        cache={
            "features": features,
            "bins": bins,
            "n_bootstraps": n_bootstraps,
            "bootstrap_random_state": 42,
            "ALE_results": ale_cache,
            "X_reference": X_ref,
        },
    )

    save_figure(fig, save_stem)


# ============================================================================
# Figure 13 helper: Single 2D ALE panel
# ============================================================================
def plot_2d_ale_manual_single(
        ax: plt.Axes,
        model: Any,
        X_ref: pd.DataFrame,
        feature_pair: Tuple[str, str],
        plot_idx: int = 1,
        bins: int = 12,
):
    f1, f2 = feature_pair

    result = compute_2d_ale(
        model,
        X_ref,
        feature_pair,
        bins=bins
    )

    Xg = np.asarray(
        result["Xg"],
        dtype=float
    )

    Yg = np.asarray(
        result["Yg"],
        dtype=float
    )

    ale_grid = np.asarray(
        result["ale"],
        dtype=float
    )

    counts = np.asarray(
        result["counts"],
        dtype=int
    )

    q1, q2, q3 = np.quantile(
        ale_grid.ravel(),
        [0.25, 0.50, 0.75]
    )

    vmin_val = np.nanmin(ale_grid)
    vmax_val = np.nanmax(ale_grid)

    if np.isclose(vmin_val, vmax_val):
        vmax_val = vmin_val + 1.0

    pcm = ax.pcolormesh(
        Xg,
        Yg,
        ale_grid,
        shading="auto",
        cmap=RDBU_CMAP,
        norm=Normalize(
            vmin=vmin_val,
            vmax=vmax_val
        ),
    )

    ax.contour(
        Xg,
        Yg,
        ale_grid,
        levels=[q1, q2, q3],
        colors=[BLUE, "#374151", RED],
        linestyles=["--", "-", "--"],
        linewidths=[1.2, 1.5, 1.2],
    )

    max_idx = np.unravel_index(
        np.nanargmax(ale_grid),
        ale_grid.shape
    )

    min_idx = np.unravel_index(
        np.nanargmin(ale_grid),
        ale_grid.shape
    )

    BRIGHT_RED = "#D8090F"
    BRIGHT_BLUE = "#3BC6FA"

    ax.scatter(
        Xg[max_idx],
        Yg[max_idx],
        marker="*",
        s=130,
        color=BRIGHT_RED,
        edgecolors="black",
        linewidths=0.5,
        zorder=4
    )

    ax.scatter(
        Xg[min_idx],
        Yg[min_idx],
        marker="o",
        s=50,
        color=BRIGHT_BLUE,
        edgecolors="black",
        linewidths=0.5,
        zorder=4
    )

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=BLUE,
            linestyle="--",
            linewidth=1.2,
            label=f"Q1={q1:.1f}"
        ),
        mlines.Line2D(
            [],
            [],
            color="#374151",
            linestyle="-",
            linewidth=1.5,
            label=f"Med={q2:.1f}"
        ),
        mlines.Line2D(
            [],
            [],
            color=RED,
            linestyle="--",
            linewidth=1.2,
            label=f"Q3={q3:.1f}"
        ),
        mlines.Line2D(
            [],
            [],
            color=BRIGHT_RED,
            marker="*",
            linestyle="None",
            markersize=10,
            label="Max"
        ),
        mlines.Line2D(
            [],
            [],
            color=BRIGHT_BLUE,
            marker="o",
            linestyle="None",
            markersize=7,
            label="Min"
        ),
    ]

    loc_str = (
        "lower right"
        if plot_idx == 6
        else "upper left"
    )

    legend = ax.legend(
        handles=legend_handles,
        loc=loc_str,
        prop={
            "size": PANEL_LEGEND_FONTSIZE,
            "weight": "normal"
        },
        frameon=True,
        facecolor="white",
        framealpha=0.5,
        edgecolor="#D9D9D9",
        labelspacing=0.3
    )

    style_panel_legend(legend)

    apply_panel_axis_format(
        ax,
        xlabel=get_feature_axis_label(f1),
        ylabel=get_feature_axis_label(f2),
        grid=False,
        square=True,
    )

    source_result = {
        "feature_pair": feature_pair,
        "edges1": result["edges1"],
        "edges2": result["edges2"],
        "centers1": result["centers1"],
        "centers2": result["centers2"],
        "Xg": Xg,
        "Yg": Yg,
        "ale": ale_grid,
        "counts": counts,
        "Q1": float(q1),
        "Median": float(q2),
        "Q3": float(q3),
        "Maximum_feature_A":
            float(Xg[max_idx]),
        "Maximum_feature_B":
            float(Yg[max_idx]),
        "Maximum_ALE":
            float(ale_grid[max_idx]),
        "Minimum_feature_A":
            float(Xg[min_idx]),
        "Minimum_feature_B":
            float(Yg[min_idx]),
        "Minimum_ALE":
            float(ale_grid[min_idx]),
    }

    return pcm, source_result


# ============================================================================
# Figure 13: 2D ALE for top interaction pairs
# ============================================================================
def plot_figure13_ale_2d_top9_pairs(
        model: Any,
        X_ref: pd.DataFrame,
        ranking_bundle: ShapRankingBundle,
        save_stem: Union[str, Path],
        bins: int = 12,
) -> None:
    X_ref = to_dataframe(X_ref)

    core_features = set(
        ranking_bundle.top6_interaction_features[:6]
    )

    pair_df = ranking_bundle.interaction_pair_df

    core_pairs_df = pair_df[
        pair_df["feature_a"].isin(core_features)
        & pair_df["feature_b"].isin(core_features)
    ]

    pairs = list(
        core_pairs_df.head(9)[
            ["feature_a", "feature_b"]
        ].itertuples(
            index=False,
            name=None
        )
    )

    ale_2d_frames = []
    summary_rows = []
    pair_cache = {}

    fig, axes = plt.subplots(
        3,
        3,
        figsize=PANEL_2D_FIGSIZE,
        dpi=FIG_DPI
    )

    axes = axes.ravel()

    add_panel_labels(
        axes,
        n_panels=len(pairs)
    )

    for i, (ax, pair) in enumerate(
            zip(axes, pairs)
    ):
        pcm, source_result = (
            plot_2d_ale_manual_single(
                ax=ax,
                model=model,
                X_ref=X_ref,
                feature_pair=pair,
                plot_idx=i + 1,
                bins=bins,
            )
        )

        feature_a, feature_b = pair

        Xg = source_result["Xg"]
        Yg = source_result["Yg"]
        ale_grid = source_result["ale"]
        counts = source_result["counts"]

        ale_2d_frames.append(
            pd.DataFrame({
                "Panel_index": i + 1,
                "Feature_A": feature_a,
                "Feature_B": feature_b,
                "Feature_A_bin_center":
                    Xg.reshape(-1),
                "Feature_B_bin_center":
                    Yg.reshape(-1),
                "ALE_2D_value":
                    ale_grid.reshape(-1),
                "Samples_in_bin":
                    counts.reshape(-1),
            })
        )

        summary_rows.append({
            "Panel_index": i + 1,
            "Feature_A": feature_a,
            "Feature_B": feature_b,
            "Q1": source_result["Q1"],
            "Median": source_result["Median"],
            "Q3": source_result["Q3"],
            "Maximum_feature_A":
                source_result["Maximum_feature_A"],
            "Maximum_feature_B":
                source_result["Maximum_feature_B"],
            "Maximum_ALE":
                source_result["Maximum_ALE"],
            "Minimum_feature_A":
                source_result["Minimum_feature_A"],
            "Minimum_feature_B":
                source_result["Minimum_feature_B"],
            "Minimum_ALE":
                source_result["Minimum_ALE"],
        })

        pair_cache[
            f"{feature_a}__{feature_b}"
        ] = source_result

        divider = make_axes_locatable(ax)

        cax = divider.append_axes(
            "right",
            size="5%",
            pad=0.08
        )

        cbar = fig.colorbar(
            pcm,
            cax=cax
        )

        cbar.locator = MaxNLocator(
            nbins=5
        )

        cbar.update_ticks()
        style_panel_colorbar(cbar)

    for j in range(len(pairs), len(axes)):
        axes[j].axis("off")

    ale_2d_source_df = pd.concat(
        ale_2d_frames,
        ignore_index=True
    )

    ale_2d_summary_df = pd.DataFrame(
        summary_rows
    )

    save_plot_source_bundle(
        save_stem=save_stem,
        figure_tag="Figure13",
        tables={
            "ALE_2D": ale_2d_source_df,
            "ALE_2D_summary":
                ale_2d_summary_df,
        },
        cache={
            "pairs": pairs,
            "bins": bins,
            "pair_results": pair_cache,
            "X_reference": X_ref,
        },
    )

    fig.subplots_adjust(
        **PANEL_2D_ADJUST
    )

    save_figure(fig, save_stem)


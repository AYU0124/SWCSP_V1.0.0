import json
import os
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd

from diagnostic_plots import generate_all_diagnostic_plots
from figure_output1 import (
    ensure_dir,
    load_predictions_file,
    plot_prediction_line_figure,
    plot_prediction_validation_figure,
    prepare_interpretation_context,
    plot_figure2_shap_global_contribution,
    plot_figure3_shap_dependence_grid,
    plot_figure4_main_vs_interaction_bar,
    plot_figure5_interaction_matrix,
    plot_figure6_core_pairwise_scatter,
    plot_figure7_feature_interaction_network,
)
from figure_output2 import (
    plot_figure8_shap_heatmap,
    plot_figure9_typical_force_waterfalls,
    plot_figure10_ice_pdp_grid,
    plot_figure11_pdp_2d_top9_pairs,
    plot_figure12_ale_1d_top11,
    plot_figure13_ale_2d_top9_pairs,
)
from model import run_all_models
from data_processing import load_and_preprocess_data, make_stratified_regression_cv

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DATASET_FILE = str(DATA_DIR / "SS-ML.xlsx")
RESULTS_DIR = str(OUTPUT_ROOT / "Results")
TEST_SIZE = 0.2

SEED_LIST = [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
PRIMARY_SPLIT_SEED = 52

if PRIMARY_SPLIT_SEED not in SEED_LIST:
    raise ValueError(
        f"PRIMARY_SPLIT_SEED={PRIMARY_SPLIT_SEED} is not included in SEED_LIST={SEED_LIST}"
    )

USE_STRATIFIED_REGRESSION_SPLIT = True
N_STRATA_BINS = 5

USE_STRATIFIED_REGRESSION_CV = True
CV_SPLITS = 10
CV_SHUFFLE = True
CV_RANDOM_STATE = 26
CV_N_STRATA_BINS = 5

# ============================================================================
# Run switches
# ============================================================================
RUN_TRAINING = True
RUN_DIAGNOSTICS = True
RUN_PLOTTING = True
RUN_MULTI_SEED_SUMMARY = True

# Set True to run a figure and False to skip it.
FIGURE_RUN_FLAGS: Dict[int, bool] = {
    1: True,
    2: True,
    3: True,
    4: True,
    5: True,
    6: True,
    7: True,
    8: True,
    9: True,
    10: True,
    11: True,
    12: True,
    13: True,
}

FIG1_MODEL_NAMES = [
    "Dummy", "Linear", "Ridge", "Lasso", "DT", "RF",
    "GB", "SVR", "KNN", "MLP", "XGB", "LGBM"
]

# Figures 2-13 use one selected tree-based model for interpretation.
TREE_MODEL_CANDIDATES = {"DT", "RF", "GB", "XGB", "LGBM"}
INTERPRET_MODEL_NAME = "LGBM"

if INTERPRET_MODEL_NAME not in TREE_MODEL_CANDIDATES:
    raise ValueError(
        f"INTERPRET_MODEL_NAME must be one of {sorted(TREE_MODEL_CANDIDATES)}, "
        f"but got: {INTERPRET_MODEL_NAME}"
    )


# ============================================================================
# General helpers
# ============================================================================
def get_enabled_figures() -> List[int]:
    return [fig_no for fig_no, enabled in FIGURE_RUN_FLAGS.items() if enabled]


def ensure_dir_local(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_cv():
    if USE_STRATIFIED_REGRESSION_CV:
        return make_stratified_regression_cv(
            n_splits=CV_SPLITS,
            shuffle=CV_SHUFFLE,
            random_state=CV_RANDOM_STATE,
            n_strata_bins=CV_N_STRATA_BINS,
        )
    raise ValueError("This script is configured for stratified regression CV only.")


# ============================================================================
# Data split artifacts
# ============================================================================
def save_split_artifacts(data_pack: dict, output_dir: str) -> None:
    ensure_dir_local(output_dir)

    data_pack["X_train_raw"].to_csv(
        os.path.join(output_dir, "X_train_raw.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    data_pack["X_test_raw"].to_csv(
        os.path.join(output_dir, "X_test_raw.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.Series(data_pack["y_train"]).to_csv(
        os.path.join(output_dir, "y_train.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.Series(data_pack["y_test"]).to_csv(
        os.path.join(output_dir, "y_test.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.Series(data_pack["train_strata"], name="stratum").to_csv(
        os.path.join(output_dir, "train_strata.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.Series(data_pack["test_strata"], name="stratum").to_csv(
        os.path.join(output_dir, "test_strata.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    data_pack["split_summary_df"].to_csv(
        os.path.join(output_dir, "split_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    meta = {
        "n_total_samples": data_pack["n_total_samples"],
        "test_size": data_pack["test_size"],
        "split_random_state": data_pack["split_random_state"],
        "use_stratified_regression_split": data_pack["use_stratified_regression_split"],
        "requested_n_strata_bins": data_pack["requested_n_strata_bins"],
        "actual_n_strata_bins": data_pack["actual_n_strata_bins"],
        "target_name": data_pack["target_name"],
        "feature_names": data_pack["feature_names"],
    }
    with open(os.path.join(output_dir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_data_pack(split_seed: int):
    return load_and_preprocess_data(
        DATASET_FILE,
        test_size=TEST_SIZE,
        split_random_state=split_seed,
        n_strata_bins=N_STRATA_BINS,
        use_stratified_regression_split=USE_STRATIFIED_REGRESSION_SPLIT,
    )


# ============================================================================
# Training and diagnostics
# ============================================================================
def run_training_pipeline():
    ensure_dir_local(RESULTS_DIR)

    data_pack = build_data_pack(PRIMARY_SPLIT_SEED)
    save_split_artifacts(data_pack, RESULTS_DIR)

    X_train_raw = data_pack["X_train_raw"]
    X_test_raw = data_pack["X_test_raw"]
    y_train = data_pack["y_train"]
    y_test = data_pack["y_test"]

    cv = build_cv()

    outputs, summary_df = run_all_models(
        X_train=X_train_raw,
        y_train=y_train,
        X_test=X_test_raw,
        y_test=y_test,
        cv=cv,
        save_root=RESULTS_DIR,
        random_state=CV_RANDOM_STATE,
    )

    print("=" * 90)
    print("Training finished on primary split.")
    print(f"Primary split seed for figures: {PRIMARY_SPLIT_SEED}")
    print(f"Summary CSV:  {os.path.join(RESULTS_DIR, 'summary_metrics.csv')}")
    print(f"Summary XLSX: {os.path.join(RESULTS_DIR, 'summary_metrics.xlsx')}")
    print("=" * 90)

    diagnostic_dir = os.path.join(RESULTS_DIR, "Diagnostic_Plots")
    diagnostic_model_name = INTERPRET_MODEL_NAME

    if RUN_DIAGNOSTICS and diagnostic_model_name in outputs:
        generate_all_diagnostic_plots(
            best_model=outputs[diagnostic_model_name]["model"],
            X_train=X_train_raw,
            y_train=y_train,
            output_dir=diagnostic_dir,
            model_name=diagnostic_model_name,
            fixed_val_size=0.2,
            fixed_val_random_state=26,
            fixed_val_n_strata_bins=CV_N_STRATA_BINS,
        )
        print(f"Diagnostic plots finished: {diagnostic_dir}")
    elif RUN_DIAGNOSTICS:
        print(
            f"[Warning] '{diagnostic_model_name}' not found in outputs. "
            "Diagnostic plots skipped."
        )
    else:
        print("Diagnostic plots are disabled.")

    return summary_df


# ============================================================================
# Multi-seed summary
# ============================================================================
def run_multi_seed_summary():
    if not RUN_MULTI_SEED_SUMMARY:
        print("Multi-seed traversal is disabled.")
        return

    multi_seed_summary_list = []
    multi_seed_temp_root = os.path.join(RESULTS_DIR, "MultiSeeds_Temp")
    ensure_dir_local(multi_seed_temp_root)

    print("=" * 90)
    print("Starting multi-seed traversal...")
    print(f"Seed list: {SEED_LIST}")
    print("=" * 90)

    for seed in SEED_LIST:
        print("=" * 90)
        print(f"Running multi-seed summary | Seed = {seed}")
        print("=" * 90)

        data_pack_seed = build_data_pack(seed)
        seed_save_root = os.path.join(multi_seed_temp_root, f"Seed_{seed}")
        ensure_dir_local(seed_save_root)
        save_split_artifacts(data_pack_seed, seed_save_root)

        X_train_raw_seed = data_pack_seed["X_train_raw"]
        X_test_raw_seed = data_pack_seed["X_test_raw"]
        y_train_seed = data_pack_seed["y_train"]
        y_test_seed = data_pack_seed["y_test"]

        cv_seed = build_cv()
        _, summary_df_seed = run_all_models(
            X_train=X_train_raw_seed,
            y_train=y_train_seed,
            X_test=X_test_raw_seed,
            y_test=y_test_seed,
            cv=cv_seed,
            save_root=seed_save_root,
            random_state=CV_RANDOM_STATE,
        )

        summary_df_seed = summary_df_seed.copy()
        summary_df_seed.insert(0, "Seed", seed)
        summary_df_seed.insert(1, "Requested_Strata_Bins", N_STRATA_BINS)
        summary_df_seed.insert(2, "Actual_Strata_Bins", data_pack_seed["actual_n_strata_bins"])
        multi_seed_summary_list.append(summary_df_seed)

    summary_all_seeds_df = (
        pd.concat(multi_seed_summary_list, axis=0, ignore_index=True)
        if multi_seed_summary_list else pd.DataFrame()
    )

    summary_all_seeds_csv = os.path.join(RESULTS_DIR, "summary_all_seeds.csv")
    summary_all_seeds_xlsx = os.path.join(RESULTS_DIR, "summary_all_seeds.xlsx")

    summary_all_seeds_df.to_csv(summary_all_seeds_csv, index=False, encoding="utf-8-sig")
    summary_all_seeds_df.to_excel(summary_all_seeds_xlsx, index=False)

    print("=" * 90)
    print("Multi-seed traversal finished.")
    print(f"Multi-seed summary CSV:  {summary_all_seeds_csv}")
    print(f"Multi-seed summary XLSX: {summary_all_seeds_xlsx}")
    print("=" * 90)


# ============================================================================
# Figure generation
# ============================================================================
def run_figure_1(results_dir: str, model_names: List[str]) -> None:
    print("[Plot] Running Figure 1 ...")
    results_root = os.path.abspath(results_dir)

    for model_name in model_names:
        model_dir = os.path.join(results_root, model_name)
        pred_path = os.path.join(model_dir, f"predictions_{model_name}.csv")

        if not os.path.exists(pred_path):
            print(f"[Info] Figure 1 skipped for {model_name}: missing {pred_path}")
            continue

        try:
            train_df, test_df = load_predictions_file(pred_path)
        except Exception as exc:
            print(f"[Warning] Figure 1 skipped for {model_name}: {exc}")
            continue

        plot_dir = os.path.join(model_dir, "plots")
        ensure_dir_local(plot_dir)

        plot_prediction_validation_figure(
            train_df["y_true"], train_df["y_pred"],
            test_df["y_true"], test_df["y_pred"],
            model_name,
            os.path.join(plot_dir, f"{model_name}_joint_parity_residual"),
        )
        plot_prediction_line_figure(
            train_df["y_true"], train_df["y_pred"],
            test_df["y_true"], test_df["y_pred"],
            model_name,
            os.path.join(plot_dir, f"{model_name}_prediction_line"),
        )


def prepare_single_figure_context(results_dir: str, model_name: str):
    ctx = prepare_interpretation_context(
        results_dir=results_dir,
        model_name=model_name,
    )
    return ctx, ctx["interpret_dir"]


def run_single_figure(fig_no: int, ctx: dict, save_dir: Path) -> None:
    model = ctx["model"]
    X_train_df = ctx["X_train_df"]
    X_eval_df = ctx["X_eval_df"]
    X_inter_df = ctx["X_inter_df"]
    shap_values_eval = ctx["shap_values_eval"]
    interaction_values = ctx["interaction_values"]
    expected_value = ctx["expected_value"]
    ranking_bundle = ctx["ranking_bundle"]

    print(f"[Plot] Running Figure {fig_no} ...")

    fig_dir = save_dir / f"{fig_no:02d}_Figure{fig_no}"
    ensure_dir(fig_dir)

    if fig_no == 2:
        plot_figure2_shap_global_contribution(
            shap_values_eval, X_eval_df, fig_dir / "Figure2_SHAP_Global_Contribution"
        )
    elif fig_no == 3:
        plot_figure3_shap_dependence_grid(
            shap_values_eval, X_eval_df, ranking_bundle, fig_dir / "Figure3_SHAP_Dependence_Grid"
        )
    elif fig_no == 4:
        plot_figure4_main_vs_interaction_bar(
            shap_values_eval, interaction_values, X_eval_df, ranking_bundle, fig_dir / "Figure4_Main_vs_Interaction_Bar"
        )
    elif fig_no == 5:
        plot_figure5_interaction_matrix(
            interaction_values, X_inter_df, ranking_bundle, fig_dir / "Figure5_Interaction_Matrix"
        )
    elif fig_no == 6:
        plot_figure6_core_pairwise_scatter(
            interaction_values, X_inter_df, ranking_bundle, fig_dir / "Figure6_Core_Pairwise_Interactions"
        )
    elif fig_no == 7:
        plot_figure7_feature_interaction_network(
            shap_values_eval, interaction_values, X_eval_df, fig_dir / "Figure7_Interaction_Network"
        )
    elif fig_no == 8:
        plot_figure8_shap_heatmap(
            shap_values_eval, X_eval_df, model, fig_dir / "Figure8_SHAP_Heatmap"
        )
    elif fig_no == 9:
        plot_figure9_typical_force_waterfalls(
            shap_values_eval,
            X_eval_df,
            model,
            fig_dir / "Figure9_Typical_Force_Plots",
            expected_value=expected_value,
        )
    elif fig_no == 10:
        plot_figure10_ice_pdp_grid(
            model, X_train_df, ranking_bundle, fig_dir / "Figure10_ICE_PDP_Grid"
        )
    elif fig_no == 11:
        plot_figure11_pdp_2d_top9_pairs(
            model, X_train_df, ranking_bundle, fig_dir / "Figure11_PDP_2D_Top9_Pairs"
        )
    elif fig_no == 12:
        plot_figure12_ale_1d_top11(
            model, X_train_df, ranking_bundle, fig_dir / "Figure12_ALE_1D_11_Features"
        )
    elif fig_no == 13:
        plot_figure13_ale_2d_top9_pairs(
            model, X_train_df, ranking_bundle, fig_dir / "Figure13_ALE_2D_Top9_Pairs"
        )
    else:
        raise ValueError(f"Unsupported figure number: {fig_no}")


def run_plot_pipeline():
    enabled_figures = get_enabled_figures()

    print("=" * 90)
    print("Plot pipeline is ON.")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Enabled single figures: {enabled_figures}")
    print("=" * 90)

    if not enabled_figures:
        print("[Info] No figures selected. Please set FIGURE_RUN_FLAGS.")
        print("=" * 90)
        print("Plot pipeline finished.")
        print("=" * 90)
        return

    if 1 in enabled_figures:
        run_figure_1(RESULTS_DIR, FIG1_MODEL_NAMES)

    figs_need_ctx = [f for f in enabled_figures if f >= 2]
    if figs_need_ctx:
        ctx, interpret_dir = prepare_single_figure_context(RESULTS_DIR, INTERPRET_MODEL_NAME)
        for fig_no in figs_need_ctx:
            run_single_figure(fig_no, ctx, interpret_dir)

    print("=" * 90)
    print("Plot pipeline finished.")
    print("=" * 90)


# ============================================================================
# Entry point
# ============================================================================
def main():
    ensure_dir_local(RESULTS_DIR)

    if RUN_TRAINING:
        run_training_pipeline()

    if RUN_MULTI_SEED_SUMMARY:
        run_multi_seed_summary()

    if RUN_PLOTTING:
        run_plot_pipeline()


if __name__ == "__main__":
    main()

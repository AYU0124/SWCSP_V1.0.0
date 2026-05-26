import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from calculations import (
    DEFAULT_INPUTS,
    INPUT_LIMITS,
    FEATURE_ORDER,
    load_model_and_metadata,
    predict_co2_uptake,
    calculate_theoretical_co2,
    calculate_process_inventory,
    calculate_lca_and_cost,
    calculate_shap_contrib,
    validate_inputs,
)
from ui_style import inject_css, t


st.set_page_config(
    page_title="基于 LGBM 的钢渣 CO₂ 固存量网页预测计算器",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


MODEL_PATH = Path("final_lgbm_model.joblib")
METADATA_PATH = Path("final_lgbm_metadata.json")


def init_session_state():
    if "lang" not in st.session_state:
        st.session_state.lang = "简体中文"

    if "page" not in st.session_state:
        st.session_state.page = "prediction"

    if "inputs" not in st.session_state:
        st.session_state.inputs = DEFAULT_INPUTS.copy()

    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    if "history" not in st.session_state:
        st.session_state.history = []


def set_input_value(key, value):
    st.session_state.inputs[key] = value


def number_control(label, key, unit="", step=1.0):
    lang = st.session_state.lang
    min_v, max_v = INPUT_LIMITS[key]
    value = float(st.session_state.inputs.get(key, DEFAULT_INPUTS[key]))

    st.markdown(f'<div class="input-row">', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([1.25, 0.34, 0.7, 0.34, 0.65])

    with c1:
        st.markdown(f'<div class="param-label">{label}</div>', unsafe_allow_html=True)

    with c2:
        if st.button("−", key=f"{key}_minus", use_container_width=True):
            new_value = max(min_v, value - step)
            set_input_value(key, round(new_value, 6))
            st.rerun()

    with c3:
        new_value = st.number_input(
            label=key,
            value=value,
            min_value=float(min_v),
            max_value=float(max_v),
            step=float(step),
            key=f"{key}_number",
            label_visibility="collapsed",
            format="%.6g",
        )
        st.session_state.inputs[key] = float(new_value)

    with c4:
        if st.button("+", key=f"{key}_plus", use_container_width=True):
            new_value = min(max_v, value + step)
            set_input_value(key, round(new_value, 6))
            st.rerun()

    with c5:
        if unit:
            st.markdown(f'<div class="unit-text">{unit}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="range-text">范围：[{min_v} - {max_v}]</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def particle_size_control():
    key = "particle_size"
    label = t("particle_size", st.session_state.lang)
    min_v, max_v = INPUT_LIMITS[key]
    value = float(st.session_state.inputs.get(key, DEFAULT_INPUTS[key]))

    st.markdown(f'<div class="input-row">', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([1.25, 0.34, 0.7, 0.34, 0.65])

    with c1:
        st.markdown(f'<div class="param-label">{label}</div>', unsafe_allow_html=True)

    with c2:
        if st.button("−", key=f"{key}_minus", use_container_width=True):
            new_value = max(min_v, value - 1.0)
            set_input_value(key, round(new_value, 6))
            st.rerun()

    with c3:
        new_value = st.number_input(
            label=key,
            value=value,
            min_value=float(min_v),
            max_value=float(max_v),
            step=1.0,
            key=f"{key}_number",
            label_visibility="collapsed",
            format="%.6g",
        )
        st.session_state.inputs[key] = float(new_value)

    with c4:
        if st.button("+", key=f"{key}_plus", use_container_width=True):
            new_value = min(max_v, value + 1.0)
            set_input_value(key, round(new_value, 6))
            st.rerun()

    with c5:
        unit = st.selectbox(
            "particle unit",
            ["μm"],
            index=0,
            key="particle_size_unit",
            label_visibility="collapsed",
        )

    st.markdown(
        f'<div class="range-text">范围：[{min_v} - {max_v}]</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def temperature_control():
    key = "temperature"
    label = t("temperature", st.session_state.lang)
    min_v, max_v = INPUT_LIMITS[key]
    value = float(st.session_state.inputs.get(key, DEFAULT_INPUTS[key]))

    st.markdown(f'<div class="input-row">', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([1.25, 0.34, 0.7, 0.34, 0.65])

    with c1:
        st.markdown(f'<div class="param-label">{label}</div>', unsafe_allow_html=True)

    with c2:
        if st.button("−", key=f"{key}_minus", use_container_width=True):
            new_value = max(min_v, value - 1.0)
            set_input_value(key, round(new_value, 6))
            st.rerun()

    with c3:
        new_value = st.number_input(
            label=key,
            value=value,
            min_value=float(min_v),
            max_value=float(max_v),
            step=1.0,
            key=f"{key}_number",
            label_visibility="collapsed",
            format="%.6g",
        )
        st.session_state.inputs[key] = float(new_value)

    with c4:
        if st.button("+", key=f"{key}_plus", use_container_width=True):
            new_value = min(max_v, value + 1.0)
            set_input_value(key, round(new_value, 6))
            st.rerun()

    with c5:
        st.selectbox(
            "temperature unit",
            ["℃"],
            index=0,
            key="temperature_unit",
            label_visibility="collapsed",
        )

    st.markdown(
        f'<div class="range-text">范围：[{min_v} - {max_v}]</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def time_control():
    key = "reaction_time"
    label = t("reaction_time", st.session_state.lang)
    min_v, max_v = INPUT_LIMITS[key]
    value = float(st.session_state.inputs.get(key, DEFAULT_INPUTS[key]))

    st.markdown(f'<div class="input-row">', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([1.25, 0.34, 0.7, 0.34, 0.65])

    with c1:
        st.markdown(f'<div class="param-label">{label}</div>', unsafe_allow_html=True)

    with c2:
        if st.button("−", key=f"{key}_minus", use_container_width=True):
            new_value = max(min_v, value - 1.0)
            set_input_value(key, round(new_value, 6))
            st.rerun()

    with c3:
        new_value = st.number_input(
            label=key,
            value=value,
            min_value=float(min_v),
            max_value=float(max_v),
            step=1.0,
            key=f"{key}_number",
            label_visibility="collapsed",
            format="%.6g",
        )
        st.session_state.inputs[key] = float(new_value)

    with c4:
        if st.button("+", key=f"{key}_plus", use_container_width=True):
            new_value = min(max_v, value + 1.0)
            set_input_value(key, round(new_value, 6))
            st.rerun()

    with c5:
        st.selectbox(
            "time unit",
            ["h"],
            index=0,
            key="time_unit",
            label_visibility="collapsed",
        )

    st.markdown(
        f'<div class="range-text">范围：[{min_v} - {max_v}]</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def pressure_control():
    key = "co2_pressure"
    label = t("co2_pressure", st.session_state.lang)
    min_v, max_v = INPUT_LIMITS[key]
    value = float(st.session_state.inputs.get(key, DEFAULT_INPUTS[key]))

    st.markdown(f'<div class="input-row">', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([1.25, 0.34, 0.7, 0.34, 0.65])

    with c1:
        st.markdown(f'<div class="param-label">{label}</div>', unsafe_allow_html=True)

    with c2:
        if st.button("−", key=f"{key}_minus", use_container_width=True):
            new_value = max(min_v, value - 0.1)
            set_input_value(key, round(new_value, 6))
            st.rerun()

    with c3:
        new_value = st.number_input(
            label=key,
            value=value,
            min_value=float(min_v),
            max_value=float(max_v),
            step=0.1,
            key=f"{key}_number",
            label_visibility="collapsed",
            format="%.6g",
        )
        st.session_state.inputs[key] = float(new_value)

    with c4:
        if st.button("+", key=f"{key}_plus", use_container_width=True):
            new_value = min(max_v, value + 0.1)
            set_input_value(key, round(new_value, 6))
            st.rerun()

    with c5:
        st.selectbox(
            "pressure unit",
            ["MPa"],
            index=0,
            key="pressure_unit",
            label_visibility="collapsed",
        )

    st.markdown(
        f'<div class="range-text">范围：[{min_v} - {max_v}]</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def left_input_panel():
    lang = st.session_state.lang

    st.markdown('<div class="left-panel">', unsafe_allow_html=True)

    st.markdown(
        f'<div class="panel-title">{t("input_parameters", lang)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="section-title">◷ {t("composition_features", lang)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    number_control("CaO", "CaO", "%", step=1.0)
    number_control("MgO", "MgO", "%", step=1.0)
    number_control("SiO2", "SiO2", "%", step=1.0)
    number_control("Fe2O3", "Fe2O3", "%", step=1.0)
    number_control("MnO", "MnO", "%", step=1.0)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="section-title">⚗ {t("process_parameters", lang)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    number_control(t("liquid_solid_ratio", lang), "liquid_solid_ratio", "", step=1.0)
    particle_size_control()
    temperature_control()
    time_control()
    pressure_control()
    number_control(t("co2_concentration", lang), "co2_concentration", "%", step=1.0)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_header():
    lang = st.session_state.lang

    st.markdown('<div class="top-header">', unsafe_allow_html=True)
    c1, c2 = st.columns([8, 1])
    with c1:
        st.markdown(
            """
            <div class="brand">
                <div class="logo-circle">◉</div>
                <div class="brand-text">SSCSP</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        selected = st.selectbox(
            "language",
            ["简体中文", "繁體中文", "English"],
            index=["简体中文", "繁體中文", "English"].index(lang),
            label_visibility="collapsed",
        )
        st.session_state.lang = selected
    st.markdown("</div>", unsafe_allow_html=True)


def render_tabs():
    lang = st.session_state.lang
    tab1_active = "active-tab" if st.session_state.page == "prediction" else "normal-tab"
    tab2_active = "active-tab" if st.session_state.page == "lca" else "normal-tab"

    c1, c2, c3 = st.columns([1.05, 2.0, 6.95])
    with c1:
        if st.button(t("prediction_result", lang), key="tab_prediction", use_container_width=True):
            st.session_state.page = "prediction"
            st.rerun()
        st.markdown(f"<style>div[data-testid='stButton'][key='tab_prediction'] button{{}}</style>", unsafe_allow_html=True)
    with c2:
        if st.button(t("lca_cost", lang), key="tab_lca", use_container_width=True):
            st.session_state.page = "lca"
            st.rerun()
    with c3:
        st.empty()


def value_box(title, value, unit):
    display_value = "" if value is None else f"{value:.6f}"
    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-title">{title}</div>
            <div class="value-area">
                <div class="result-number">{display_value}</div>
                <div class="result-unit">{unit}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def execute_prediction(model, metadata):
    lang = st.session_state.lang
    inputs = st.session_state.inputs.copy()

    errors = validate_inputs(inputs)
    if errors:
        st.error("；".join(errors))
        return

    pred = predict_co2_uptake(model, metadata, inputs)
    theoretical = calculate_theoretical_co2(inputs)
    inventory = calculate_process_inventory(inputs, pred)
    lca_cost = calculate_lca_and_cost(inputs, pred, inventory)
    shap_df = calculate_shap_contrib(model, metadata, inputs)

    result = {
        "inputs": inputs,
        "prediction": pred,
        "theoretical": theoretical,
        "inventory": inventory,
        "lca_cost": lca_cost,
        "shap_df": shap_df,
    }
    st.session_state.prediction_result = result

    row = {
        "序号": len(st.session_state.history) + 1,
        "CaO": inputs["CaO"],
        "MgO": inputs["MgO"],
        "SiO2": inputs["SiO2"],
        "Fe2O3": inputs["Fe2O3"],
        "MnO": inputs["MnO"],
        "液固比": inputs["liquid_solid_ratio"],
        "粒径": inputs["particle_size"],
        "温度": inputs["temperature"],
        "反应时间": inputs["reaction_time"],
        "CO2分压": inputs["co2_pressure"],
        "CO2浓度": inputs["co2_concentration"],
        "CO2分压单位": "MPa",
        "CO2浓度单位": "%",
        "CO2预测": pred,
        "CO2理论": theoretical,
    }
    st.session_state.history.append(row)


def reset_inputs():
    st.session_state.inputs = DEFAULT_INPUTS.copy()
    st.session_state.prediction_result = None


def import_example():
    st.session_state.inputs = {
        "CaO": 45.0,
        "MgO": 8.0,
        "SiO2": 15.0,
        "Fe2O3": 5.0,
        "MnO": 1.0,
        "liquid_solid_ratio": 10.0,
        "particle_size": 75.0,
        "temperature": 80.0,
        "reaction_time": 24.0,
        "co2_pressure": 1.0,
        "co2_concentration": 99.0,
    }
    st.session_state.prediction_result = None


def prediction_page(model, metadata):
    lang = st.session_state.lang
    result = st.session_state.prediction_result

    c1, c2 = st.columns([1.12, 1.0])

    with c1:
        st.markdown('<div class="blue-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="block-title">{t("co2_uptake", lang)}</div>', unsafe_allow_html=True)
        value_box(
            t("predicted_value", lang),
            None if result is None else result["prediction"],
            "g-CO₂/kg-DSS",
        )
        value_box(
            t("theoretical_value", lang),
            None if result is None else result["theoretical"],
            "g-CO₂/kg-DSS",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="block-title">{t("current_sample_contribution", lang)}</div>', unsafe_allow_html=True)
        if result is not None and result["shap_df"] is not None and len(result["shap_df"]) > 0:
            st.bar_chart(
                result["shap_df"].set_index("feature")["abs_value"],
                height=250,
            )
        else:
            placeholder = pd.DataFrame(
                {
                    "feature": ["液固比", "粒径", "CaO", "温度", "时间"],
                    "abs_value": [130, 100, 230, 140, 100],
                }
            )
            st.bar_chart(placeholder.set_index("feature")["abs_value"], height=250)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="button-row">', unsafe_allow_html=True)
    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        if st.button(t("execute_prediction", lang), key="execute_prediction", use_container_width=True):
            execute_prediction(model, metadata)
            st.rerun()
    with b2:
        if st.button(t("reset_input", lang), key="reset_input", use_container_width=True):
            reset_inputs()
            st.rerun()
    with b3:
        if st.button(t("import_example", lang), key="import_example", use_container_width=True):
            import_example()
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="history-card">', unsafe_allow_html=True)
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
    else:
        hist_df = pd.DataFrame(
            columns=[
                "序号",
                "CaO",
                "MgO",
                "SiO2",
                "Fe2O3",
                "MnO",
                "液固比",
                "粒径",
                "温度",
                "反应时间",
                "CO2分压",
                "CO2浓度",
                "CO2分压单位",
                "CO2浓度单位",
                "CO2分压",
                "CO2浓度",
            ]
        )

    st.dataframe(hist_df, use_container_width=True, height=315, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def lca_page():
    lang = st.session_state.lang
    result = st.session_state.prediction_result

    if result is None:
        st.info(t("lca_notice", lang))

    c1, c2 = st.columns([1.08, 1.0])

    with c1:
        st.markdown('<div class="blue-card lca-summary-card">', unsafe_allow_html=True)
        value_box(
            t("net_carbon_reduction", lang),
            None if result is None else result["lca_cost"]["net_carbon_reduction"],
            "kg CO₂-eq/t",
        )
        value_box(
            t("process_cost", lang),
            None if result is None else result["lca_cost"]["total_cost"],
            "$/t",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="block-title">{t("cost_contribution", lang)}</div>', unsafe_allow_html=True)

        if result is not None:
            pie_data = pd.DataFrame(
                {
                    "source": ["Electricity", "Heat", "Water"],
                    "cost": [
                        result["lca_cost"]["electricity_cost"],
                        result["lca_cost"]["heat_cost"],
                        result["lca_cost"]["water_cost"],
                    ],
                }
            )
        else:
            pie_data = pd.DataFrame(
                {
                    "source": ["Electricity", "Heat", "Water"],
                    "cost": [24.7, 72.6, 2.8],
                }
            )

        st.vega_lite_chart(
            pie_data,
            {
                "width": 270,
                "height": 270,
                "mark": {"type": "arc", "innerRadius": 65, "outerRadius": 120},
                "encoding": {
                    "theta": {"field": "cost", "type": "quantitative"},
                    "color": {
                        "field": "source",
                        "type": "nominal",
                        "scale": {
                            "range": ["#f4b6b6", "#9ed9e5", "#cde5b5"]
                        },
                    },
                    "tooltip": [
                        {"field": "source", "type": "nominal"},
                        {"field": "cost", "type": "quantitative", "format": ".2f"},
                    ],
                },
            },
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="white-card impact-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="impact-title">{t("impact_contribution", lang)}</div>', unsafe_allow_html=True)

    if result is not None:
        impact_df = result["lca_cost"]["impact_contribution_df"]
    else:
        impact_df = pd.DataFrame(
            {
                "impact": [
                    "TAP", "GWP", "FETP", "METP", "TETP", "FFP", "FEP", "MEP",
                    "HTPc", "HTPnc", "IRP", "LOP", "SOP", "ODP", "PMFP",
                    "HOFP", "EOFP", "WCP"
                ],
                "Electricity": [80, 40, 70, 72, 75, 35, 82, 55, 57, 90, 93, 88, 25, 63, 78, 76, 72, 88],
                "Heat": [15, 52, 18, 16, 14, 60, 12, 35, 25, 8, 4, 8, 55, 30, 15, 18, 20, 5],
                "Water": [5, 8, 12, 12, 11, 5, 6, 10, 18, 2, 3, 4, 20, 7, 7, 6, 8, 7],
                "Fixed CO₂": [0, -125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            }
        )

    long_df = impact_df.melt(
        id_vars=["impact"],
        value_vars=["Electricity", "Heat", "Water", "Fixed CO₂"],
        var_name="source",
        value_name="percent",
    )

    st.vega_lite_chart(
        long_df,
        {
            "width": 760,
            "height": 360,
            "mark": {"type": "bar"},
            "encoding": {
                "x": {
                    "field": "impact",
                    "type": "nominal",
                    "sort": None,
                    "axis": {"labelAngle": -55},
                },
                "y": {
                    "field": "percent",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"domain": [-150, 110]},
                },
                "color": {
                    "field": "source",
                    "type": "nominal",
                    "scale": {
                        "range": ["#f4b6b6", "#9ed9e5", "#cde5b5", "#f3c37c"]
                    },
                },
                "tooltip": [
                    {"field": "impact", "type": "nominal"},
                    {"field": "source", "type": "nominal"},
                    {"field": "percent", "type": "quantitative", "format": ".2f"},
                ],
            },
        },
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    init_session_state()
    inject_css()

    try:
        model, metadata = load_model_and_metadata(MODEL_PATH, METADATA_PATH)
    except Exception as exc:
        model, metadata = None, None
        st.error(
            f"模型文件读取失败。请确认 final_lgbm_model.joblib 和 final_lgbm_metadata.json 与 app.py 放在同一目录。错误：{exc}"
        )

    render_header()

    st.markdown(
        f'<h1 class="main-title">{t("main_title", st.session_state.lang)}</h1>',
        unsafe_allow_html=True,
    )

    main_left, main_right = st.columns([1.05, 2.35], gap="large")

    with main_left:
        left_input_panel()

    with main_right:
        render_tabs()
        st.markdown('<div class="right-area">', unsafe_allow_html=True)

        if st.session_state.page == "prediction":
            prediction_page(model, metadata)
        else:
            lca_page()

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from calculations import (
    FEATURE_NAMES,
    INPUT_LIMITS,
    EXAMPLE_INPUTS,
    load_model,
    validate_inputs,
    particle_size_to_um,
    temperature_to_c,
    time_to_h,
    pressure_to_mpa,
    run_all,
)
from ui_style import TEXT, inject_css, card


st.set_page_config(
    page_title="SS-CSP | CO₂ Calculator",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "final_lgbm_model.joblib"


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def init_state():
    if "lang" not in st.session_state:
        st.session_state.lang = "zh_CN"
    if "inputs" not in st.session_state:
        st.session_state.inputs = {k: None for k in FEATURE_NAMES}
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "records" not in st.session_state:
        st.session_state.records = []
    if "errors" not in st.session_state:
        st.session_state.errors = {}


def set_example():
    st.session_state.inputs.update(EXAMPLE_INPUTS)
    st.session_state.errors = {}
    st.session_state.last_result = None


def reset_all():
    st.session_state.inputs = {k: None for k in FEATURE_NAMES}
    st.session_state.errors = {}
    st.session_state.last_result = None


def render_number_input(label, key, min_value=None, max_value=None, step=0.01, unit_label=""):
    errors = st.session_state.get("errors", {})
    label_show = f"{label} {unit_label}".strip()
    if key in errors:
        label_show = f"🔴 {label_show} — {errors[key]}"
    value = st.session_state.inputs.get(key)
    return st.number_input(
        label_show,
        min_value=None,
        max_value=None,
        value=value,
        step=step,
        key=f"input_{key}",
        placeholder=f"{INPUT_LIMITS[key][0]:g}–{INPUT_LIMITS[key][1]:g}",
    )


def collect_sidebar_inputs(t):
    st.sidebar.markdown(f"### {t['input_params']}")
    with st.sidebar:
        st.markdown(f"#### 🕘 {t['composition']}")
        for key in ["CaO", "MgO", "SiO2", "Fe2O3", "MnO"]:
            st.session_state.inputs[key] = render_number_input(key, key, step=0.01, unit_label="%")

        st.markdown("---")
        st.markdown(f"#### ⚗️ {t['process']}")

        st.session_state.inputs["W/S"] = render_number_input("液固比 / W/S", "W/S", step=0.1)

        col1, col2 = st.columns([2, 1])
        with col1:
            raw_particle = st.number_input(
                "粒径 / Particle size",
                value=st.session_state.inputs["Particle size"],
                step=0.1,
                key="input_particle_raw",
                placeholder="5.86–200 μm",
            )
        with col2:
            particle_unit = st.selectbox("单位", ["μm", "mm", "mesh"], key="particle_unit")
        st.session_state.inputs["Particle size"] = None if raw_particle is None else particle_size_to_um(raw_particle, particle_unit)

        col1, col2 = st.columns([2, 1])
        with col1:
            raw_temp = st.number_input(
                "温度 / Temperature",
                value=st.session_state.inputs["Temperature"],
                step=0.1,
                key="input_temp_raw",
                placeholder="20–100 ℃",
            )
        with col2:
            temp_unit = st.selectbox("单位", ["℃", "K"], key="temp_unit")
        st.session_state.inputs["Temperature"] = None if raw_temp is None else temperature_to_c(raw_temp, temp_unit)

        col1, col2 = st.columns([2, 1])
        with col1:
            raw_time = st.number_input(
                "反应时间 / Time",
                value=st.session_state.inputs["Time"],
                step=0.1,
                key="input_time_raw",
                placeholder="5.86–200 h",
            )
        with col2:
            time_unit = st.selectbox("单位", ["h", "min", "d"], key="time_unit")
        st.session_state.inputs["Time"] = None if raw_time is None else time_to_h(raw_time, time_unit)

        col1, col2 = st.columns([2, 1])
        with col1:
            raw_pressure = st.number_input(
                "CO₂分压 / CO₂ partial pressure",
                value=st.session_state.inputs["CO2 partial pressure"],
                step=0.01,
                key="input_pressure_raw",
                placeholder="0.01–3 MPa",
            )
        with col2:
            pressure_unit = st.selectbox("单位", ["MPa", "KPa", "Pa", "bar"], key="pressure_unit")
        st.session_state.inputs["CO2 partial pressure"] = None if raw_pressure is None else pressure_to_mpa(raw_pressure, pressure_unit)

        st.session_state.inputs["CO2 concentration"] = render_number_input(
            "CO₂浓度 / CO₂ concentration",
            "CO2 concentration",
            step=0.1,
            unit_label="%",
        )

        c1, c2, c3 = st.columns(3)
        run_clicked = c1.button(t["execute"], type="primary", use_container_width=True)
        reset_clicked = c2.button(t["reset"], use_container_width=True)
        example_clicked = c3.button(t["example"], use_container_width=True)

    return run_clicked, reset_clicked, example_clicked


def perform_prediction(model, t):
    errors = validate_inputs(st.session_state.inputs)
    st.session_state.errors = errors
    if errors:
        st.session_state.last_result = None
        st.error(t["error_title"])
        return

    result = run_all(model, st.session_state.inputs)
    st.session_state.last_result = result

    record = {name: st.session_state.inputs[name] for name in FEATURE_NAMES}
    record["CO2 sequestration/(kg-CO2/t-DSS)"] = result["CO2_sequestration_kg_per_t"]
    record["CO2 theoretical/(kg-CO2/t-DSS)"] = result["CO2_theoretical"]
    record["Carbon net/(kg CO2-eq/t)"] = result["Carbon_net"]
    record["Total cost/($/t)"] = result["costs"]["C_total"]
    st.session_state.records.append(record)


def render_prediction_page(result, t):
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### CO₂吸收量 / CO₂ uptake")
        c1, c2 = st.columns(2)
        with c1:
            card(t["co2_result"], f"{result['CO2_sequestration']:.6g}", "g·CO₂/kg·DSS")
        with c2:
            card(t["theoretical"], f"{result['CO2_theoretical']:.6g}", "g·CO₂/kg·DSS")

    with right:
        st.markdown(f"#### {t['current_shap']}")
        shap_df = result["shap_top5"].copy()
        fig = px.bar(
            shap_df.sort_values("Contribution"),
            x="Contribution",
            y="Feature",
            orientation="h",
            text="Contribution",
            labels={"Contribution": "SHAP contribution", "Feature": "Feature"},
        )
        fig.update_traces(texttemplate="%{text:.3g}", textposition="outside")
        fig.update_layout(height=315, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### {t['records']}")
    st.caption(t["copy_hint"])
    records_df = pd.DataFrame(st.session_state.records)
    st.dataframe(records_df, use_container_width=True, height=260)
    if not records_df.empty:
        st.download_button(
            t["download_csv"],
            data=records_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="prediction_records.csv",
            mime="text/csv",
        )


def render_lca_cost_page(result, t):
    st.warning(t["warning_formula"])
    left, right = st.columns([1, 1])
    with left:
        card(t["net_carbon"], f"{result['Carbon_net']:.6g}", "kg CO₂-eq/t")
        st.write("")
        card(t["total_cost"], f"{result['costs']['C_total']:.6g}", "$/t")

    with right:
        st.markdown(f"#### {t['cost_share']}")
        cost_df = pd.DataFrame({
            "Cost item": ["Electricity", "Heat", "Water"],
            "Cost": [
                result["costs"]["C_elec"],
                result["costs"]["C_heat"],
                result["costs"]["C_water"],
            ],
        })
        fig = px.pie(cost_df, names="Cost item", values="Cost", hole=0.55)
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### {t['impact_share']}")
    lca = result["lca"].copy()
    plot_df = lca.melt(
        id_vars=["Impact"],
        value_vars=["Electricity_%", "Heat_%", "Water_%", "Fixed_CO2_%"],
        var_name="Contribution source",
        value_name="Contribution (%)",
    )
    plot_df["Contribution source"] = plot_df["Contribution source"].str.replace("_%", "", regex=False)
    fig = px.bar(
        plot_df,
        x="Impact",
        y="Contribution (%)",
        color="Contribution source",
        barmode="relative",
        hover_data={"Contribution (%)": ":.4f"},
    )
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("查看 LCA 明细 / LCA details"):
        st.dataframe(lca, use_container_width=True, height=360)


def main():
    init_state()
    model = get_model()

    lang_label_to_key = {"简体中文": "zh_CN", "繁體中文": "zh_TW", "English": "en"}
    reverse = {v: k for k, v in lang_label_to_key.items()}

    top_left, top_right = st.columns([4, 1])
    with top_right:
        selected = st.selectbox(
            "",
            list(lang_label_to_key.keys()),
            index=list(lang_label_to_key.keys()).index(reverse[st.session_state.lang]),
            label_visibility="collapsed",
        )
        st.session_state.lang = lang_label_to_key[selected]

    t = TEXT[st.session_state.lang]
    top_left.title(t["title"])

    run_clicked, reset_clicked, example_clicked = collect_sidebar_inputs(t)

    if reset_clicked:
        reset_all()
        st.rerun()
    if example_clicked:
        set_example()
        st.rerun()
    if run_clicked:
        perform_prediction(model, t)

    result = st.session_state.last_result

    tab1, tab2 = st.tabs([t["prediction_page"], t["lca_page"]])
    if result is None:
        with tab1:
            st.info("请在左侧输入参数后点击“执行预测”。 / Enter parameters on the left and run prediction.")
        with tab2:
            st.info("完成一次有效预测后，将自动显示生命周期评价及成本分析。 / LCA and cost results appear after a valid prediction.")
        return

    with tab1:
        render_prediction_page(result, t)
    with tab2:
        render_lca_cost_page(result, t)


if __name__ == "__main__":
    main()

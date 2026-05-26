from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from calculations import (
    EXAMPLE_INPUTS,
    IMPACT_FACTORS,
    WORD_EXAMPLE_CO2,
    convert_and_validate,
    full_calculation,
    load_model,
    record_row,
)
from ui_style import APP_TITLE, CSS, TEXT

st.set_page_config(page_title="SSCSP CO2 Calculator", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

COMPOSITION_KEYS = ["CaO", "MgO", "SiO2", "Fe2O3", "MnO"]
PROCESS_KEYS = ["W/S", "Particle_size", "Temperature", "Time", "CO2_partial_pressure", "CO2_concentration"]
LABELS = {
    "CaO": "CaO %",
    "MgO": "MgO %",
    "SiO2": "SiO₂ %",
    "Fe2O3": "Fe₂O₃ %",
    "MnO": "MnO %",
    "W/S": "液固比 / W/S",
    "Particle_size": "粒径 / Particle size",
    "Temperature": "温度 / Temperature",
    "Time": "反应时间 / Time",
    "CO2_partial_pressure": "CO₂分压 / CO2 partial pressure",
    "CO2_concentration": "CO₂浓度 / CO2 concentration %",
}
PLACEHOLDERS = {
    "CaO": "20.91–65",
    "MgO": "1.65–13.5",
    "SiO2": "4.35–33.74",
    "Fe2O3": "0.25–42.98",
    "MnO": "0–6.8",
    "W/S": "1–50",
    "Particle_size": "5.86–200",
    "Temperature": "20–100",
    "Time": "5.86–200",
    "CO2_partial_pressure": "0.01–3",
    "CO2_concentration": "9–100",
}


@st.cache_resource
def get_model():
    return load_model()


def init_state() -> None:
    for key in COMPOSITION_KEYS + PROCESS_KEYS:
        st.session_state.setdefault(f"v_{key}", "")
    st.session_state.setdefault("u_Particle_size", "μm")
    st.session_state.setdefault("u_Temperature", "℃")
    st.session_state.setdefault("u_Time", "h")
    st.session_state.setdefault("u_CO2_partial_pressure", "MPa")
    st.session_state.setdefault("records", [])
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("errors", {})
    st.session_state.setdefault("lang", "简体中文")


def raw_inputs() -> dict:
    return {key: st.session_state.get(f"v_{key}", "") for key in COMPOSITION_KEYS + PROCESS_KEYS}


def units() -> dict:
    return {
        "Particle_size": st.session_state.get("u_Particle_size", "μm"),
        "Temperature": st.session_state.get("u_Temperature", "℃"),
        "Time": st.session_state.get("u_Time", "h"),
        "CO2_partial_pressure": st.session_state.get("u_CO2_partial_pressure", "MPa"),
    }


def import_example() -> None:
    for key, value in EXAMPLE_INPUTS.items():
        st.session_state[f"v_{key}"] = str(value)
    st.session_state["u_Particle_size"] = "μm"
    st.session_state["u_Temperature"] = "℃"
    st.session_state["u_Time"] = "h"
    st.session_state["u_CO2_partial_pressure"] = "MPa"
    st.session_state["errors"] = {}
    st.session_state["result"] = None


def reset_all() -> None:
    for key in COMPOSITION_KEYS + PROCESS_KEYS:
        st.session_state[f"v_{key}"] = ""
    st.session_state["u_Particle_size"] = "μm"
    st.session_state["u_Temperature"] = "℃"
    st.session_state["u_Time"] = "h"
    st.session_state["u_CO2_partial_pressure"] = "MPa"
    st.session_state["result"] = None
    st.session_state["errors"] = {}
    st.session_state["records"] = []


def run_prediction(model) -> None:
    base, errors = convert_and_validate(raw_inputs(), units())
    st.session_state["errors"] = errors
    if errors:
        st.session_state["result"] = None
        return
    result = full_calculation(model, base)
    st.session_state["result"] = result
    st.session_state["records"].append(record_row(result))


def input_row(key: str, unit_options: list[str] | None = None) -> None:
    errors = st.session_state.get("errors", {})
    c1, c2 = st.columns([0.72, 0.28]) if unit_options else (st.container(), None)
    with c1:
        st.text_input(LABELS[key], key=f"v_{key}", placeholder=PLACEHOLDERS[key], label_visibility="visible")
    if unit_options:
        with c2:
            st.selectbox("单位", unit_options, key=f"u_{key}", label_visibility="visible")
    st.markdown(f"<div class='small-hint'>范围：{PLACEHOLDERS[key]}</div>", unsafe_allow_html=True)
    if key in errors:
        st.markdown(f"<div class='err'>⚠ {errors[key]}</div>", unsafe_allow_html=True)


def render_input_panel(t) -> None:
    st.markdown(f"<div class='left-title'>{t['input']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>◷ {t['composition']}</div>", unsafe_allow_html=True)
    for key in COMPOSITION_KEYS:
        input_row(key)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>⚗ {t['process']}</div>", unsafe_allow_html=True)
    input_row("W/S")
    input_row("Particle_size", ["μm", "mm", "mesh"])
    input_row("Temperature", ["℃", "K"])
    input_row("Time", ["h", "min", "d"])
    input_row("CO2_partial_pressure", ["MPa", "kPa", "Pa", "bar"])
    input_row("CO2_concentration")


def metric_box(title: str, value, unit: str, decimals: int = 4) -> None:
    text = "—" if value is None else f"{float(value):,.{decimals}f}"
    st.markdown(
        f"<div class='result-label'>{title}</div><div class='result-box'><span>{text}</span><span class='unit'>{unit}</span></div>",
        unsafe_allow_html=True,
    )


def render_prediction_tab(t, model) -> None:
    result = st.session_state.get("result")
    left, right = st.columns([1.1, 1.0], gap="large")
    with left:
        st.markdown("<div class='blue-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-label'>{t['co2']}</div>", unsafe_allow_html=True)
        metric_box(t["pred"], result["CO2_sequestration"] if result else None, "g-CO₂/kg-DSS")
        metric_box(t["theory"], result["CO2_theoretical"] if result else None, "g-CO₂/kg-DSS")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-label'>{t['shap']}</div>", unsafe_allow_html=True)
        if result is None or result["shap_df"].empty:
            st.markdown(f"<div class='info-box'>{t['empty_pred']}</div>", unsafe_allow_html=True)
        else:
            fig = px.bar(
                result["shap_df"],
                x="value", y="display", orientation="h",
                text=result["shap_df"]["value"].map(lambda x: f"{x:.2f}"),
                labels={"value": "SHAP contribution", "display": ""},
            )
            fig.update_layout(height=285, margin=dict(l=10, r=15, t=10, b=10), showlegend=False)
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button(t["run"], type="primary", use_container_width=True):
            run_prediction(model)
            st.rerun()
    with b2:
        if st.button(t["reset"], use_container_width=True):
            reset_all()
            st.rerun()
    with b3:
        if st.button(t["example"], use_container_width=True):
            import_example()
            st.rerun()

    if st.session_state.get("errors"):
        st.error(t["bad_input"])

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-label'>{t['record']}</div>", unsafe_allow_html=True)
    records = st.session_state.get("records", [])
    if records:
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True, height=300, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        tsv = df.to_csv(index=False, sep="\t")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(t["download"], csv, "prediction_records.csv", "text/csv", use_container_width=True)
        with c2:
            with st.expander(t["copy"]):
                st.text_area("TSV", tsv, height=160, label_visibility="collapsed")
    else:
        st.dataframe(pd.DataFrame(columns=record_row({
            "inputs": EXAMPLE_INPUTS,
            "CO2_sequestration": 0,
            "CO2_theoretical": 0,
            "carbon": {"Carbon_net": 0},
            "cost": {"C_total": 0},
        }).keys()), use_container_width=True, height=250, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_lca_tab(t) -> None:
    result = st.session_state.get("result")
    if result is None:
        st.markdown(f"<div class='info-box'>{t['empty_lca']}</div>", unsafe_allow_html=True)
        return

    left, right = st.columns([1.0, 1.12], gap="large")
    with left:
        st.markdown("<div class='blue-card'>", unsafe_allow_html=True)
        metric_box(t["net_carbon"], result["carbon"]["Carbon_net"], "kg CO₂-eq/t")
        metric_box(t["cost"], result["cost"]["C_total"], "$/t")
        with st.expander("过程清单 / Process inventory"):
            inv_df = pd.DataFrame([result["inventory"]]).T.reset_index()
            inv_df.columns = ["variable", "value"]
            st.dataframe(inv_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-label'>{t['cost_pie']}</div>", unsafe_allow_html=True)
        cost = result["cost"]
        cost_df = pd.DataFrame({
            "Source": ["Electricity", "Heat", "Water"],
            "Cost": [cost["C_elec"], cost["C_heat"], cost["C_water"]],
        })
        fig = px.pie(cost_df, names="Source", values="Cost", hole=0.5)
        fig.update_traces(textinfo="percent+label", textposition="outside")
        fig.update_layout(height=325, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-label'>{t['impact']}</div>", unsafe_allow_html=True)
    if len(IMPACT_FACTORS) < 18:
        st.markdown("<div class='warn-box'>Word 因子表目前只有 17 项；如果要显示截图里的 WCP，请补充 WCP 的电/热/水/CO₂ 固存因子。</div>", unsafe_allow_html=True)
    lca_df = result["lca_df"].copy()
    plot_df = lca_df[["impact", "Electricity_%", "Heat_%", "Water_%", "Fixed_CO2_%"]].melt(
        id_vars="impact", var_name="source", value_name="contribution"
    )
    plot_df["source"] = plot_df["source"].map({
        "Electricity_%": "Electricity", "Heat_%": "Heat", "Water_%": "Water", "Fixed_CO2_%": "Fixed CO₂"
    })
    fig = px.bar(plot_df, x="impact", y="contribution", color="source", barmode="relative")
    fig.update_layout(height=470, yaxis_title="Contribution (%)", xaxis_title="", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("LCA 明细表 / LCA data table"):
        st.dataframe(lca_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    init_state()
    model = get_model()

    top_l, top_r = st.columns([0.75, 0.25])
    with top_l:
        st.markdown("<div class='logo-row'><div class='logo-mark'></div><div class='logo-text'>SSCSP</div></div>", unsafe_allow_html=True)
    with top_r:
        st.selectbox("Language", ["简体中文", "繁體中文", "English"], key="lang", label_visibility="collapsed")

    t = TEXT[st.session_state["lang"]]
    st.markdown(f"<div class='main-title'>{APP_TITLE[st.session_state['lang']]}</div>", unsafe_allow_html=True)

    with st.expander(t["check"], expanded=False):
        check = full_calculation(model, EXAMPLE_INPUTS, co2_override=WORD_EXAMPLE_CO2)
        cost = check["cost"]
        st.write(
            f"按 Word 示例 CS={WORD_EXAMPLE_CO2:.6f} 代入，成本占比："
            f"电 {cost['electricity_cost_ratio']:.1f}% / 热 {cost['heat_cost_ratio']:.1f}% / 水 {cost['water_cost_ratio']:.1f}%"
        )
        st.markdown(f"<div class='warn-box'>{t['warning']}</div>", unsafe_allow_html=True)

    left, right = st.columns([0.30, 0.70], gap="large")
    with left:
        st.markdown("<div class='left-panel'>", unsafe_allow_html=True)
        render_input_panel(t)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        tab1, tab2 = st.tabs([t["tab_pred"], t["tab_lca"]])
        with tab1:
            render_prediction_tab(t, model)
        with tab2:
            render_lca_tab(t)


if __name__ == "__main__":
    main()

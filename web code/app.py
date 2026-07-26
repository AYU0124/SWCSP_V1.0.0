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
from ui_style import (
    HEADER_CSS,
    LANG_LABEL_TO_KEY,
    LCA_COLOR_MAP,
    TEXT,
    get_lca_detail_display_df,
    inject_css,
    resolve_asset_path,
    svg_data_uri,
)


st.set_page_config(
    page_title="SS-CSP | CO₂ Calculator",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "final_lgbm_model.joblib"

RECORD_OUTPUT_KEY = "CO2_sequestration"
LEGACY_RECORD_OUTPUT_KEYS = ("CO2 sequestration/(g-CO2/kg-DSS)",)


def get_record_display_schema(t: dict) -> tuple[dict[str, str], str, list[str], dict[str, str]]:
    """Return current-language labels and rename map for prediction records."""
    record_columns = t["record_columns"]
    column_labels = {name: record_columns[name] for name in FEATURE_NAMES}
    output_column = t["output_column"]
    display_order = [column_labels[name] for name in FEATURE_NAMES] + [output_column]
    rename_map = {**column_labels, RECORD_OUTPUT_KEY: output_column}
    return column_labels, output_column, display_order, rename_map


def canonicalize_record_columns(records_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize old display-label records to internal canonical column names."""
    reverse_display_columns = {}

    for lang_text in TEXT.values():
        for feature_name, display_name in lang_text.get("record_columns", {}).items():
            reverse_display_columns[display_name] = feature_name

        output_column = lang_text.get("output_column")
        if output_column:
            reverse_display_columns[output_column] = RECORD_OUTPUT_KEY

    for legacy_key in LEGACY_RECORD_OUTPUT_KEYS:
        reverse_display_columns[legacy_key] = RECORD_OUTPUT_KEY

    return records_df.rename(columns=reverse_display_columns)


def render_fixed_header():
    logo_uri = svg_data_uri(resolve_asset_path("main_logo.svg"))
    if logo_uri:
        logo_html = f'<img class="sscsp-brand-logo" src="{logo_uri}" alt="SSCSP logo" />'
    else:
        logo_html = '<div style="width:42px;height:42px;border-radius:50%;background:#1F5BD8;"></div>'

    st.markdown(HEADER_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sscsp-fixed-header">
            <div class="sscsp-brand">
                {logo_html}
                <div class="sscsp-brand-text">SSCSP</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_section_title(icon_path: Path | None, text_label: str):
    icon_uri = svg_data_uri(icon_path)
    if icon_uri:
        icon_html = f'<img src="{icon_uri}" alt="" />'
    else:
        icon_html = '<span style="width:22px;height:22px;display:inline-block;"></span>'

    st.markdown(
        f"""
        <div class="sidebar-section-title">
            {icon_html}
            <span>{text_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_main_title(text_label: str):
    st.markdown(
        f"""<div class="sidebar-main-title">{text_label}</div>""",
        unsafe_allow_html=True,
    )


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


def get_current_text() -> dict:
    return TEXT.get(st.session_state.get("lang", "zh_CN"), TEXT["zh_CN"])


def is_empty_number(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def get_display_limits(feature_key: str, unit_label: str = "") -> tuple[float, float]:
    """Return valid input limits in the unit currently shown to the user."""
    lo, hi = INPUT_LIMITS[feature_key]

    if feature_key == "Particle size":
        if unit_label == "mm":
            return lo / 1000.0, hi / 1000.0
        if unit_label == "mesh":
            # particle_size_um = 15000 / mesh, so the display range is inverted.
            return 15000.0 / hi, 15000.0 / lo
        return lo, hi

    if feature_key == "Temperature":
        if unit_label == "K":
            return lo + 273.15, hi + 273.15
        return lo, hi

    if feature_key == "Time":
        if unit_label == "min":
            return lo * 60.0, hi * 60.0
        if unit_label == "d":
            return lo / 24.0, hi / 24.0
        return lo, hi

    if feature_key == "CO2 partial pressure":
        if unit_label == "KPa":
            return lo * 1000.0, hi * 1000.0
        if unit_label == "Pa":
            return lo * 1_000_000.0, hi * 1_000_000.0
        if unit_label == "bar":
            return lo * 10.0, hi * 10.0
        return lo, hi

    return lo, hi


def format_limit_placeholder(feature_key: str, unit_label: str = "") -> str:
    """Return only the numeric valid range for the current display unit.

    This is used as number_input placeholder text. Streamlit shows it only when
    the input value is empty; once the user enters a value, the placeholder is hidden.
    """
    lo, hi = get_display_limits(feature_key, unit_label)
    return f"{lo:g}–{hi:g}"


def format_range_error(t: dict, feature_key: str, unit_label: str = "") -> str:
    lo, hi = get_display_limits(feature_key, unit_label)
    return t.get("range_error", "out_of_range: {lo:g}–{hi:g}").format(lo=lo, hi=hi)


def update_existing_error_for_direct_input(feature_key: str):
    errors = st.session_state.setdefault("errors", {})
    if feature_key not in errors:
        return

    t = get_current_text()
    value = st.session_state.get(f"input_{feature_key}")
    lo, hi = INPUT_LIMITS[feature_key]

    if is_empty_number(value):
        errors[feature_key] = t.get("missing_error", "missing")
    elif value < lo or value > hi:
        errors[feature_key] = format_range_error(t, feature_key)
    else:
        errors.pop(feature_key, None)


def update_existing_error_for_converted_input(
    feature_key: str,
    raw_key: str,
    unit_key: str,
    default_unit: str,
    converter,
    error_message: str | None = None,
):
    errors = st.session_state.setdefault("errors", {})
    if feature_key not in errors:
        return

    t = get_current_text()
    raw_value = st.session_state.get(raw_key)
    unit_value = st.session_state.get(unit_key, default_unit)

    if is_empty_number(raw_value):
        st.session_state.inputs[feature_key] = None
        errors[feature_key] = t.get("missing_error", "missing")
        return

    try:
        converted_value = converter(raw_value, unit_value)
    except ValueError:
        st.session_state.inputs[feature_key] = None
        errors[feature_key] = error_message or t.get("missing_error", "missing")
        return

    st.session_state.inputs[feature_key] = converted_value
    lo, hi = INPUT_LIMITS[feature_key]
    if converted_value < lo or converted_value > hi:
        errors[feature_key] = format_range_error(t, feature_key, unit_value)
    else:
        errors.pop(feature_key, None)


def clear_error_callback(
    feature_key: str,
    raw_key: str | None = None,
    unit_key: str | None = None,
    default_unit: str | None = None,
    converter=None,
    error_message: str | None = None,
):
    """Refresh an existing validation error instead of blindly clearing it.

    This keeps red errors visible while the current value is still invalid,
    and removes them only after the value becomes valid.
    """
    if raw_key is None or unit_key is None or default_unit is None or converter is None:
        update_existing_error_for_direct_input(feature_key)
        return

    update_existing_error_for_converted_input(
        feature_key,
        raw_key,
        unit_key,
        default_unit,
        converter,
        error_message,
    )


def set_example():
    for key in ["CaO", "MgO", "SiO2", "Fe2O3", "MnO", "W/S", "CO2 concentration"]:
        st.session_state[f"input_{key}"] = EXAMPLE_INPUTS[key]

    st.session_state["input_particle_raw"] = EXAMPLE_INPUTS["Particle size"]
    st.session_state["particle_unit"] = "μm"

    st.session_state["input_temp_raw"] = EXAMPLE_INPUTS["Temperature"]
    st.session_state["temp_unit"] = "℃"

    st.session_state["input_time_raw"] = EXAMPLE_INPUTS["Time"]
    st.session_state["time_unit"] = "h"

    st.session_state["input_pressure_raw"] = EXAMPLE_INPUTS["CO2 partial pressure"]
    st.session_state["pressure_unit"] = "MPa"

    st.session_state.inputs.update(EXAMPLE_INPUTS)
    st.session_state.errors = {}
    st.session_state.last_result = None


def reset_all():
    widget_keys = [
        *(f"input_{k}" for k in ["CaO", "MgO", "SiO2", "Fe2O3", "MnO", "W/S", "CO2 concentration"]),
        "input_particle_raw",
        "input_temp_raw",
        "input_time_raw",
        "input_pressure_raw",
    ]
    for widget_key in widget_keys:
        st.session_state[widget_key] = None

    st.session_state.inputs = {k: None for k in FEATURE_NAMES}
    st.session_state.errors = {}
    st.session_state.last_result = None


def chem_label_html(label: str) -> str:
    mapping = {
        "CaO": "CaO",
        "MgO": "MgO",
        "SiO2": "SiO<sub>2</sub>",
        "Fe2O3": "Fe<sub>2</sub>O<sub>3</sub>",
        "MnO": "MnO",
        "CO2 concentration": "CO<sub>2</sub> concentration",
    }
    return mapping.get(label, label)


def render_number_input(label, key, min_value=None, max_value=None, step=0.01, unit_label=""):
    errors = st.session_state.get("errors", {})

    label_html = chem_label_html(label)
    label_show = f"{label_html} {unit_label}".strip()

    if key in errors:
        label_show = f"🔴 {label_show} — {errors[key]}"

    st.markdown(
        f"<div class='custom-input-label'>{label_show}</div>",
        unsafe_allow_html=True,
    )

    value = st.session_state.inputs.get(key)

    return st.number_input(
        "",
        min_value=None,
        max_value=None,
        value=value,
        step=step,
        key=f"input_{key}",
        placeholder=f"{INPUT_LIMITS[key][0]:g}–{INPUT_LIMITS[key][1]:g}",
        label_visibility="collapsed",
        on_change=clear_error_callback,
        args=(key,)
    )

def get_inline_error_label(label: str, feature_key: str) -> str:
    errors = st.session_state.get("errors", {})
    if feature_key in errors:
        return f"🔴 {label} — {errors[feature_key]}"
    return label


def get_validation_messages(t: dict) -> dict[str, str]:
    return {
        "missing": t.get("missing_error", "missing"),
        "out_of_range": t.get("range_error", "out_of_range: {lo:g}–{hi:g}"),
    }


def get_conversion_errors(t: dict) -> dict[str, str]:
    mesh_error = t.get("mesh_error", "mesh_value_must_be_greater_than_0")
    errors = st.session_state.get("errors", {})
    if errors.get("Particle size") == mesh_error:
        return {"Particle size": mesh_error}
    return {}


def rewrite_converted_errors_for_display(errors: dict[str, str], t: dict) -> dict[str, str]:
    """Show converted-input range errors in the unit selected by the user."""
    converted_inputs = [
        ("Particle size", "input_particle_raw", "particle_unit", "μm", particle_size_to_um, t.get("mesh_error", "mesh_value_must_be_greater_than_0")),
        ("Temperature", "input_temp_raw", "temp_unit", "℃", temperature_to_c, None),
        ("Time", "input_time_raw", "time_unit", "h", time_to_h, None),
        ("CO2 partial pressure", "input_pressure_raw", "pressure_unit", "MPa", pressure_to_mpa, None),
    ]

    display_errors = dict(errors)

    for feature_key, raw_key, unit_key, default_unit, converter, conversion_error in converted_inputs:
        if feature_key not in display_errors:
            continue

        raw_value = st.session_state.get(raw_key)
        unit_value = st.session_state.get(unit_key, default_unit)

        if is_empty_number(raw_value):
            display_errors[feature_key] = t.get("missing_error", "missing")
            continue

        try:
            converted_value = converter(raw_value, unit_value)
        except ValueError:
            display_errors[feature_key] = conversion_error or t.get("missing_error", "missing")
            continue

        lo, hi = INPUT_LIMITS[feature_key]
        if converted_value < lo or converted_value > hi:
            display_errors[feature_key] = format_range_error(t, feature_key, unit_value)

    return display_errors


def sync_converted_input(
    feature_key: str,
    raw_key: str,
    unit_key: str,
    default_unit: str,
    converter,
    error_message: str | None = None,
):
    raw_value = st.session_state.get(raw_key)
    unit_value = st.session_state.get(unit_key, default_unit)

    if raw_value is None:
        st.session_state.inputs[feature_key] = None
        if error_message and st.session_state.errors.get(feature_key) == error_message:
            st.session_state.errors.pop(feature_key, None)
        return

    try:
        st.session_state.inputs[feature_key] = converter(raw_value, unit_value)
        if error_message and st.session_state.errors.get(feature_key) == error_message:
            st.session_state.errors.pop(feature_key, None)
    except ValueError:
        st.session_state.inputs[feature_key] = None
        if error_message:
            st.session_state.errors[feature_key] = error_message


def collect_sidebar_inputs(t):
    sidebar_main_title(t["input_params"])

    with st.container(key="composition_block"):
        sidebar_section_title(resolve_asset_path("composition_icon.svg", "1.svg"), t["composition"])
        for key in ["CaO", "MgO", "SiO2", "Fe2O3", "MnO"]:
            st.session_state.inputs[key] = render_number_input(key, key, step=0.01, unit_label="%")

    with st.container(key="process_block"):
        sidebar_section_title(resolve_asset_path("process_icon.svg", "2.svg"), t["process"])

        st.session_state.inputs["W/S"] = render_number_input(t["ws_label"], "W/S", step=0.1)

        particle_error = t.get("mesh_error", "mesh_value_must_be_greater_than_0")
        sync_converted_input(
            "Particle size",
            "input_particle_raw",
            "particle_unit",
            "μm",
            particle_size_to_um,
            particle_error,
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            raw_particle = st.number_input(
                get_inline_error_label(t["particle_label"], "Particle size"),
                value=st.session_state.get("input_particle_raw"),
                step=0.1,
                key="input_particle_raw",
                placeholder=format_limit_placeholder("Particle size", st.session_state.get("particle_unit", "μm")),
                on_change=clear_error_callback,
                args=("Particle size", "input_particle_raw", "particle_unit", "μm", particle_size_to_um, particle_error)
            )
        with col2:
            particle_unit = st.selectbox(t["unit_label"], ["μm", "mm", "mesh"], key="particle_unit", on_change=clear_error_callback,
                args=("Particle size", "input_particle_raw", "particle_unit", "μm", particle_size_to_um, particle_error))
        sync_converted_input(
            "Particle size",
            "input_particle_raw",
            "particle_unit",
            "μm",
            particle_size_to_um,
            particle_error,
        )

        sync_converted_input(
            "Temperature",
            "input_temp_raw",
            "temp_unit",
            "℃",
            temperature_to_c,
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            raw_temp = st.number_input(
                get_inline_error_label(t["temperature_label"], "Temperature"),
                value=st.session_state.get("input_temp_raw"),
                step=0.1,
                key="input_temp_raw",
                placeholder=format_limit_placeholder("Temperature", st.session_state.get("temp_unit", "℃")),
                on_change=clear_error_callback, args=("Temperature", "input_temp_raw", "temp_unit", "℃", temperature_to_c, None)
            )
        with col2:
            temp_unit = st.selectbox(t["unit_label"], ["℃", "K"], key="temp_unit", on_change=clear_error_callback, args=("Temperature", "input_temp_raw", "temp_unit", "℃", temperature_to_c, None))
        sync_converted_input(
            "Temperature",
            "input_temp_raw",
            "temp_unit",
            "℃",
            temperature_to_c,
        )

        sync_converted_input(
            "Time",
            "input_time_raw",
            "time_unit",
            "h",
            time_to_h,
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            raw_time = st.number_input(
                get_inline_error_label(t["time_label"], "Time"),
                value=st.session_state.get("input_time_raw"),
                step=0.1,
                key="input_time_raw",
                placeholder=format_limit_placeholder("Time", st.session_state.get("time_unit", "h")),
                on_change=clear_error_callback, args=("Time", "input_time_raw", "time_unit", "h", time_to_h, None)
            )
        with col2:
            time_unit = st.selectbox(t["unit_label"], ["h", "min", "d"], key="time_unit", on_change=clear_error_callback, args=("Time", "input_time_raw", "time_unit", "h", time_to_h, None))
        sync_converted_input(
            "Time",
            "input_time_raw",
            "time_unit",
            "h",
            time_to_h,
        )

        sync_converted_input(
            "CO2 partial pressure",
            "input_pressure_raw",
            "pressure_unit",
            "MPa",
            pressure_to_mpa,
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            raw_pressure = st.number_input(
                get_inline_error_label(t["pressure_label"], "CO2 partial pressure"),
                value=st.session_state.get("input_pressure_raw"),
                step=0.01,
                key="input_pressure_raw",
                placeholder=format_limit_placeholder("CO2 partial pressure", st.session_state.get("pressure_unit", "MPa")),
                on_change=clear_error_callback, args=("CO2 partial pressure", "input_pressure_raw", "pressure_unit", "MPa", pressure_to_mpa, None)
            )
        with col2:
            pressure_unit = st.selectbox(t["unit_label"], ["MPa", "KPa", "Pa", "bar"], key="pressure_unit", on_change=clear_error_callback, args=("CO2 partial pressure", "input_pressure_raw", "pressure_unit", "MPa", pressure_to_mpa, None))
        sync_converted_input(
            "CO2 partial pressure",
            "input_pressure_raw",
            "pressure_unit",
            "MPa",
            pressure_to_mpa,
        )

        st.session_state.inputs["CO2 concentration"] = render_number_input(
            t["concentration_label"],
            "CO2 concentration",
            step=0.1,
            unit_label="%",
        )

    with st.container(key="action_button_area"):
        c1, c2, c3 = st.columns(3, gap=None)

        with c1:
            run_clicked = st.button(
                t["run_label"],
                type="primary",
                use_container_width=False,
                key="run_button",
            )

        with c2:
            st.button(
                t["reset_label"],
                use_container_width=False,
                on_click=reset_all,
                key="reset_button",
            )

        with c3:
            st.button(
                t["example_label"],
                use_container_width=False,
                on_click=set_example,
                key="example_button",
            )

    return run_clicked


def perform_prediction(model, t):
    errors = validate_inputs(
        st.session_state.inputs,
        messages=get_validation_messages(t),
        existing_errors=get_conversion_errors(t),
    )
    errors = rewrite_converted_errors_for_display(errors, t)
    st.session_state.errors = errors

    if errors:
        st.session_state.last_result = None
        st.rerun()
        return

    result = run_all(model, st.session_state.inputs)
    st.session_state.last_result = result

    record = {
        name: st.session_state.inputs[name]
        for name in FEATURE_NAMES
    }
    record[RECORD_OUTPUT_KEY] = result["CO2_sequestration_kg_per_t"]

    st.session_state.records.append(record)



def render_prediction_page(result, t):
    pred_value = f"{result['CO2_sequestration']:.4f}"
    theory_value = f"{result['CO2_theoretical']:.3f}"
    co2_warnings = []

    if result["CO2_sequestration"] < 0:
        co2_warnings.append(
            t.get(
                "co2_negative_warning",
                "The predicted value is below 0 and may not be physically meaningful.",
            )
        )

    if result["CO2_sequestration"] > result["CO2_theoretical"]:
        co2_warnings.append(
            t.get(
                "co2_above_theory_warning",
                "The predicted value exceeds the theoretical CO₂ sequestration capacity. Please interpret this result with caution.",
            )
        )

    co2_html = (
        f'<div class="co2-result-wrap">'
        f'<div class="co2-result-outer">'
        f'<div class="co2-result-title">{t["co2_card_title"]}</div>'
        f'<div class="co2-result-middle">'

        f'<div class="co2-metric-block">'
        f'<div class="co2-metric-title">{t["predicted_value"]}</div>'
        f'<div class="co2-value-card">'
        f'<div class="co2-big-value">{pred_value}</div>'
        f'<div class="co2-unit">g-CO<sub>2</sub>/kg-DSS</div>'
        f'</div>'
        f'</div>'

        f'<div class="co2-metric-block">'
        f'<div class="co2-metric-title">'
        f'{t["theoretical_value"]}'
        f'<span class="theory-tip" data-tooltip="{t["theory_tooltip"]}"></span>'
        f'</div>'
        f'<div class="co2-value-card">'
        f'<div class="co2-big-value">{theory_value}</div>'
        f'<div class="co2-unit">g-CO<sub>2</sub>/kg-DSS</div>'
        f'</div>'
        f'</div>'

        f'</div>'
        f'</div>'
        f'</div>'
    )

    left_col, right_col = st.columns([0.38, 0.62], gap="small")

    with left_col:
        st.markdown(co2_html, unsafe_allow_html=True)

        warning_separator = "；" if st.session_state.lang in ("zh_CN", "zh_TW") else "; "

        if co2_warnings:
            st.markdown(
                f"""
                <div style="
                    margin-left: 24px;
                    width: max-content;
                    max-width: none;
                    background: #FFFBE6;
                    color: #9A6700;
                    border-radius: 8px;
                    padding: 16px 20px;
                    font-size: 16px;
                    line-height: 1.6;
                    box-sizing: border-box;
                    margin-top: 12px;
                    white-space: nowrap;
                    overflow: visible;
                ">
                    {warning_separator.join(co2_warnings)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        with st.container(key="shap_card_box"):
            st.markdown(
                f"<div class='shap-card-title'>{t['current_shap']}</div>",
                unsafe_allow_html=True,
            )

            shap_df = result.get("shap_top6")
            if shap_df is None:
                shap_df = result.get("shap_top5")

            if shap_df is None:
                st.warning(t.get("no_shap", "No SHAP contribution data available."))
            else:
                shap_df = shap_df.copy()
                shap_df["Feature_display"] = shap_df["Feature"].replace(t["feature_label_map"])

                c_min = shap_df["Contribution"].min()
                c_max = shap_df["Contribution"].max()
                margin_ext = (c_max - c_min) * 0.20 if c_max != c_min else 10
                x_range = [c_min - margin_ext, c_max + margin_ext]

                shap_contribution_label = t.get("shap_contribution", "SHAP contribution")

                fig = px.bar(
                    shap_df.sort_values("Contribution"),
                    x="Contribution",
                    y="Feature_display",
                    orientation="h",
                    text="Contribution",
                    labels={
                        "Contribution": shap_contribution_label,
                        "Feature_display": "",
                    },
                )

                fig.update_traces(
                    texttemplate="%{text:.3g}",
                    textposition="outside",
                    textfont=dict(
                        size=15,
                        color="#6B7280",
                        family="Arial, sans-serif",
                    ),
                    cliponaxis=False,
                    hovertemplate=f"<b>%{{y}}</b><br>{shap_contribution_label}: %{{x:.3g}}<extra></extra>"
                )

                fig.update_layout(
                    autosize=True,
                    height=410,
                    margin=dict(l=10, r=10, t=10, b=55),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    hovermode="closest",

                    xaxis=dict(
                        title=f"<b>{shap_contribution_label}</b>",
                        title_font=dict(size=18, color="#111827", family="Microsoft YaHei, SimHei, Arial, sans-serif"),
                        tickfont=dict(size=15, color="#111827", family="Microsoft YaHei, SimHei, Arial, sans-serif"),
                        range=x_range,
                        zeroline=True,
                        zerolinecolor="#D1D5DB",
                        showgrid=True,
                        gridcolor="#F3F4F6",
                        automargin=True,
                    ),

                    yaxis=dict(
                        title="",
                        tickfont=dict(size=15, color="#111827", family="Microsoft YaHei, SimHei, Arial, sans-serif"),
                        ticklen=10,
                        tickcolor="rgba(0,0,0,0)",
                        automargin=True,
                    ),
                )

                with st.container(key="shap_plot_inner"):
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={
                            "displaylogo": False,
                            "responsive": True,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                        },
                    )

    st.markdown("---")

    with st.container(key="records_section"):
        st.markdown(
            f"""
            <div style="font-size:24px;font-weight:800;margin:0 0 10px 0;color:#1F2937;">
                {t['records']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        records_df = pd.DataFrame(st.session_state.records)

        if not records_df.empty:
            records_df = canonicalize_record_columns(records_df)
            canonical_order = [*FEATURE_NAMES, RECORD_OUTPUT_KEY]

            for col in canonical_order:
                if col not in records_df.columns:
                    records_df[col] = pd.NA

            _, _, _, display_rename_map = get_record_display_schema(t)
            records_df = records_df[canonical_order].rename(columns=display_rename_map)
            records_df.index = range(1, len(records_df) + 1)
            records_df.index.name = t["sample_index_name"]

        st.dataframe(records_df, use_container_width=True, height=420)

        _, clear_col = st.columns([0.84, 0.16])
        with clear_col:
            if st.button(t["clear_records"], key="clear_records_button", use_container_width=True):
                st.session_state.records = []
                st.rerun()


def render_lca_cost_page(result, t):
    left_col, right_col = st.columns([0.38, 0.62], gap="small")

    with left_col:
        lca_cards_html = (
            f'<div class="lca-result-wrap">'
            f'<div class="lca-result-outer">'
            f'<div class="lca-inner-stack">'

            f'<div class="lca-metric-block">'
            f'<div class="lca-metric-title">{t["net_carbon"]}</div>'
            f'<div class="lca-metric-card">'
            f'<div class="lca-metric-value">{result["Carbon_net"]:.6g}</div>'
            f'<div class="lca-metric-unit">kg CO<sub>2</sub>-eq/t</div>'
            f'</div>'
            f'</div>'

            f'<div class="lca-metric-block">'
            f'<div class="lca-metric-title">{t["total_cost"]}</div>'
            f'<div class="lca-metric-card">'
            f'<div class="lca-metric-value">{result["costs"]["C_total"]:.6g}</div>'
            f'<div class="lca-metric-unit">$/t</div>'
            f'</div>'
            f'</div>'

            f'</div>'
            f'</div>'
            f'</div>'
        )

        st.markdown(lca_cards_html, unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="lca-cost-chart-wrap">', unsafe_allow_html=True)

        st.markdown(
            f"<div class='lca-chart-title'>{t['cost_share']}</div>",
            unsafe_allow_html=True,
        )

        lca_detail_columns = t["lca_detail_columns"]
        cost_items = ["Heat", "Electricity", "Water"]
        cost_df = pd.DataFrame({
            "Cost item key": cost_items,
            "Cost item": [lca_detail_columns[item] for item in cost_items],
            "Cost": [
                result["costs"]["C_heat"],
                result["costs"]["C_elec"],
                result["costs"]["C_water"],
            ],
        })

        cost_color_map = {
            lca_detail_columns[item]: LCA_COLOR_MAP[item]
            for item in cost_items
        }

        fig = px.pie(
            cost_df,
            names="Cost item",
            values="Cost",
            hole=0.55,
            color="Cost item",
            color_discrete_map=cost_color_map,
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            sort=False,
            hovertemplate=(
                f"<b>%{{label}}</b><br>"
                f"{t['cost_value_label']}: %{{value:.4g}}<br>"
                f"{t['cost_share_label']}: %{{percent}}"
                "<extra></extra>"
            ),
        )

        fig.update_layout(
            showlegend=False,
            height=390,
            width=520,
            margin=dict(l=40, r=40, t=10, b=10),

            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(
                    text=f"● {lca_detail_columns['Heat']}",
                    x=0.45,
                    y=0.57,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=18, color=LCA_COLOR_MAP["Heat"]),
                ),
                dict(
                    text=f"● {lca_detail_columns['Electricity']}",
                    x=0.45,
                    y=0.50,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=18, color=LCA_COLOR_MAP["Electricity"]),
                ),
                dict(
                    text=f"● {lca_detail_columns['Water']}",
                    x=0.45,
                    y=0.43,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=18, color=LCA_COLOR_MAP["Water"]),
                ),
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.container(key="lca_impact_section"):
        st.markdown(
            f"""
            <div class="lca-impact-title">
                {t['impact_share']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        lca = result["lca"].copy()

        contribution_source_column = t["contribution_source_label"]
        contribution_value_column = t["impact_y_label"]
        source_key_map = {
            "Heat_%": "Heat",
            "Electricity_%": "Electricity",
            "Water_%": "Water",
            "Fixed_CO2_%": "Fixed_CO2",
        }
        source_color_key_map = {
            "Heat": "Heat",
            "Electricity": "Electricity",
            "Water": "Water",
            "Fixed_CO2": "Fixed CO₂",
        }
        source_order = ["Heat", "Electricity", "Water", "Fixed_CO2"]
        source_display_order = [lca_detail_columns[source] for source in source_order]

        plot_df = lca.melt(
            id_vars=["Impact"],
            value_vars=list(source_key_map.keys()),
            var_name="Contribution source key",
            value_name=contribution_value_column,
        )

        plot_df["Contribution source key"] = plot_df["Contribution source key"].replace(source_key_map)
        plot_df[contribution_source_column] = plot_df["Contribution source key"].map(lca_detail_columns)

        impact_color_map = {
            lca_detail_columns[source]: LCA_COLOR_MAP[source_color_key_map[source]]
            for source in source_order
        }

        fig = px.bar(
            plot_df,
            x="Impact",
            y=contribution_value_column,
            color=contribution_source_column,
            barmode="relative",
            hover_data={contribution_value_column: ":.4f"},
            color_discrete_map=impact_color_map,
            category_orders={
                contribution_source_column: source_display_order
            },
            labels={
                "Impact": t["impact_x_label"],
                contribution_value_column: t["impact_y_label"],
                contribution_source_column: "",
            },
        )

        fig.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=75, b=10),
            xaxis_tickangle=-45,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.08,
                xanchor="left",
                x=0,
                title_text="",
                font=dict(
                    size=15,
                    color="#111827",
                    family="Microsoft YaHei, SimHei, Arial, sans-serif",
                ),
            ),
            xaxis=dict(
                title=f"<b>{t['impact_x_label']}</b>",
                title_font=dict(
                    size=18,
                    color="#111827",
                    family="Microsoft YaHei, SimHei, Arial, sans-serif",
                ),
                tickfont=dict(
                    size=15,
                    color="#111827",
                    family="Microsoft YaHei, SimHei, Arial, sans-serif",
                ),
                automargin=True,
            ),
            yaxis=dict(
                title=f"<b>{t['impact_y_label']}</b>",
                title_font=dict(
                    size=18,
                    color="#111827",
                    family="Microsoft YaHei, SimHei, Arial, sans-serif",
                ),
                tickfont=dict(
                    size=15,
                    color="#111827",
                    family="Microsoft YaHei, SimHei, Arial, sans-serif",
                ),
                automargin=True,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )

        lca_detail = get_lca_detail_display_df(lca, t)

        with st.expander(t["lca_detail_title"]):
            st.dataframe(lca_detail, use_container_width=True, height=360)


def main():
    init_state()
    model = get_model()

    reverse = {v: k for k, v in LANG_LABEL_TO_KEY.items()}

    render_fixed_header()

    selected = st.selectbox(
        "",
        list(LANG_LABEL_TO_KEY.keys()),
        index=list(LANG_LABEL_TO_KEY.keys()).index(reverse[st.session_state.lang]),
        label_visibility="collapsed",
        key="language_selector",
    )
    st.session_state.lang = LANG_LABEL_TO_KEY[selected]

    t = TEXT[st.session_state.lang]

    st.markdown(
        f"<div class='sscsp-page-title'>{t['title']}</div>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([0.27, 0.73], gap="small")

    with left_col:
        with st.container(key="left_input_panel"):
            run_clicked = collect_sidebar_inputs(t)

    with right_col:
        if run_clicked:
            perform_prediction(model, t)

        result = st.session_state.last_result

        tab1, tab2 = st.tabs([t["prediction_page"], t["lca_page"]])
        if result is None:
            with tab1:
                errors = st.session_state.get("errors", {})

                if errors:
                    st.markdown(
                        f"""
                        <div class="empty-result-hint-wrap">
                            <div style="
                                width:min(620px,70%);
                                background:#FFF0F0;
                                color:#D32F2F;
                                border:1px solid #F5C2C7;
                                border-radius:8px;
                                padding:16px 24px;
                                text-align:center;
                                font-size:18px;
                                font-weight:600;
                                line-height:1.6;
                            ">
                                {t["error_title"]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="empty-result-hint-wrap">
                            <div class="empty-result-hint">
                                {t["empty_prediction_hint"]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with tab2:
                st.markdown(
                    f"""
                    <div class="empty-result-hint-wrap">
                        <div class="empty-result-hint">
                            {t["empty_lca_hint"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            return

        with tab1:
            render_prediction_page(result, t)
        with tab2:
            render_lca_cost_page(result, t)


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Dict, List, Tuple, Any
import math
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path


FEATURE_NAMES = [
    "CaO",
    "MgO",
    "SiO2",
    "Fe2O3",
    "MnO",
    "Particle size",
    "Temperature",
    "CO2 partial pressure",
    "CO2 concentration",
    "W/S",
    "Time",
]


INPUT_LIMITS: Dict[str, Tuple[float, float]] = {
    "CaO": (20.91, 65.00),
    "MgO": (1.65, 13.50),
    "SiO2": (4.35, 33.74),
    "Fe2O3": (0.25, 42.98),
    "MnO": (0.00, 6.80),
    "W/S": (1.00, 50.00),
    "Particle size": (5.86, 200.00),
    "Temperature": (20.00, 100.00),
    "Time": (0.08, 240.00),
    "CO2 partial pressure": (0.01, 3.00),
    "CO2 concentration": (9.00, 100.00),
}

EXAMPLE_INPUTS: Dict[str, float] = {
    "CaO": 42.5,
    "MgO": 4.81,
    "SiO2": 31.9,
    "Fe2O3": 0.34,
    "MnO": 0.5,
    "Particle size": 10.0,
    "Temperature": 20.0,
    "CO2 partial pressure": 2.997,
    "CO2 concentration": 99.9,
    "W/S": 1.0,
    "Time": 48.0,
}

PRICE_FACTORS = {
    "electricity_usd_per_kwh": 0.0913,
    "heat_usd_per_mj": 0.0145,
    "water_usd_per_t": 0.707,
}


IMPACT_FACTORS = pd.DataFrame(
    [
        # Impact, unit, electricity/kWh, natural-gas heat/MJ, tap water/kg, fixed CO2
        ("TAP", "kg SO2-Eq",      2.731126e-03, 3.211480e-05, 4.255122e-06,  0.0),
        # Full-precision GWP factors are retained; Table S6 displays rounded values.
        ("GWP", "kg CO2-Eq",      5.777000e-01, 7.0598562e-02, 1.199167e-03, -1.0),
        ("FETP", "kg 1,4-DCB-Eq", 1.195035e-02, 2.725215e-04, 6.957537e-05,  0.0),
        ("METP", "kg 1,4-DCB-Eq", 1.672427e-02, 3.801663e-04, 9.657820e-05,  0.0),
        ("TETP", "kg 1,4-DCB-Eq", 4.635803e-01, 9.362156e-03, 2.382921e-03,  0.0),
        ("FFP", "kg oil-Eq",      1.766903e-01, 2.600145e-02, 3.082891e-04,  0.0),
        ("FEP", "kg P-Eq",        1.298340e-04, 1.325669e-06, 4.545212e-07,  0.0),
        ("MEP", "kg N-Eq",        1.082511e-05, 6.194780e-07, 5.091126e-08,  0.0),
        ("HTPc", "kg 1,4-DCB-Eq", 1.091972e-01, 3.379091e-03, 1.941034e-03,  0.0),
        ("HTPnc", "kg 1,4-DCB-Eq",6.925450e-01, 6.289319e-03, 1.542508e-03,  0.0),
        ("IRP", "kBq Co-60-Eq",   8.444581e-02, 1.986841e-04, 8.706073e-05,  0.0),
        ("LOP", "m2*a crop-Eq",   9.305792e-03, 7.297756e-05, 1.844653e-05,  0.0),
        ("SOP", "kg Cu-Eq",       3.339274e-03, 5.529601e-04, 1.112647e-04,  0.0),
        ("ODP", "kg CFC-11-Eq",   1.941033e-07, 8.517092e-09, 6.451502e-10,  0.0),
        ("PMFP", "kg PM2.5-Eq",   1.220591e-03, 1.401426e-05, 2.554610e-06,  0.0),
        ("HOFP", "kg NOx-Eq",     2.255168e-03, 5.646282e-05, 3.187920e-06,  0.0),
        ("EOFP", "kg NOx-Eq",     2.272443e-03, 6.732381e-05, 3.259350e-06,  0.0),
        ("WCP", "m3",             2.574587e-03, 3.070590e-05, 7.250577e-06,  0.0),
    ],
    columns=["Impact", "Unit", "Electricity", "Heat", "Water", "Fixed_CO2"],
)


def load_model(model_path: str | Path = "final_lgbm_model.joblib"):
    return joblib.load(model_path)


def load_metadata(metadata_path: str | Path = "final_lgbm_metadata.json") -> Dict[str, Any]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def particle_size_to_um(value: float, unit: str) -> float:
    if unit == "μm":
        return value
    if unit == "mm":
        return value * 1000.0
    if unit == "mesh":
        # Given relation: mesh * μm = 15000
        if value <= 0:
            raise ValueError("Mesh value must be greater than 0")
        return 15000.0 / value
    raise ValueError(f"Unsupported particle size unit: {unit}")


def temperature_to_c(value: float, unit: str) -> float:
    if unit == "℃":
        return value
    if unit == "K":
        # Standard Kelvin to Celsius conversion.
        return value - 273.15
    raise ValueError(f"Unsupported temperature unit: {unit}")


def time_to_h(value: float, unit: str) -> float:
    if unit == "h":
        return value
    if unit == "min":
        return value / 60.0
    if unit == "d":
        return value * 24.0
    raise ValueError(f"Unsupported time unit: {unit}")


def pressure_to_mpa(value: float, unit: str) -> float:
    if unit == "MPa":
        return value
    if unit == "KPa":
        return value / 1000.0
    if unit == "Pa":
        return value / 1_000_000.0
    if unit == "bar":
        return value / 10.0
    raise ValueError(f"Unsupported pressure unit: {unit}")


def validate_inputs(
    inputs: Dict[str, float],
    messages: Dict[str, str] | None = None,
    existing_errors: Dict[str, str] | None = None,
) -> Dict[str, str]:
    errors: Dict[str, str] = dict(existing_errors or {})
    message_templates = {
        "missing": "missing",
        "out_of_range": "out_of_range: {lo:g}–{hi:g}",
    }
    if messages:
        message_templates.update(messages)

    for name, (lo, hi) in INPUT_LIMITS.items():
        if name in errors:
            continue

        v = inputs.get(name)
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            errors[name] = message_templates["missing"].format(name=name, lo=lo, hi=hi)
        elif v < lo or v > hi:
            errors[name] = message_templates["out_of_range"].format(name=name, lo=lo, hi=hi)
    return errors


def make_feature_frame(inputs: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([[inputs[k] for k in FEATURE_NAMES]], columns=FEATURE_NAMES)


def predict_co2(model, inputs: Dict[str, float]) -> float:
    X = make_feature_frame(inputs)
    pred = float(model.predict(X)[0])
    return pred


def theoretical_co2(CaO: float, MgO: float) -> float:
    return 10.0 * (0.785 * CaO + 1.091 * MgO)


def calculate_process_inventory(inputs: Dict[str, float], cs_kg_per_t: float) -> Dict[str, float]:
    """
    Calculate the five-stage inventory for the functional unit of 1 t-DSS.

    The equations follow Supplementary Eqs. Se1-Se23:
    - electricity: CO2 supply, slag grinding, and product filtration;
    - heat: slurry sensible heating plus reactor heat loss, and product drying;
    - water: slurry-preparation water determined by W/S.

    ``W_total`` is returned in kg water because the environmental factors are
    expressed per kg water. ``M_w_t`` is also returned for the cost calculation
    and for direct comparison with the supplementary equations.
    """
    W_S = float(inputs["W/S"])
    temperature = float(inputs["Temperature"])
    reaction_time = float(inputs["Time"])

    # Functional unit and fixed process parameters from the supplementary method.
    M_DSS = 1.0  # t dry steel slag
    rho_slag = 3.2  # t/m3
    T_ref = 25.0  # degC
    c_p_slag = 0.8  # MJ/(t*K)
    c_p_water = 4.18  # MJ/(t*K)
    U = 0.56  # W/(m2*K)
    eta_heat = 0.8
    eta_dry = 0.8

    # S1: CO2 capture, purification, compression, and delivery (Se1-Se2).
    M_CO2 = float(cs_kg_per_t)  # kg CO2/t-DSS
    E_comp = 0.275 * M_CO2

    # S2: steel slag grinding (Se3).
    E_grind = 56.25 * M_DSS

    # S3: slurry preparation and heated carbonation (Se4-Se14).
    M_w_t = W_S * M_DSS  # t water/t-DSS
    W_total = M_w_t * 1000.0  # kg water/t-DSS
    V_water = W_S  # m3/t-DSS, assuming 1 t/m3 water density
    V_slag = M_DSS / rho_slag
    V_mix = V_slag + V_water
    V_react = 1.1 * V_mix
    D_react = (4.0 * V_react / math.pi) ** (1.0 / 3.0)
    A_reactor = 1.5 * math.pi * D_react**2

    delta_T = max(temperature - T_ref, 0.0)
    if delta_T == 0.0:
        Q_sensible = 0.0
        Q_loss = 0.0
        Q_useful = 0.0
        Q_heat = 0.0
    else:
        Q_sensible = delta_T * (M_DSS * c_p_slag + M_w_t * c_p_water)
        Q_loss = A_reactor * U * delta_T * reaction_time * 0.0036
        Q_useful = Q_sensible + Q_loss
        Q_heat = Q_useful / eta_heat

    # S4: product filtration (Se15-Se16).
    M_carbon_dry = M_DSS + M_CO2 / 1000.0
    E_filter = 10.0 * M_carbon_dry
    E_total = E_comp + E_grind + E_filter

    # S5: product drying (Se17-Se21).
    moisture_ratio_initial = 0.30 / (1.0 - 0.30)
    moisture_ratio_final = 0.05 / (1.0 - 0.05)
    M_evap = M_carbon_dry * (moisture_ratio_initial - moisture_ratio_final)
    Q_evap = M_evap * (c_p_water * (100.0 - T_ref) + 2257.0)
    Q_dry = Q_evap / eta_dry

    # Section B: aggregated electricity, natural-gas heat, and water demands.
    Q_total = Q_heat + Q_dry

    return {
        "E_comp": E_comp,
        "E_grind": E_grind,
        "E_filter": E_filter,
        "E_total": E_total,
        "M_w_t": M_w_t,
        "W_total": W_total,
        "V_water": V_water,
        "V_slag": V_slag,
        "V_mix": V_mix,
        "V_react": V_react,
        "D_react": D_react,
        "A_reactor": A_reactor,
        "Q_sensible": Q_sensible,
        "Q_loss": Q_loss,
        "Q_useful": Q_useful,
        "Q_heat": Q_heat,
        "M_carbon_dry": M_carbon_dry,
        "M_evap": M_evap,
        "Q_evap": Q_evap,
        "Q_dry": Q_dry,
        "Q_total": Q_total,
        "M_CO2": M_CO2,
    }


def calculate_lca(inv: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for _, r in IMPACT_FACTORS.iterrows():
        electricity = inv["E_total"] * r["Electricity"]
        heat = inv["Q_total"] * r["Heat"]
        water = inv["W_total"] * r["Water"]
        fixed = inv["M_CO2"] * r["Fixed_CO2"]
        positive_total = electricity + heat + water
        total_with_fixed = positive_total + fixed

        def pct(x: float) -> float:
            # All utility contributions and the negative sequestration contribution
            # are referenced to the same positive process-impact denominator.
            return (x / positive_total * 100.0) if positive_total else 0.0

        rows.append({
            "Impact": r["Impact"],
            "Unit": r["Unit"],
            "Electricity": electricity,
            "Heat": heat,
            "Water": water,
            "Fixed_CO2": fixed,
            "Positive_total": positive_total,
            "Total_with_fixed": total_with_fixed,
            "Electricity_%": pct(electricity),
            "Heat_%": pct(heat),
            "Water_%": pct(water),
            "Fixed_CO2_%": pct(fixed),
        })
    return pd.DataFrame(rows)


def calculate_net_carbon(lca_df: pd.DataFrame) -> float:
    # Net carbon is represented by GWP including fixed CO2.
    row = lca_df.loc[lca_df["Impact"] == "GWP"].iloc[0]
    return float(row["Total_with_fixed"])


def calculate_costs(inv: Dict[str, float]) -> Dict[str, float]:
    C_elec = inv["E_total"] * PRICE_FACTORS["electricity_usd_per_kwh"]
    C_heat = inv["Q_total"] * PRICE_FACTORS["heat_usd_per_mj"]
    # Water price is USD/t water, while W_total is kg water.
    C_water = (inv["W_total"] / 1000.0) * PRICE_FACTORS["water_usd_per_t"]
    C_total = C_elec + C_heat + C_water

    def ratio(x: float) -> float:
        return (x / C_total * 100.0) if C_total else 0.0

    return {
        "C_elec": C_elec,
        "C_heat": C_heat,
        "C_water": C_water,
        "C_total": C_total,
        "electricity_cost_ratio": ratio(C_elec),
        "heat_cost_ratio": ratio(C_heat),
        "water_cost_ratio": ratio(C_water),
    }


def calculate_shap_topn(model, inputs: Dict[str, float], n: int = 6) -> pd.DataFrame:
    """
    Returns top-N current-sample contributions.
    Uses SHAP TreeExplainer when installed; falls back to model feature_importances_.
    """
    X = make_feature_frame(inputs)

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)

        if isinstance(values, list):
            values = values[0]

        values = np.asarray(values).reshape(-1)

        df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Contribution": values,
            "AbsContribution": np.abs(values),
        })

        df = df.sort_values("AbsContribution", ascending=False).head(n)

        return df[["Feature", "Contribution", "AbsContribution"]]

    except Exception:
        importances = getattr(model, "feature_importances_", np.ones(len(FEATURE_NAMES)))

        df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Contribution": importances,
            "AbsContribution": np.abs(importances),
        })

        df = df.sort_values("AbsContribution", ascending=False).head(n)

        return df[["Feature", "Contribution", "AbsContribution"]]


def run_all(model, inputs: Dict[str, float]) -> Dict[str, Any]:
    errors = validate_inputs(inputs)
    if errors:
        raise ValueError(f"Invalid inputs: {errors}")

    cs = predict_co2(model, inputs)
    # g/kg is numerically equivalent to kg/t
    cs_kg_per_t = cs
    theory = theoretical_co2(inputs["CaO"], inputs["MgO"])
    inv = calculate_process_inventory(inputs, cs_kg_per_t)
    lca = calculate_lca(inv)
    carbon_net = calculate_net_carbon(lca)
    costs = calculate_costs(inv)
    shap_top6 = calculate_shap_topn(model, inputs, n=6)

    return {
        "CO2_sequestration": cs,
        "CO2_sequestration_kg_per_t": cs_kg_per_t,
        "CO2_theoretical": theory,
        "inventory": inv,
        "lca": lca,
        "Carbon_net": carbon_net,
        "costs": costs,
        "shap_top6": shap_top6,
        "shap_top5": shap_top6.head(5).copy(),
    }


from __future__ import annotations

from dataclasses import dataclass
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

# Input limits in the canonical model units.
# Particle size: μm; Temperature: ℃; Pressure: MPa; Time: h
INPUT_LIMITS: Dict[str, Tuple[float, float]] = {
    "CaO": (20.91, 65.00),
    "MgO": (1.65, 13.50),
    "SiO2": (4.35, 33.74),
    "Fe2O3": (0.25, 42.98),
    "MnO": (0.00, 6.80),
    "W/S": (1.00, 50.00),
    "Particle size": (5.86, 200.00),
    "Temperature": (20.00, 100.00),
    "Time": (5.86, 200.00),
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
    "Temperature": 25.0,
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

# ReCiPe-style 18-category impact factors per unit consumption.
# Electricity: per kWh; Heat: per MJ; Water: per kg water; Fixed_CO2: per kg CO2 stored
IMPACT_FACTORS = pd.DataFrame(
    [
        ("TAP", "kg SO2-Eq",      0.002731126, 3.21e-05,    4.26e-06,   0.0),
        ("GWP", "kg CO2-Eq",      0.5777,      0.070598562, 0.001199167, -1.0),
        ("FETP", "kg 1,4-DCB-Eq", 0.011950347, 2.73e-04,   6.96e-05,   0.0),
        ("METP", "kg 1,4-DCB-Eq", 0.016724271, 3.80e-04,   9.66e-05,   0.0),
        ("TETP", "kg 1,4-DCB-Eq", 0.463580254, 0.009362156,0.002382921,0.0),
        ("FFP", "kg oil-Eq",      0.176690322, 0.026001448,3.08e-04,   0.0),
        ("FEP", "kg P-Eq",        1.30e-04,    1.33e-06,   4.55e-07,   0.0),
        ("MEP", "kg N-Eq",        1.08e-05,    6.19e-07,   5.09e-08,   0.0),
        ("HTPc", "kg 1,4-DCB-Eq", 0.109197232, 0.003379091,0.001941034,0.0),
        ("HTPnc", "kg 1,4-DCB-Eq",0.692545038, 0.006289319,0.001542508,0.0),
        ("IRP", "kBq Co-60-Eq",   0.084445805, 1.99e-04,   8.71e-05,   0.0),
        ("LOP", "m2*a crop-Eq",   0.009305792, 7.30e-05,   1.84e-05,   0.0),
        ("SOP", "kg Cu-Eq",       0.003339274, 5.53e-04,   1.11e-04,   0.0),
        ("ODP", "kg CFC-11-Eq",   1.94e-07,    8.52e-09,   6.45e-10,   0.0),
        ("PMFP", "kg PM2.5-Eq",   0.001220591, 1.40e-05,   2.55e-06,   0.0),
        ("HOFP", "kg NOx-Eq",     0.002255168, 5.65e-05,   3.19e-06,   0.0),
        ("EOFP", "kg NOx-Eq",     0.002272443, 6.73e-05,   3.26e-06,   0.0),
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
        return 15000.0 / value
    raise ValueError(f"Unsupported particle size unit: {unit}")


def temperature_to_c(value: float, unit: str) -> float:
    if unit == "℃":
        return value
    if unit == "K":
        # Standard conversion. If your original document intended 1℃ = 274.15K literally,
        # change this line to value - 274.15.
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


def validate_inputs(inputs: Dict[str, float]) -> Dict[str, str]:
    errors: Dict[str, str] = {}
    for name, (lo, hi) in INPUT_LIMITS.items():
        v = inputs.get(name)
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            errors[name] = "Missing"
        elif v < lo or v > hi:
            errors[name] = f"Out of range: {lo:g}–{hi:g}"
    return errors


def make_feature_frame(inputs: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([[inputs[k] for k in FEATURE_NAMES]], columns=FEATURE_NAMES)


def predict_co2(model, inputs: Dict[str, float]) -> float:
    X = make_feature_frame(inputs)
    pred = float(model.predict(X)[0])
    return pred


def theoretical_co2(CaO: float, MgO: float) -> float:
    # g CO2/kg DSS is numerically equivalent to kg CO2/t DSS
    return 10.0 * (0.785 * CaO + 1.091 * MgO)


def calculate_process_inventory(inputs: Dict[str, float], cs_kg_per_t: float) -> Dict[str, float]:
    """
    Inventory for 1 t dry steel slag.

    Important: the original Word file leaves Q_heat and Q_dry formulas blank.
    Implemented editable assumptions:
    - Q_heat heats water from 25℃ to reaction temperature: water_kg * 4.18 kJ/(kg·℃) * ΔT / 1000.
    - Q_dry defaults to 0 MJ/t until a drying heat formula is supplied.
    """
    W_S = inputs["W/S"]
    temperature = inputs["Temperature"]

    E_comp = 0.275 * cs_kg_per_t
    E_grind = 56.25
    E_filter = 10.0 * (1000.0 + cs_kg_per_t) / 1000.0
    E_total = E_comp + E_grind + E_filter

    M_w = W_S * 1000.0  # kg water/t dry slag
    M_CO2 = cs_kg_per_t

    if temperature <= 25:
        Q_heat = 0.0
    else:
        Q_heat = M_w * 4.18 * (temperature - 25.0) / 1000.0

    moisture_ratio_initial = 0.3 / 0.7
    moisture_ratio_final = 0.05 / 0.95
    product_mass_factor = (1000.0 + cs_kg_per_t) / 1000.0
    latent_heat_term = 4.18 * 75.0 + 2257.0
    dryer_efficiency = 0.8 * 0.95

    Q_dry = (
            moisture_ratio_initial
            / moisture_ratio_final
            * product_mass_factor
            * latent_heat_term
            / dryer_efficiency
    )

    Q_total = Q_heat + Q_dry

    return {
        "E_comp": E_comp,
        "E_grind": E_grind,
        "E_filter": E_filter,
        "E_total": E_total,
        "Q_heat": Q_heat,
        "Q_dry": Q_dry,
        "Q_total": Q_total,
        "W_total": M_w,
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


def calculate_shap_top5(model, inputs: Dict[str, float]) -> pd.DataFrame:
    """
    Returns top-5 current-sample contributions.
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
        }).sort_values("AbsContribution", ascending=False).head(5)
        return df[["Feature", "Contribution", "AbsContribution"]]
    except Exception:
        importances = getattr(model, "feature_importances_", np.ones(len(FEATURE_NAMES)))
        df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Contribution": importances,
            "AbsContribution": np.abs(importances),
        }).sort_values("AbsContribution", ascending=False).head(5)
        return df[["Feature", "Contribution", "AbsContribution"]]


def run_all(model, inputs: Dict[str, float]) -> Dict[str, Any]:
    cs = predict_co2(model, inputs)
    # g/kg is numerically equivalent to kg/t
    cs_kg_per_t = cs
    theory = theoretical_co2(inputs["CaO"], inputs["MgO"])
    inv = calculate_process_inventory(inputs, cs_kg_per_t)
    lca = calculate_lca(inv)
    carbon_net = calculate_net_carbon(lca)
    costs = calculate_costs(inv)
    shap_top5 = calculate_shap_top5(model, inputs)
    return {
        "CO2_sequestration": cs,
        "CO2_sequestration_kg_per_t": cs_kg_per_t,
        "CO2_theoretical": theory,
        "inventory": inv,
        "lca": lca,
        "Carbon_net": carbon_net,
        "costs": costs,
        "shap_top5": shap_top5,
    }

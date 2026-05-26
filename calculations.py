from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd


FEATURE_ORDER = [
    "CaO",
    "MgO",
    "SiO2",
    "Fe2O3",
    "MnO",
    "liquid_solid_ratio",
    "particle_size",
    "temperature",
    "reaction_time",
    "co2_pressure",
    "co2_concentration",
]


DEFAULT_INPUTS: Dict[str, float] = {
    "CaO": 1.0,
    "MgO": 1.0,
    "SiO2": 1.0,
    "Fe2O3": 1.0,
    "MnO": 1.0,
    "liquid_solid_ratio": 1.0,
    "particle_size": 1.0,
    "temperature": 1.0,
    "reaction_time": 1.0,
    "co2_pressure": 1.0,
    "co2_concentration": 1.0,
}


INPUT_LIMITS: Dict[str, Tuple[float, float]] = {
    "CaO": (20.91, 65.0),
    "MgO": (1.65, 13.5),
    "SiO2": (4.35, 33.74),
    "Fe2O3": (0.25, 42.98),
    "MnO": (0.0, 6.8),
    "liquid_solid_ratio": (1.0, 50.0),
    "particle_size": (5.86, 200.0),
    "temperature": (20.0, 100.0),
    "reaction_time": (5.86, 200.0),
    "co2_pressure": (0.01, 3.0),
    "co2_concentration": (9.0, 100.0),
}


def load_model_and_metadata(model_path: Path, metadata_path: Path):
    model = joblib.load(model_path)

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return model, metadata


def get_feature_order(metadata: Dict[str, Any]) -> list[str]:
    possible_keys = [
        "feature_names",
        "features",
        "feature_order",
        "input_features",
        "columns",
    ]

    for key in possible_keys:
        if key in metadata and isinstance(metadata[key], list):
            return metadata[key]

    return FEATURE_ORDER


def build_input_dataframe(metadata: Dict[str, Any], inputs: Dict[str, float]) -> pd.DataFrame:
    feature_order = get_feature_order(metadata)

    row = {}
    for feature in feature_order:
        if feature in inputs:
            row[feature] = inputs[feature]
        else:
            normalized = feature.strip()
            row[feature] = inputs.get(normalized, 0.0)

    return pd.DataFrame([row], columns=feature_order)


def validate_inputs(inputs: Dict[str, float]) -> list[str]:
    errors = []

    for key, (min_v, max_v) in INPUT_LIMITS.items():
        value = inputs.get(key)

        if value is None:
            errors.append(f"{key} 不能为空")
            continue

        try:
            value = float(value)
        except Exception:
            errors.append(f"{key} 必须是数字")
            continue

        if value < min_v or value > max_v:
            errors.append(f"{key} 超出范围 [{min_v}, {max_v}]")

    return errors


def predict_co2_uptake(model, metadata: Dict[str, Any], inputs: Dict[str, float]) -> float:
    if model is None:
        return 0.0

    X = build_input_dataframe(metadata, inputs)

    prediction = model.predict(X)

    if isinstance(prediction, (list, tuple, np.ndarray)):
        prediction = float(np.asarray(prediction).reshape(-1)[0])
    else:
        prediction = float(prediction)

    return prediction


def calculate_theoretical_co2(inputs: Dict[str, float]) -> float:
    """
    理论 CO₂ 固存量，单位：g-CO₂/kg-DSS。

    按常用氧化物理论固碳量计算：
    CaO + CO₂ -> CaCO₃
    MgO + CO₂ -> MgCO₃

    若你的 Word 里理论值还有 FeO、MnO 或其他修正项，可在这里继续补。
    """
    cao = float(inputs["CaO"])
    mgo = float(inputs["MgO"])

    co2_from_cao = cao / 100.0 * 1000.0 * 44.01 / 56.08
    co2_from_mgo = mgo / 100.0 * 1000.0 * 44.01 / 40.30

    return co2_from_cao + co2_from_mgo


def calculate_process_inventory(inputs: Dict[str, float], cs_kg_per_t: float) -> Dict[str, float]:
    """
    过程清单计算。

    cs_kg_per_t:
    模型输出是 g-CO₂/kg-DSS。
    换算到 kg-CO₂/t-DSS 后数值相同：
    1 g/kg = 1 kg/t。
    """
    liquid_solid_ratio = float(inputs["liquid_solid_ratio"])
    temperature = float(inputs["temperature"])

    # 处理 1 t 干钢渣时，水质量 kg/t-DSS
    M_w = liquid_solid_ratio * 1000.0

    # Q_heat：将水从 25℃ 加热到目标温度。
    # 温度 ≤ 25℃ 时默认为 0。
    if temperature <= 25.0:
        Q_heat = 0.0
    else:
        Q_heat = M_w * 4.18 * (temperature - 25.0) / 1000.0

    # Q_dry：按用户公式图。
    # (((0.3/0.7)-(0.05/0.95)) × ((1000 + 预测值)/1000) × (4.18×75 + 2257)) / (0.8×0.95)
    Q_dry = (
        ((0.3 / 0.7) - (0.05 / 0.95))
        * ((1000.0 + cs_kg_per_t) / 1000.0)
        * (4.18 * 75.0 + 2257.0)
    ) / (0.8 * 0.95)

    Q_total = Q_heat + Q_dry

    return {
        "M_w": M_w,
        "Q_heat": Q_heat,
        "Q_dry": Q_dry,
        "Q_total": Q_total,
    }


def calculate_lca_and_cost(
    inputs: Dict[str, float],
    cs_kg_per_t: float,
    inventory: Dict[str, float],
) -> Dict[str, Any]:
    """
    LCA 与成本计算。

    这里的数值用于网页内置计算。
    若 Word 表格里的排放因子/成本因子有更新，优先修改这里。
    """

    Q_total = inventory["Q_total"]
    M_w = inventory["M_w"]

    # 成本参数
    electricity_consumption = 24.0
    electricity_price = 0.0935

    heat_price = 0.00015
    water_price = 0.707

    electricity_cost = electricity_consumption * electricity_price
    heat_cost = Q_total * heat_price
    water_cost = (M_w / 1000.0) * water_price
    total_cost = electricity_cost + heat_cost + water_cost

    # 排放参数
    electricity_emission_factor = 0.581
    heat_emission_factor = 0.07
    water_emission_factor = 0.344

    electricity_emission = electricity_consumption * electricity_emission_factor
    heat_emission = Q_total * heat_emission_factor
    water_emission = (M_w / 1000.0) * water_emission_factor

    process_emission = electricity_emission + heat_emission + water_emission
    fixed_co2 = cs_kg_per_t
    net_carbon_reduction = fixed_co2 - process_emission

    impact_factors = get_default_impact_factors()

    records = []
    for impact_name, factors in impact_factors.items():
        elec = electricity_consumption * factors["Electricity"]
        heat = Q_total * factors["Heat"]
        water = (M_w / 1000.0) * factors["Water"]
        fixed = -fixed_co2 * factors["Fixed CO₂"]

        positive_total = elec + heat + water

        if abs(positive_total) < 1e-12:
            elec_pct = heat_pct = water_pct = fixed_pct = 0.0
        else:
            elec_pct = elec / positive_total * 100.0
            heat_pct = heat / positive_total * 100.0
            water_pct = water / positive_total * 100.0
            fixed_pct = fixed / positive_total * 100.0

        records.append(
            {
                "impact": impact_name,
                "Electricity": elec_pct,
                "Heat": heat_pct,
                "Water": water_pct,
                "Fixed CO₂": fixed_pct,
            }
        )

    impact_contribution_df = pd.DataFrame(records)

    return {
        "electricity_cost": electricity_cost,
        "heat_cost": heat_cost,
        "water_cost": water_cost,
        "total_cost": total_cost,
        "electricity_emission": electricity_emission,
        "heat_emission": heat_emission,
        "water_emission": water_emission,
        "process_emission": process_emission,
        "fixed_co2": fixed_co2,
        "net_carbon_reduction": net_carbon_reduction,
        "impact_contribution_df": impact_contribution_df,
    }


def get_default_impact_factors() -> Dict[str, Dict[str, float]]:
    """
    18项环境影响因子。

    这里先给出可运行版本。
    如果你的 Word 表格里有精确因子，请直接替换这里对应数值。
    Fixed CO₂ 是固存 CO₂ 的折减因子。
    """
    return {
        "TAP": {"Electricity": 0.00210, "Heat": 0.000018, "Water": 0.000010, "Fixed CO₂": 0.000000},
        "GWP": {"Electricity": 0.58100, "Heat": 0.070000, "Water": 0.344000, "Fixed CO₂": 1.000000},
        "FETP": {"Electricity": 0.01800, "Heat": 0.000160, "Water": 0.000050, "Fixed CO₂": 0.000000},
        "METP": {"Electricity": 0.02000, "Heat": 0.000170, "Water": 0.000052, "Fixed CO₂": 0.000000},
        "TETP": {"Electricity": 0.01900, "Heat": 0.000150, "Water": 0.000046, "Fixed CO₂": 0.000000},
        "FFP": {"Electricity": 0.16000, "Heat": 0.025000, "Water": 0.004000, "Fixed CO₂": 0.000000},
        "FEP": {"Electricity": 0.00035, "Heat": 0.000003, "Water": 0.000001, "Fixed CO₂": 0.000000},
        "MEP": {"Electricity": 0.00028, "Heat": 0.000005, "Water": 0.000002, "Fixed CO₂": 0.000000},
        "HTPc": {"Electricity": 0.03000, "Heat": 0.000400, "Water": 0.000220, "Fixed CO₂": 0.000000},
        "HTPnc": {"Electricity": 0.02500, "Heat": 0.000180, "Water": 0.000010, "Fixed CO₂": 0.000000},
        "IRP": {"Electricity": 0.00180, "Heat": 0.000006, "Water": 0.000001, "Fixed CO₂": 0.000000},
        "LOP": {"Electricity": 0.00400, "Heat": 0.000020, "Water": 0.000004, "Fixed CO₂": 0.000000},
        "SOP": {"Electricity": 0.00600, "Heat": 0.000160, "Water": 0.000060, "Fixed CO₂": 0.000000},
        "ODP": {"Electricity": 0.00001, "Heat": 0.0000001, "Water": 0.00000002, "Fixed CO₂": 0.000000},
        "PMFP": {"Electricity": 0.00150, "Heat": 0.000020, "Water": 0.000004, "Fixed CO₂": 0.000000},
        "HOFP": {"Electricity": 0.00220, "Heat": 0.000025, "Water": 0.000006, "Fixed CO₂": 0.000000},
        "EOFP": {"Electricity": 0.00190, "Heat": 0.000022, "Water": 0.000006, "Fixed CO₂": 0.000000},
        "WCP": {"Electricity": 0.10000, "Heat": 0.000400, "Water": 0.020000, "Fixed CO₂": 0.000000},
    }


def calculate_shap_contrib(model, metadata: Dict[str, Any], inputs: Dict[str, float]) -> pd.DataFrame:
    """
    当前样本贡献分析。

    优先使用 LightGBM 原生 pred_contrib=True。
    如果模型不支持，则返回空 DataFrame。
    """
    if model is None:
        return pd.DataFrame(columns=["feature", "value", "abs_value"])

    X = build_input_dataframe(metadata, inputs)
    feature_order = list(X.columns)

    try:
        contrib = model.predict(X, pred_contrib=True)
        contrib = np.asarray(contrib)

        if contrib.ndim == 2:
            contrib = contrib[0]

        # LightGBM 最后一列通常是 bias，不作为特征贡献展示。
        if len(contrib) == len(feature_order) + 1:
            contrib = contrib[:-1]

        df = pd.DataFrame(
            {
                "feature": feature_order[: len(contrib)],
                "value": contrib,
            }
        )
        df["abs_value"] = df["value"].abs()
        df = df.sort_values("abs_value", ascending=False).head(5)

        name_map = {
            "CaO": "CaO",
            "MgO": "MgO",
            "SiO2": "SiO2",
            "Fe2O3": "Fe2O3",
            "MnO": "MnO",
            "liquid_solid_ratio": "液固比",
            "particle_size": "粒径",
            "temperature": "温度",
            "reaction_time": "时间",
            "co2_pressure": "CO2分压",
            "co2_concentration": "CO2浓度",
        }
        df["feature"] = df["feature"].map(lambda x: name_map.get(x, x))

        return df

    except Exception:
        return pd.DataFrame(columns=["feature", "value", "abs_value"])

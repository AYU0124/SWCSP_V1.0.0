from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
ASSETS_DIR = ROOT / "assets"

LANG_LABEL_TO_KEY = {
    "简体中文": "zh_CN",
    "繁體中文": "zh_TW",
    "English": "en",
}

LCA_COLOR_MAP = {
    "Electricity": "#7EC4F4",
    "Heat": "#006CC8",
    "Water": "#F2A3A3",
    "Fixed CO₂": "#FF2B2B",
}

TEXT = {
    "zh_CN": {
        "title": "工业固废矿化固碳量预测与环境经济效益评估软件",
        "language": "简体中文",
        "input_params": "输入参数",
        "composition": "组分特征",
        "process": "工艺参数",
        "prediction_page": "预测结果",
        "lca_page": "环境经济效益评估",
        "execute": "执行预测",
        "reset": "重置输入",
        "example": "导入示例",
        "co2_result": "CO₂吸收量",
        "theoretical": "理论值",
        "current_shap": "当前样本贡献分析（前6）",
        "records": "预测记录表",
        "copy_hint": "可在表格中选择并复制；也可下载 CSV。",
        "net_carbon": "净碳排",
        "total_cost": "工艺成本",
        "cost_share": "成本贡献分析",
        "cost_item_label": "成本项",
        "cost_value_label": "成本",
        "cost_share_label": "占比",
        "contribution_source_label": "贡献来源",
        "impact_share": "18项环境影响贡献分析",
        "error_title": "请检查红色输入框：有参数缺失或超出范围。",
        "empty_prediction_hint": "请在左侧输入参数，点击执行预测查看结果。",
        "co2_card_title": "CO<sub>2</sub>吸收量",
        "predicted_value": "预测值",
        "theoretical_value": "理论值",
        "theory_tooltip": "理论值依据 Steinour 方程计算",
        "shap_contribution": "SHAP贡献",
        "download_csv": "导出CSV",
        "clear_records": "清空记录",
        "empty_lca_hint": "请先执行预测，再查看环境经济效益评估结果。",
        "ws_label": "液固比",
        "particle_label": "粒径",
        "temperature_label": "温度",
        "time_label": "反应时间",
        "pressure_label": "CO₂分压",
        "concentration_label": "CO₂浓度",
        "unit_label": "单位",
        "run_label": "执行预测",
        "reset_label": "重置输入",
        "example_label": "导入示例",
        "sample_index_name": "样本",
        "no_shap": "暂无 SHAP 贡献度数据。",
        "mesh_error": "目数必须大于 0",
        "missing_error": "参数缺失",
        "range_error": "超出范围：{lo:g}–{hi:g}",
        "co2_negative_warning": "预测值小于 0，结果可能不具有物理意义。",
        "co2_above_theory_warning": "预测值超过理论 CO₂ 固存量，请谨慎解释。",
        "record_columns": {
            "CaO": "CaO (%)",
            "MgO": "MgO (%)",
            "SiO2": "SiO₂ (%)",
            "Fe2O3": "Fe₂O₃ (%)",
            "MnO": "MnO (%)",
            "Particle size": "粒径 (μm)",
            "Temperature": "温度 (℃)",
            "CO2 partial pressure": "CO₂分压 (MPa)",
            "CO2 concentration": "CO₂浓度 (%)",
            "W/S": "液固比",
            "Time": "反应时间 (h)",
        },
        "output_column": "CO₂吸收量 (g-CO₂/kg-DSS)",
        "impact_x_label": "环境影响",
        "impact_y_label": "贡献率 (%)",
        "feature_label_map": {
            "Time": "反应时间",
            "Particle size": "粒径",
            "Temperature": "温度",
            "W/S": "液固比",
            "CO2 partial pressure": "CO₂分压",
            "CO2 concentration": "CO₂浓度",
            "CaO": "CaO",
            "MgO": "MgO",
            "SiO2": "SiO₂",
            "Fe2O3": "Fe₂O₃",
            "MnO": "MnO",
        },
        "lca_detail_title": "LCA明细",
        "lca_detail_index": "序号",
        "lca_detail_columns": {
            "Impact": "环境影响",
            "Impact_full_name": "环境影响全称",
            "Unit": "单位",
            "Electricity": "电力",
            "Heat": "热量",
            "Water": "用水",
            "Fixed_CO2": "固定 CO₂",
            "Positive_total": "正向合计",
            "Total_with_fixed": "含固定 CO₂合计",
            "Electricity_%": "电力 %",
            "Heat_%": "热量 %",
            "Water_%": "用水 %",
            "Fixed_CO2_%": "固定 CO₂ %",
        },
        "lca_impact_full_names": {
            "GWP": "全球变暖潜势",
            "FEP": "淡水富营养化潜势",
            "MEP": "海洋富营养化潜势",
            "TAP": "陆地酸化潜势",
            "FETP": "淡水生态毒性潜势",
            "METP": "海洋生态毒性潜势",
            "TETP": "陆地生态毒性潜势",
            "HTPc": "人类毒性潜势（致癌）",
            "HTPnc": "人类毒性潜势（非致癌）",
            "IRP": "电离辐射潜势",
            "LOP": "土地占用潜势",
            "SOP": "矿物资源稀缺潜势",
            "ODP": "臭氧层消耗潜势",
            "PMFP": "细颗粒物形成潜势",
            "HOFP": "人类健康臭氧形成潜势",
            "EOFP": "生态系统臭氧形成潜势",
            "WCP": "水消耗潜势",
            "FFP": "化石资源稀缺潜势",
        },
    },
    "zh_TW": {
        "title": "工業固廢礦化固碳量預測與環境經濟效益評估軟體",
        "language": "繁體中文",
        "input_params": "輸入參數",
        "composition": "組分特徵",
        "process": "工藝參數",
        "prediction_page": "預測結果",
        "lca_page": "環境經濟效益評估",
        "execute": "執行預測",
        "reset": "重置輸入",
        "example": "匯入範例",
        "co2_result": "CO₂吸收量",
        "theoretical": "理論值",
        "current_shap": "目前樣本貢獻分析（前6）",
        "records": "預測紀錄表",
        "copy_hint": "可在表格中選擇並複製；也可下載 CSV。",
        "net_carbon": "淨碳排",
        "total_cost": "工藝成本",
        "cost_share": "成本貢獻分析",
        "cost_item_label": "成本項",
        "cost_value_label": "成本",
        "cost_share_label": "佔比",
        "contribution_source_label": "貢獻來源",
        "impact_share": "18項環境影響貢獻分析",
        "error_title": "請檢查紅色輸入框：有參數缺失或超出範圍。",
        "empty_prediction_hint": "請在左側輸入參數，點擊執行預測查看結果。",
        "co2_card_title": "CO<sub>2</sub>吸收量",
        "predicted_value": "預測值",
        "theoretical_value": "理論值",
        "theory_tooltip": "理論值依據 Steinour 方程計算",
        "shap_contribution": "SHAP貢獻",
        "download_csv": "匯出CSV",
        "clear_records": "清空紀錄",
        "empty_lca_hint": "請先執行預測，再查看環境經濟效益評估結果。",
        "ws_label": "液固比",
        "particle_label": "粒徑",
        "temperature_label": "溫度",
        "time_label": "反應時間",
        "pressure_label": "CO₂分壓",
        "concentration_label": "CO₂濃度",
        "unit_label": "單位",
        "run_label": "執行預測",
        "reset_label": "重置輸入",
        "example_label": "匯入範例",
        "sample_index_name": "樣本",
        "no_shap": "暫無 SHAP 貢獻度數據。",
        "mesh_error": "目數必須大於 0",
        "missing_error": "參數缺失",
        "range_error": "超出範圍：{lo:g}–{hi:g}",
        "co2_negative_warning": "預測值小於 0，結果可能不具有物理意義。",
        "co2_above_theory_warning": "預測值超過理論 CO₂ 固存量，請謹慎解釋。",
        "record_columns": {
            "CaO": "CaO (%)",
            "MgO": "MgO (%)",
            "SiO2": "SiO₂ (%)",
            "Fe2O3": "Fe₂O₃ (%)",
            "MnO": "MnO (%)",
            "Particle size": "粒徑 (μm)",
            "Temperature": "溫度 (℃)",
            "CO2 partial pressure": "CO₂分壓 (MPa)",
            "CO2 concentration": "CO₂濃度 (%)",
            "W/S": "液固比",
            "Time": "反應時間 (h)",
        },
        "output_column": "CO₂吸收量 (g-CO₂/kg-DSS)",
        "impact_x_label": "環境影響",
        "impact_y_label": "貢獻率 (%)",
        "feature_label_map": {
            "Time": "反應時間",
            "Particle size": "粒徑",
            "Temperature": "溫度",
            "W/S": "液固比",
            "CO2 partial pressure": "CO₂分壓",
            "CO2 concentration": "CO₂濃度",
            "CaO": "CaO",
            "MgO": "MgO",
            "SiO2": "SiO₂",
            "Fe2O3": "Fe₂O₃",
            "MnO": "MnO",
        },
        "lca_detail_title": "LCA明細",
        "lca_detail_index": "序號",
        "lca_detail_columns": {
            "Impact": "環境影響",
            "Impact_full_name": "環境影響全稱",
            "Unit": "單位",
            "Electricity": "電力",
            "Heat": "熱量",
            "Water": "用水",
            "Fixed_CO2": "固定 CO₂",
            "Positive_total": "正向合計",
            "Total_with_fixed": "含固定 CO₂合計",
            "Electricity_%": "電力 %",
            "Heat_%": "熱量 %",
            "Water_%": "用水 %",
            "Fixed_CO2_%": "固定 CO₂ %",
        },
        "lca_impact_full_names": {
            "GWP": "全球暖化潛勢",
            "FEP": "淡水優養化潛勢",
            "MEP": "海洋優養化潛勢",
            "TAP": "陸地酸化潛勢",
            "FETP": "淡水生態毒性潛勢",
            "METP": "海洋生態毒性潛勢",
            "TETP": "陸地生態毒性潛勢",
            "HTPc": "人類毒性潛勢（致癌）",
            "HTPnc": "人類毒性潛勢（非致癌）",
            "IRP": "游離輻射潛勢",
            "LOP": "土地占用潛勢",
            "SOP": "礦物資源稀缺潛勢",
            "ODP": "臭氧層消耗潛勢",
            "PMFP": "細懸浮微粒形成潛勢",
            "HOFP": "人類健康臭氧形成潛勢",
            "EOFP": "生態系統臭氧形成潛勢",
            "WCP": "水消耗潛勢",
            "FFP": "化石資源稀缺潛勢",
        },
    },
    "en": {
        "title": "Industrial Solid Waste Mineralization CO₂ Sequestration Prediction and Environmental and Economic Benefit Assessment Software",
        "language": "English",
        "input_params": "Input Parameters",
        "composition": "Composition Features",
        "process": "Process Parameters",
        "prediction_page": "Prediction results",
        "lca_page": "Environmental–economic benefit assessment",
        "execute": "Run prediction",
        "reset": "Reset inputs",
        "example": "Load example",
        "co2_result": "CO₂ uptake",
        "theoretical": "Theoretical value",
        "current_shap": "Current sample contribution analysis (Top 6)",
        "records": "Prediction records",
        "copy_hint": "Select and copy cells in the table, or download CSV.",
        "net_carbon": "Net carbon emission",
        "total_cost": "Process cost",
        "cost_share": "Cost contribution analysis",
        "cost_item_label": "Cost item",
        "cost_value_label": "Cost",
        "cost_share_label": "Share",
        "contribution_source_label": "Contribution source",
        "impact_share": "18 impact category contribution analysis",
        "error_title": "Please check red input boxes: missing or out-of-range values.",
        "empty_prediction_hint": "Enter parameters on the left and run prediction to view results.",
        "co2_card_title": "CO<sub>2</sub> sequestration",
        "predicted_value": "Predicted value",
        "theoretical_value": "Theoretical value",
        "theory_tooltip": "The theoretical value is calculated using the Steinour equation.",
        "shap_contribution": "SHAP contribution",
        "download_csv": "Export CSV",
        "clear_records": "Clear records",
        "empty_lca_hint": "Please run a prediction first, then view the environmental–economic benefit assessment results.",
        "ws_label": "W/S",
        "particle_label": "Particle size",
        "temperature_label": "Temperature",
        "time_label": "Time",
        "pressure_label": "CO₂ partial pressure",
        "concentration_label": "CO₂ concentration",
        "unit_label": "Unit",
        "run_label": "Run prediction",
        "reset_label": "Reset inputs",
        "example_label": "Load example",
        "sample_index_name": "Sample",
        "no_shap": "No SHAP contribution data available.",
        "mesh_error": "Mesh value must be greater than 0",
        "missing_error": "Missing",
        "range_error": "Out of range: {lo:g}–{hi:g}",
        "co2_negative_warning": "The predicted value is below 0 and may not be physically meaningful.",
        "co2_above_theory_warning": "The predicted value exceeds the theoretical CO₂ sequestration capacity. Please interpret this result with caution.",
        "record_columns": {
            "CaO": "CaO (%)",
            "MgO": "MgO (%)",
            "SiO2": "SiO₂ (%)",
            "Fe2O3": "Fe₂O₃ (%)",
            "MnO": "MnO (%)",
            "Particle size": "Particle size (μm)",
            "Temperature": "Temperature (℃)",
            "CO2 partial pressure": "CO₂ partial pressure (MPa)",
            "CO2 concentration": "CO₂ concentration (%)",
            "W/S": "W/S",
            "Time": "Time (h)",
        },
        "output_column": "CO₂ sequestration (g-CO₂/kg-DSS)",
        "impact_x_label": "Impact",
        "impact_y_label": "Contribution (%)",
        "feature_label_map": {
            "Time": "Time",
            "Particle size": "Particle size",
            "Temperature": "Temperature",
            "W/S": "W/S",
            "CO2 partial pressure": "CO₂ partial pressure",
            "CO2 concentration": "CO₂ concentration",
            "CaO": "CaO",
            "MgO": "MgO",
            "SiO2": "SiO₂",
            "Fe2O3": "Fe₂O₃",
            "MnO": "MnO",
        },
        "lca_detail_title": "LCA details",
        "lca_detail_index": "No.",
        "lca_detail_columns": {
            "Impact": "Impact",
            "Impact_full_name": "Impact full name",
            "Unit": "Unit",
            "Electricity": "Electricity",
            "Heat": "Heat",
            "Water": "Water",
            "Fixed_CO2": "Fixed CO₂",
            "Positive_total": "Positive total",
            "Total_with_fixed": "Total with fixed CO₂",
            "Electricity_%": "Electricity %",
            "Heat_%": "Heat %",
            "Water_%": "Water %",
            "Fixed_CO2_%": "Fixed CO₂ %",
        },
        "lca_impact_full_names": {
            "GWP": "Global warming potential",
            "FEP": "Freshwater eutrophication potential",
            "MEP": "Marine eutrophication potential",
            "TAP": "Terrestrial acidification potential",
            "FETP": "Freshwater ecotoxicity potential",
            "METP": "Marine ecotoxicity potential",
            "TETP": "Terrestrial ecotoxicity potential",
            "HTPc": "Human toxicity potential, carcinogenic",
            "HTPnc": "Human toxicity potential, non-carcinogenic",
            "IRP": "Ionizing radiation potential",
            "LOP": "Land occupation potential",
            "SOP": "Mineral resource scarcity potential",
            "ODP": "Ozone depletion potential",
            "PMFP": "Particulate matter formation potential",
            "HOFP": "Ozone formation potential, human health",
            "EOFP": "Ozone formation potential, ecosystems",
            "WCP": "Water consumption potential",
            "FFP": "Fossil resource scarcity potential",
        },
    },
}

HEADER_CSS = """
<style>
:root {
    --swcsp-header-height: 78px;
    --swcsp-header-bg: #EFEFEF;
    --swcsp-blue: #0B1F8A;
    --swcsp-blue-2: #1F5BD8;
    --swcsp-border: #D8D8D8;
    --page-max-width: 1440px;
}

html, body {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
    padding-top: 0 !important;
}

div[data-testid="stHorizontalBlock"],
div[data-testid="stHorizontalBlock"] > div {
    min-width: 0 !important;
    max-width: 100% !important;
}

.swcsp-fixed-header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: var(--swcsp-header-height);
    background: var(--swcsp-header-bg);
    z-index: 999990;
    border-bottom: 1px solid var(--swcsp-border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 52px;
    box-sizing: border-box;
}

.swcsp-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    height: 100%;
}

.swcsp-brand-logo {
    width: 50px;
    height: 50px;
    display: block;
    object-fit: contain;
}

.swcsp-brand-text {
    font-family: "Trebuchet MS", "Arial Rounded MT Bold", "Segoe UI", Arial, sans-serif;
    font-size: 38px;
    font-weight: 1000;
    color: var(--swcsp-blue);
    letter-spacing: 1.8px;
    line-height: 1;
    font-style: italic;
    transform: skew(-4deg);
    -webkit-text-stroke: 0.4px #0B1F8A;
    text-shadow:
        1px 1px 0 #1F5BD8,
        2px 2px 0 rgba(31, 91, 216, 0.18);
}

.st-key-language_selector {
    position: fixed !important;
    top: 20px !important;
    right: 52px !important;
    width: 128px !important;
    z-index: 1000000 !important;
}

.st-key-language_selector [data-testid="stSelectbox"] {
    width: 128px !important;
}

.st-key-language_selector [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #D9D9D9 !important;
    border-radius: 8px !important;
}

.st-key-language_selector [data-baseweb="select"] > div:focus-within {
    border-color: #0B1F8A !important;
    box-shadow: 0 0 0 1px #0B1F8A !important;
}

.st-key-language_selector [data-baseweb="select"] svg {
    color: #0B1F8A !important;
}

.st-key-language_selector label {
    display: none !important;
}

section[data-testid="stSidebar"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent;
    height: 0;
}

.block-container {
    width: 100% !important;
    min-width: 0 !important;
    max-width: 1440px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    padding-top: var(--swcsp-header-height) !important;
    box-sizing: border-box !important;
}

.swcsp-page-title {
    text-align: left;
    font-size: 24px;
    font-weight: 700;
    margin-top: -30px;
    margin-bottom: 16px;
    margin-left: 32px;
    color: var(--swcsp-blue) !important;
}

.st-key-left_input_panel {
    background: #EAF7FF;
    border-radius: 0;
    padding: 18px 18px 28px 18px;
    min-height: calc(100vh - var(--swcsp-header-height));
}

.sidebar-main-title {
    width: 78%;
    margin: 0 auto 22px auto;
    padding: 6px 0;
    background: #FFFFFF;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: 850;
    letter-spacing: 0px;
    color: #1F2937;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}

.st-key-composition_block,
.st-key-process_block {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 22px 18px 24px 18px;
    margin-bottom: 22px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}

.sidebar-section-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 22px !important;
    font-weight: 850 !important;
    color: #1F2937;
    margin-top: 0 !important;
    margin-bottom: 18px !important;
}

.sidebar-section-title img {
    width: 26px !important;
    height: 26px !important;
    display: block;
    object-fit: contain;
}

.custom-input-label {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #1F2937 !important;
    margin-bottom: 6px !important;
    line-height: 1.2 !important;
}

.st-key-process_block [data-testid="stWidgetLabel"] p,
.st-key-process_block label p {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #1F2937 !important;
}

.custom-input-label sub {
    font-size: 70% !important;
    vertical-align: sub !important;
    line-height: 0 !important;
}

.st-key-left_input_panel input {
    font-size: 16px !important;
}

.st-key-action_button_area {
    background: transparent !important;
    padding: 0 0 24px 0 !important;
    margin-top: -8px !important;
    overflow: visible !important;
}

.st-key-action_button_area div[data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-wrap: wrap !important;
    justify-content: space-between !important;
    gap: 8px !important;
    width: 100% !important;
}

.st-key-action_button_area div[data-testid="column"] {
    flex: 1 1 30% !important;
    min-width: 80px !important;
    width: auto !important;
    max-width: none !important;
    padding: 0 !important;
}

.st-key-run_button,
.st-key-reset_button,
.st-key-example_button {
    width: 100% !important;
}

.st-key-run_button button,
.st-key-reset_button button,
.st-key-example_button button {
    width: 100% !important;
    height: 38px !important;
    min-height: 38px !important;
    border-radius: 10px !important;
    padding: 2px 4px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.st-key-run_button button p,
.st-key-reset_button button p,
.st-key-example_button button p {
    font-size: 16px !important;
    font-weight: 600 !important;
    line-height: 1.1 !important;
    margin: 0 !important;
    white-space: nowrap !important;
}

.st-key-run_button button {
    background: #0B1F8A !important;
    color: #FFFFFF !important;
    border: 1.5px solid #0B1F8A !important;
}

.st-key-reset_button button {
    background: #FFFFFF !important;
    color: #0B1F8A !important;
    border: 1.5px solid #0B1F8A !important;
}

.st-key-example_button button {
    background: #EFEFEF !important;
    color: #0B1F8A !important;
    border: 1.5px solid #0B1F8A !important;
}

.st-key-run_button button:hover,
.st-key-reset_button button:hover,
.st-key-example_button button:hover {
    border-color: #0B1F8A !important;
    box-shadow: 0 0 0 1px rgba(11, 31, 138, 0.18) !important;
}

div[data-testid="stTabs"] {
    margin-top: 0px !important;
}

div[data-testid="stTabs"] div[role="tablist"] {
    position: relative !important;
    gap: 0 !important;
    border-bottom: none !important;
}

div[data-testid="stTabs"] div[role="tablist"]::after {
    content: "" !important;
    position: absolute !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    height: 2px !important;
    background: #E5E7EB !important;
    z-index: 0 !important;
}

div[data-testid="stTabs"] button[role="tab"] {
    position: relative !important;
    z-index: 1 !important;
    min-width: 150px !important;
    height: 46px !important;
    padding: 0 22px !important;
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-bottom: none !important;
    border-radius: 0 !important;
    margin-right: -1px !important;
}

div[data-testid="stTabs"] button[role="tab"] p {
    font-size: 18px !important;
    font-weight: 400 !important;
    color: #1F2937 !important;
    line-height: 1 !important;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    border-color: #E5E7EB !important;
    border-bottom: 2px solid #0B1F8A !important;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
    color: #0B1F8A !important;
    font-weight: 800 !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: transparent !important;
}

:root {
    --card-unified-height: 520px;
    --card-inner-height: 410px;
    --card-unified-padding: 22px 24px;
}

div[data-testid="stHorizontalBlock"]:has(.co2-result-wrap) {
    align-items: stretch !important;
}

div[data-testid="stHorizontalBlock"]:has(.co2-result-wrap) > div[data-testid="column"] {
    display: flex !important;
    align-items: stretch !important;
    padding-top: 0 !important;
}

.co2-result-wrap {
    width: calc(100% - 24px) !important;
    margin-left: 24px !important;
    margin-top: 0 !important;
    margin-bottom: 15px !important;
    padding-left: 0 !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: stretch !important;
}

.co2-result-outer {
    width: 100% !important;
    max-width: 100% !important;
    height: var(--card-unified-height) !important;
    min-height: var(--card-unified-height) !important;
    max-height: var(--card-unified-height) !important;
    background: #E8F7FF;
    border-radius: 6px;
    padding: var(--card-unified-padding) !important;
    box-sizing: border-box !important;
}

.st-key-shap_card_box {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    height: var(--card-unified-height) !important;
    min-height: var(--card-unified-height) !important;
    max-height: var(--card-unified-height) !important;
    background: #E8F7FF !important;
    border-radius: 6px !important;
    padding: var(--card-unified-padding) !important;
    box-sizing: border-box !important;
    overflow: visible !important;
}

.co2-result-title,
.shap-card-title {
    color: #0B1F8A;
    font-size: 24px;
    font-weight: 820;
    margin: 0 0 16px 0 !important;
    line-height: 1.2;
}

.co2-result-middle {
    background: #FFFFFF;
    border-radius: 6px;
    padding: 22px 24px 24px 24px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: flex-start !important;
    gap: 20px !important;
    height: var(--card-inner-height) !important;
}

.st-key-shap_plot_inner {
    background: #FFFFFF !important;
    height: var(--card-inner-height) !important;
    min-height: var(--card-inner-height) !important;
    max-height: var(--card-inner-height) !important;
    padding: 0 !important;
    margin: 7px 0 0 0 !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
}

.st-key-shap_plot_inner [data-testid="stPlotlyChart"] {
    height: var(--card-inner-height) !important;
    min-height: var(--card-inner-height) !important;
    max-height: var(--card-inner-height) !important;
    margin: 0 !important;
    background: #FFFFFF !important;
}

.st-key-shap_plot_inner [data-testid="stPlotlyChart"] > div {
    height: var(--card-inner-height) !important;
    min-height: var(--card-inner-height) !important;
    max-height: var(--card-inner-height) !important;
}

.co2-metric-block {
    margin: 0 !important;
}

.co2-metric-title {
    color: #0B1F8A;
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 12px;
    display: flex;
    align-items: flex-start;
    gap: 4px;
}

.co2-value-card {
    position: relative;
    background: #E8F7FF;
    border-radius: 8px;
    height: 130px;
    padding: 15px 14px;
    box-sizing: border-box;
}

.co2-big-value {
    color: #0B1F8A;
    font-size: 44px;
    font-weight: 880;
    line-height: 1.05;
    margin-top: 26px;
    text-align: center;
}

.co2-unit {
    position: absolute;
    right: 18px;
    bottom: 16px;
    color: #111827;
    font-size: 15px;
    font-weight: 800;
    line-height: 1;
}

.st-key-shap_card_box [data-testid="stPlotlyChart"] {
    margin-top: 0 !important;
    background: transparent !important;
}

.st-key-shap_card_box [data-testid="stPlotlyChart"] > div {
    width: 100% !important;
    max-width: 100% !important;
}

.st-key-shap_card_box [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

.st-key-shap_card_box .xtick text,
.st-key-shap_card_box .ytick text,
.st-key-shap_card_box .g-xtitle text,
.st-key-shap_card_box .g-ytitle text {
    font-weight: 800 !important;
}

.st-key-lca_impact_section .xtick text,
.st-key-lca_impact_section .ytick text,
.st-key-lca_impact_section .g-xtitle text,
.st-key-lca_impact_section .g-ytitle text {
    font-weight: 800 !important;
}

.st-key-shap_card_box text.bartext,
.st-key-shap_card_box .bartext,
.st-key-shap_card_box .bartext-outside {
    font-weight: 400 !important;
}

.theory-tip {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 12px !important;
    height: 12px !important;
    border: 1px solid #9CA3AF !important;
    border-radius: 50% !important;
    position: relative !important;
    top: -7px !important;
    margin-left: 4px !important;
    cursor: help !important;
    box-sizing: border-box !important;
    font-size: 0 !important;
    color: transparent !important;
    line-height: 0 !important;
    text-decoration: none !important;
}

.theory-tip::before {
    content: "?" !important;
    color: #9CA3AF !important;
    font-size: 8px !important;
    font-weight: 700 !important;
    line-height: 1 !important;
}

.theory-tip::after {
    content: attr(data-tooltip) !important;
    position: absolute !important;
    left: 18px !important;
    top: -10px !important;
    width: max-content !important;
    max-width: 260px !important;
    background: rgba(255, 255, 255, 0.98) !important;
    color: #374151 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 8px !important;
    padding: 8px 10px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    line-height: 1.35 !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12) !important;
    opacity: 0 !important;
    visibility: hidden !important;
    transform: translateY(2px) !important;
    transition: opacity 0.12s ease, transform 0.12s ease !important;
    z-index: 9999 !important;
    pointer-events: none !important;
    white-space: normal !important;
}

.theory-tip:hover::after {
    opacity: 1 !important;
    visibility: visible !important;
    transform: translateY(0) !important;
}

.empty-result-hint-wrap {
    width: 100%;
    display: flex;
    justify-content: center;
    margin-top: 18px;
}

.empty-result-hint {
    width: min(620px, 70%);
    background: #E8F2FF;
    color: #005BAC;
    border-radius: 8px;
    padding: 16px 24px;
    text-align: center;
    font-size: 18px;
    font-weight: 500;
    line-height: 1.6;
}

div[data-testid="stHorizontalBlock"]:has(.st-key-left_input_panel) {
    gap: 0px !important;
}

.st-key-records_section {
    margin-left: 28px !important;
    width: calc(100% - 28px) !important;
    max-width: calc(100% - 28px) !important;
    box-sizing: border-box !important;
}

.st-key-records_section [data-testid="stDataFrame"] {
    width: 100% !important;
}

.st-key-clear_records_button button {
    height: 38px !important;
    min-height: 38px !important;
    border-radius: 10px !important;
}

.st-key-lca_impact_section {
    margin-left: 28px !important;
    width: calc(100% - 28px) !important;
    max-width: calc(100% - 28px) !important;
    box-sizing: border-box !important;
}

.lca-impact-title {
    font-size: 24px;
    font-weight: 800;
    margin: 0 0 10px 0;
    color: #1F2937;
    line-height: 1.2;
}

.st-key-lca_impact_section [data-testid="stPlotlyChart"] {
    width: 100% !important;
}

.st-key-lca_impact_section [data-testid="stExpander"] {
    width: 100% !important;
}

div[data-testid="stHorizontalBlock"]:has(.lca-card-stack) {
    align-items: stretch !important;
}

.lca-card-stack {
    width: calc(100% - 24px) !important;
    margin-left: 24px !important;
    display: flex;
    flex-direction: column;
    gap: 20px;
    margin-top: 0;
    box-sizing: border-box !important;
}

.lca-single-card {
    background: #FFFFFF;
    border-radius: 6px;
    padding: 22px 24px 24px 24px;
    box-sizing: border-box;
}

.lca-single-card .co2-metric-title {
    margin-bottom: 12px !important;
}

.lca-single-card .co2-value-card {
    height: 130px;
}

.lca-single-card .co2-big-value {
    margin-top: 26px;
}

div[data-testid="stHorizontalBlock"]:has(.lca-result-wrap) {
    align-items: stretch !important;
}

div[data-testid="stHorizontalBlock"]:has(.lca-result-wrap) > div[data-testid="column"] {
    display: flex !important;
    align-items: stretch !important;
    padding-top: 0 !important;
}

.lca-result-wrap {
    width: calc(100% - 24px) !important;
    margin-left: 24px !important;
    margin-top: 0 !important;
    margin-bottom: 15px !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: stretch !important;
}

.lca-result-outer {
    width: 100% !important;
    max-width: 100% !important;
    height: var(--card-unified-height) !important;
    min-height: var(--card-unified-height) !important;
    max-height: var(--card-unified-height) !important;
    background: #E8F7FF;
    border-radius: 6px;
    padding: var(--card-unified-padding) !important;
    box-sizing: border-box !important;
    display: flex;
    flex-direction: column;
}

.lca-inner-stack {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 20px;
    min-height: 0;
}

.lca-metric-block {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
}

.lca-metric-title {
    color: #0B1F8A;
    font-size: 21px;
    font-weight: 800;
    margin: 0 0 12px 0;
    line-height: 1.2;
}

.lca-metric-card {
    flex: 1;
    position: relative;
    background: #FFFFFF;
    border-radius: 6px;
    padding: 15px 14px;
    box-sizing: border-box;
    min-height: 130px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.lca-metric-value {
    color: #0B1F8A;
    font-size: 44px;
    font-weight: 880;
    line-height: 1.05;
    text-align: center;
    margin-top: 10px;
}

.lca-metric-unit {
    position: absolute;
    right: 18px;
    bottom: 16px;
    color: #111827;
    font-size: 15px;
    font-weight: 800;
    line-height: 1;
}

.lca-chart-title {
    text-align: center;
    font-size: 24px;
    font-weight: 850;
    color: #111827;
    margin: 0 0 8px 0;
}

.lca-cost-chart-wrap {
    margin-top: 32px;
}

</style>
"""


def resolve_asset_path(*names: str) -> Path | None:
    for name in names:
        for candidate in (ASSETS_DIR / name, ROOT / name):
            if candidate.exists():
                return candidate
    return None


def svg_data_uri(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    svg_text = path.read_text(encoding="utf-8")
    encoded = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main {
            background: #f7fbff;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_lca_detail_display_df(lca_df: pd.DataFrame, t: dict) -> pd.DataFrame:
    display_df = lca_df.copy()

    if "Impact" in display_df.columns:
        impact_full_names = t.get("lca_impact_full_names", {})
        display_df.insert(
            display_df.columns.get_loc("Impact") + 1,
            "Impact_full_name",
            display_df["Impact"].astype(str).map(lambda x: impact_full_names.get(x, x)),
        )

    unit_replace_map = {
        "kg SO2-Eq": "kg SO₂-Eq",
        "kg CO2-Eq": "kg CO₂-Eq",
        "m2*a crop-Eq": "m²·a crop-Eq",
        "kg PM2.5-Eq": "kg PM₂.₅-Eq",
        "m3": "m³",
    }

    display_df["Unit"] = display_df["Unit"].replace(unit_replace_map)

    column_order = [
        "Impact",
        "Impact_full_name",
        "Unit",
        "Electricity",
        "Heat",
        "Water",
        "Fixed_CO2",
        "Positive_total",
        "Total_with_fixed",
        "Electricity_%",
        "Heat_%",
        "Water_%",
        "Fixed_CO2_%",
    ]

    existing_cols = [col for col in column_order if col in display_df.columns]
    other_cols = [col for col in display_df.columns if col not in existing_cols]
    display_df = display_df[existing_cols + other_cols]

    display_df = display_df.rename(columns=t.get("lca_detail_columns", {}))

    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = t.get("lca_detail_index", "序号")

    return display_df

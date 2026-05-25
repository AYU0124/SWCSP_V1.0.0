from __future__ import annotations

import streamlit as st

TEXT = {
    "zh_CN": {
        "title": "基于 LGBM 的钢渣 CO₂ 固存量网页预测计算器",
        "language": "简体中文",
        "input_params": "输入参数",
        "composition": "组分特征",
        "process": "工艺参数",
        "prediction_page": "预测结果",
        "lca_page": "生命周期评价及成本分析",
        "execute": "执行预测",
        "reset": "重置输入",
        "example": "导入示例",
        "co2_result": "CO₂吸收量",
        "theoretical": "理论值",
        "current_shap": "当前样本贡献分析",
        "records": "预测记录表",
        "copy_hint": "可在表格中选择并复制；也可下载 CSV。",
        "download_csv": "导出 CSV",
        "net_carbon": "净碳排",
        "total_cost": "工艺总成本",
        "cost_share": "成本贡献分析",
        "impact_share": "18项环境影响贡献分析",
        "error_title": "请检查红色输入框：有参数缺失或超出范围。",
        "warning_formula": "提示：Word 中 Q_heat 与 Q_dry 公式不完整。当前代码默认 Q_heat=加热水耗至目标温度，Q_dry=0，可在 calculations.py 中替换。",
    },
    "zh_TW": {
        "title": "基於 LGBM 的鋼渣 CO₂ 固存量網頁預測計算器",
        "language": "繁體中文",
        "input_params": "輸入參數",
        "composition": "組分特徵",
        "process": "工藝參數",
        "prediction_page": "預測結果",
        "lca_page": "生命週期評價及成本分析",
        "execute": "執行預測",
        "reset": "重置輸入",
        "example": "匯入範例",
        "co2_result": "CO₂吸收量",
        "theoretical": "理論值",
        "current_shap": "目前樣本貢獻分析",
        "records": "預測紀錄表",
        "copy_hint": "可在表格中選擇並複製；也可下載 CSV。",
        "download_csv": "匯出 CSV",
        "net_carbon": "淨碳排",
        "total_cost": "工藝總成本",
        "cost_share": "成本貢獻分析",
        "impact_share": "18項環境影響貢獻分析",
        "error_title": "請檢查紅色輸入框：有參數缺失或超出範圍。",
        "warning_formula": "提示：Word 中 Q_heat 與 Q_dry 公式不完整。當前程式預設 Q_heat=加熱水耗至目標溫度，Q_dry=0，可在 calculations.py 中替換。",
    },
    "en": {
        "title": "LGBM-based Web Calculator for CO₂ Sequestration in Steel Slag",
        "language": "English",
        "input_params": "Input parameters",
        "composition": "Composition features",
        "process": "Process parameters",
        "prediction_page": "Prediction results",
        "lca_page": "Life cycle assessment and cost analysis",
        "execute": "Run prediction",
        "reset": "Reset inputs",
        "example": "Load example",
        "co2_result": "CO₂ uptake",
        "theoretical": "Theoretical value",
        "current_shap": "Current sample contribution analysis",
        "records": "Prediction records",
        "copy_hint": "Select and copy cells in the table, or download CSV.",
        "download_csv": "Export CSV",
        "net_carbon": "Net carbon emission",
        "total_cost": "Total process cost",
        "cost_share": "Cost contribution analysis",
        "impact_share": "18 impact category contribution analysis",
        "error_title": "Please check red input boxes: missing or out-of-range values.",
        "warning_formula": "Note: Q_heat and Q_dry formulas were incomplete in the Word file. Current defaults: Q_heat = heating water to target temperature; Q_dry = 0. Edit calculations.py after confirmation.",
    },
}

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main { background: #f7fbff; }
        .block-container { padding-top: 1.2rem; max-width: 1220px; }
        [data-testid="stSidebar"] { background: #eaf7ff; }
        .result-card {
            background: #e8f7ff;
            border-radius: 14px;
            padding: 18px 20px;
            min-height: 112px;
            border: 1px solid #d7edf7;
            box-shadow: 0 2px 10px rgba(20, 90, 140, 0.06);
        }
        .result-card h4 {
            margin: 0 0 14px 0;
            font-size: 1.02rem;
            color: #17446a;
        }
        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #0a2f88;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        .unit {
            margin-top: 6px;
            color: #334155;
            font-weight: 600;
        }
        .small-note {
            color: #475569;
            font-size: .88rem;
        }
        div.stButton > button {
            border-radius: 9px;
            min-height: 2.5rem;
            font-weight: 650;
        }
        div[data-testid="stAlert"] {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def card(title: str, value: str, unit: str = "") -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <h4>{title}</h4>
            <div class="metric-value">{value}</div>
            <div class="unit">{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

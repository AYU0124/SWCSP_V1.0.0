import streamlit as st


TEXT = {
    "简体中文": {
        "main_title": "基于 LGBM 的钢渣 CO₂ 固存量网页预测计算器",
        "input_parameters": "输入参数",
        "composition_features": "组分特征",
        "process_parameters": "工艺参数",
        "liquid_solid_ratio": "液固比",
        "particle_size": "粒径",
        "temperature": "温度",
        "reaction_time": "反应时间",
        "co2_pressure": "CO2分压",
        "co2_concentration": "CO2浓度",
        "prediction_result": "预测结果",
        "lca_cost": "生命周期评价及成本分析",
        "co2_uptake": "CO2吸收量",
        "predicted_value": "预测值",
        "theoretical_value": "理论值",
        "current_sample_contribution": "当前样本贡献分析",
        "execute_prediction": "执行预测",
        "reset_input": "重置输入",
        "import_example": "导入示例",
        "net_carbon_reduction": "净碳排",
        "process_cost": "工艺成本",
        "cost_contribution": "成本贡献分析",
        "impact_contribution": "18环境影响贡献分析",
        "lca_notice": "完成一次有效预测后，将自动显示生命周期评价及成本分析。",
    },
    "繁體中文": {
        "main_title": "基於 LGBM 的鋼渣 CO₂ 固存量網頁預測計算器",
        "input_parameters": "輸入參數",
        "composition_features": "組分特徵",
        "process_parameters": "工藝參數",
        "liquid_solid_ratio": "液固比",
        "particle_size": "粒徑",
        "temperature": "溫度",
        "reaction_time": "反應時間",
        "co2_pressure": "CO2分壓",
        "co2_concentration": "CO2濃度",
        "prediction_result": "預測結果",
        "lca_cost": "生命週期評價及成本分析",
        "co2_uptake": "CO2吸收量",
        "predicted_value": "預測值",
        "theoretical_value": "理論值",
        "current_sample_contribution": "當前樣本貢獻分析",
        "execute_prediction": "執行預測",
        "reset_input": "重置輸入",
        "import_example": "導入示例",
        "net_carbon_reduction": "淨碳排",
        "process_cost": "工藝成本",
        "cost_contribution": "成本貢獻分析",
        "impact_contribution": "18環境影響貢獻分析",
        "lca_notice": "完成一次有效預測後，將自動顯示生命週期評價及成本分析。",
    },
    "English": {
        "main_title": "LGBM-based Web Calculator for CO₂ Sequestration of Steel Slag",
        "input_parameters": "Input Parameters",
        "composition_features": "Composition Features",
        "process_parameters": "Process Parameters",
        "liquid_solid_ratio": "L/S",
        "particle_size": "Particle size",
        "temperature": "Temperature",
        "reaction_time": "Time",
        "co2_pressure": "CO2 pressure",
        "co2_concentration": "CO2 concentration",
        "prediction_result": "Prediction Result",
        "lca_cost": "LCA and Cost Analysis",
        "co2_uptake": "CO2 Uptake",
        "predicted_value": "Predicted value",
        "theoretical_value": "Theoretical value",
        "current_sample_contribution": "Current Sample Contribution",
        "execute_prediction": "Run Prediction",
        "reset_input": "Reset",
        "import_example": "Import Example",
        "net_carbon_reduction": "Net carbon reduction",
        "process_cost": "Process cost",
        "cost_contribution": "Cost Contribution",
        "impact_contribution": "18 Environmental Impact Contribution",
        "lca_notice": "LCA and cost results appear after a valid prediction.",
    },
}


def t(key: str, lang: str) -> str:
    return TEXT.get(lang, TEXT["简体中文"]).get(key, key)


def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --main-blue: #001489;
            --light-blue: #eaf8ff;
            --card-blue: #e7f6fd;
            --deep-text: #00136b;
            --border-gray: #d7d7d7;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: #ffffff;
            font-family: "Arial", "Microsoft YaHei", sans-serif;
        }

        [data-testid="stHeader"] {
            display: none;
        }

        [data-testid="stToolbar"] {
            display: none;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        .block-container {
            padding-top: 0 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            max-width: 100% !important;
        }

        .top-header {
            width: 100%;
            height: 72px;
            background: #eeeeee;
            border: 1px solid #d1d1d1;
            display: flex;
            align-items: center;
            padding: 0 58px;
            box-sizing: border-box;
        }

        .brand {
            display: flex;
            align-items: center;
            height: 72px;
        }

        .logo-circle {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: radial-gradient(circle at center, #0e4fae 0%, #0e4fae 30%, white 32%, white 55%, #b9c7df 57%, #b9c7df 100%);
            color: #0a2f88;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            margin-right: 10px;
            font-weight: 700;
        }

        .brand-text {
            color: #000f88;
            font-size: 36px;
            font-weight: 800;
            letter-spacing: 1px;
        }

        .main-title {
            font-size: 28px;
            font-weight: 500;
            color: #111;
            margin: 28px 0 26px 60px;
        }

        div[data-testid="column"] {
            padding: 0 !important;
        }

        .left-panel {
            background: var(--light-blue);
            width: 100%;
            min-height: 945px;
            padding: 22px 26px 24px 26px;
            box-sizing: border-box;
            margin-left: 76px;
        }

        .panel-title {
            background: white;
            color: #008cff;
            font-weight: 700;
            text-align: center;
            padding: 8px 0;
            border-radius: 6px;
            font-size: 20px;
            margin-bottom: 12px;
        }

        .section-title {
            color: #111;
            font-size: 24px;
            font-weight: 700;
            margin: 14px 0 10px 0;
            line-height: 1.2;
        }

        .input-card {
            background: white;
            border-radius: 7px;
            padding: 11px 13px 12px 13px;
            box-sizing: border-box;
        }

        .input-row {
            margin-bottom: 8px;
        }

        .param-label {
            font-size: 19px;
            line-height: 36px;
            color: #111;
            text-align: right;
            padding-right: 6px;
            white-space: nowrap;
        }

        .unit-text {
            font-size: 16px;
            font-weight: 700;
            line-height: 36px;
            color: #111;
            padding-left: 4px;
        }

        .range-text {
            color: #777;
            font-size: 12px;
            text-align: right;
            margin-top: -3px;
            padding-right: 3px;
        }

        .divider {
            border-top: 1px solid #c5d4dc;
            margin: 30px 0 28px 0;
        }

        .right-area {
            margin-left: 20px;
            margin-right: 76px;
        }

        .blue-card {
            background: #e7f6fd;
            border-radius: 7px;
            padding: 20px 22px;
            min-height: 330px;
            box-sizing: border-box;
        }

        .white-card {
            background: #ffffff;
            border-radius: 7px;
            padding: 20px 22px;
            min-height: 330px;
            box-sizing: border-box;
            border: 1px solid #f0f0f0;
        }

        .block-title {
            color: #001489;
            font-size: 20px;
            font-weight: 800;
            margin-bottom: 10px;
        }

        .result-box {
            margin: 10px 0 18px 0;
        }

        .result-title {
            color: #001489;
            font-size: 20px;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .value-area {
            height: 108px;
            background: #e8f8ff;
            border-radius: 7px;
            position: relative;
        }

        .result-number {
            position: absolute;
            left: 18px;
            bottom: 18px;
            color: #111;
            font-size: 28px;
            font-weight: 800;
        }

        .result-unit {
            position: absolute;
            right: 12px;
            bottom: 18px;
            color: #111;
            font-size: 17px;
            font-weight: 800;
        }

        .button-row {
            margin: 18px 65px 20px 65px;
        }

        .history-card {
            background: white;
            border-radius: 8px;
            border: 1px solid #ededed;
            padding: 8px 8px 12px 8px;
            min-height: 335px;
            box-sizing: border-box;
        }

        .lca-summary-card {
            min-height: 365px;
            padding: 36px 42px;
        }

        .impact-card {
            margin-top: 20px;
            min-height: 500px;
            padding: 26px 36px;
        }

        .impact-title {
            text-align: center;
            color: #001489;
            font-size: 20px;
            font-weight: 800;
            margin-bottom: 10px;
        }

        div.stButton > button {
            border-radius: 5px;
            border: 1px solid #001489;
            color: #111;
            background: white;
            font-size: 18px;
            font-weight: 700;
            height: 38px;
            padding: 0 12px;
        }

        div.stButton > button:hover {
            border: 1px solid #001489;
            color: #001489;
            background: #eef8ff;
        }

        div.stButton > button:active,
        div.stButton > button:focus {
            background: #001489;
            color: white;
            border: 1px solid #001489;
        }

        div[data-testid="stNumberInput"] input {
            height: 32px;
            font-size: 16px;
            text-align: center;
            border: none;
            background: white;
            padding: 0;
        }

        div[data-testid="stNumberInput"] button {
            display: none;
        }

        div[data-baseweb="select"] {
            height: 34px;
        }

        div[data-baseweb="select"] > div {
            min-height: 34px;
            height: 34px;
            border-radius: 6px;
            border-color: #e2e2e2;
            font-size: 14px;
        }

        div[data-testid="stDataFrame"] {
            font-size: 13px;
        }

        .stAlert {
            margin: 0 0 14px 0;
        }

        @media (max-width: 1100px) {
            .left-panel {
                margin-left: 20px;
            }

            .right-area {
                margin-right: 20px;
            }

            .main-title {
                margin-left: 24px;
            }

            .top-header {
                padding: 0 24px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

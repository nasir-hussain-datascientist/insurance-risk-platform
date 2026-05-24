import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Insurance Risk AI Platform",
    page_icon="📊",
    layout="wide"
)

# =========================
# CLEAN MODERN UI
# =========================
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #F7F9FC;
}

/* HEADER */
.header {
    background: linear-gradient(90deg, #2563EB, #60A5FA);
    padding: 25px;
    border-radius: 14px;
    color: white;
    margin-bottom: 20px;
}

/* TITLE */
.title {
    font-size: 34px;
    font-weight: 800;
}

/* SUBTITLE */
.subtitle {
    font-size: 15px;
    opacity: 0.9;
    margin-top: 4px;
}

/* DEVELOPER BOX (VERY PROMINENT) */
.dev {
    margin-top: 12px;
    padding: 12px;
    background: rgba(255,255,255,0.2);
    border-radius: 10px;
    font-size: 14px;
}

/* CARDS */
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    text-align: center;
}

/* METRIC TEXT */
.metric {
    font-size: 28px;
    font-weight: 700;
    color: #111827;
}

.label {
    font-size: 13px;
    color: #6B7280;
}

</style>

<div class="header">
    <div class="title">Insurance Risk Analytics Platform</div>
    <div class="subtitle">
        AI-powered claim prediction, risk scoring & customer risk analysis
    </div>

    <div class="dev">
        <b>Developed by:</b> Nasir Hussain (Data Scientist) <br>
        <b>Email:</b> nasir.swat.hussain@gmail.com
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.title("Customer Information")

age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
bloodpressure = st.sidebar.slider("Blood Pressure", 80, 200, 120)
children = st.sidebar.slider("Children", 0, 10, 0)

gender = st.sidebar.selectbox("Gender", ["male", "female"])
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
diabetic = st.sidebar.selectbox("Diabetic", ["no", "yes"])

region = st.sidebar.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# =========================
# FEATURE ENGINEERING
# =========================
input_data = dict.fromkeys(columns, 0)

input_data["age"] = age
input_data["bmi"] = bmi
input_data["bloodpressure"] = bloodpressure
input_data["children"] = children

input_data["gender_male"] = 1 if gender == "male" else 0
input_data["smoker_yes"] = 1 if smoker == "yes" else 0
input_data["diabetic_yes"] = 1 if diabetic == "yes" else 0

region_col = f"region_{region}"
if region_col in input_data:
    input_data[region_col] = 1

# BMI categories
if bmi < 18.5:
    input_data["bmi_category_Underweight"] = 1
elif bmi < 25:
    pass
elif bmi < 30:
    input_data["bmi_category_Overweight"] = 1
else:
    input_data["bmi_category_Obese"] = 1

# =========================
# PREDICTION
# =========================
input_df = pd.DataFrame([input_data])[columns]
prediction = model.predict(input_df)[0]

# =========================
# RISK SCORE
# =========================
risk_score = min(100, max(0, (prediction / 50000) * 100))

if risk_score < 33:
    risk_level = "LOW"
elif risk_score < 66:
    risk_level = "MEDIUM"
else:
    risk_level = "HIGH"

# =========================
# GAUGE CHART
# =========================
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_score,
    title={"text": "Risk Score"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "#2563EB"},
        "steps": [
            {"range": [0, 33], "color": "#DCFCE7"},
            {"range": [33, 66], "color": "#FEF9C3"},
            {"range": [66, 100], "color": "#FECACA"},
        ],
    }
))

# =========================
# KPI CARDS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="label">Predicted Claim</div>
        <div class="metric">${:,.0f}</div>
    </div>
    """.format(prediction), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="label">Risk Score</div>
        <div class="metric">{:.0f}/100</div>
    </div>
    """.format(risk_score), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="label">Risk Level</div>
        <div class="metric">{}</div>
    </div>
    """.format(risk_level), unsafe_allow_html=True)

st.plotly_chart(fig, use_container_width=True)

# =========================
# WHAT-IF ANALYSIS
# =========================
st.subheader("Scenario Analysis")

if smoker == "yes":
    alt = input_data.copy()
    alt["smoker_yes"] = 0

    alt_df = pd.DataFrame([alt])[columns]
    new_pred = model.predict(alt_df)[0]

    st.success(f"After quitting smoking: ${new_pred:,.0f}")
    st.info(f"Potential Savings: ${prediction - new_pred:,.0f}")

# =========================
# SHAP EXPLANATION
# =========================
st.subheader("Feature Importance (Explainable AI)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

fig2, ax = plt.subplots()
shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
st.pyplot(fig2)

# =========================
# PDF REPORT
# =========================
def generate_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Insurance Risk Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Age: {age}", styles["Normal"]))
    story.append(Paragraph(f"BMI: {bmi}", styles["Normal"]))
    story.append(Paragraph(f"Blood Pressure: {bloodpressure}", styles["Normal"]))
    story.append(Paragraph(f"Smoker: {smoker}", styles["Normal"]))
    story.append(Paragraph(f"Predicted Claim: ${prediction:,.0f}", styles["Normal"]))
    story.append(Paragraph(f"Risk Score: {risk_score:.0f}", styles["Normal"]))
    story.append(Paragraph(f"Risk Level: {risk_level}", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

pdf = generate_pdf()

st.download_button(
    "Download Insurance Report (PDF)",
    pdf,
    file_name="insurance_report.pdf",
    mime="application/pdf"
)

# =========================
# FOOTER
# =========================
st.markdown("""
---
<div style="text-align:center; color:#6B7280; font-size:13px;">
© 2026 Nasir Hussain — Data Science Portfolio Project | Insurance Risk AI Platform
</div>
""", unsafe_allow_html=True)
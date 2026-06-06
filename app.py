import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# -------------------------
# TITLE
# -------------------------
st.title("🔧 AI Predictive Maintenance Digital Twin")
st.markdown("Industry 4.0 Smart Maintenance Dashboard")

# -------------------------
# LOAD DATA
# -------------------------
try:
    df = pd.read_csv("industrial_drilling_dataset.csv")
    st.success("Dataset Loaded Successfully")
except Exception as e:
    st.error(f"Dataset Error: {e}")
    st.stop()

# -------------------------
# SHOW DATA
# -------------------------
with st.expander("View Dataset"):
    st.dataframe(df.head())

# -------------------------
# COLUMN CHECK
# -------------------------
st.subheader("Dataset Columns")

st.write(df.columns.tolist())

# Replace these names if your CSV uses different columns
feature_columns = df.columns[:4]

# Last column assumed target
target_column = df.columns[-1]

# -------------------------
# TRAIN MODEL
# -------------------------
try:
    X = df[feature_columns]
    y = df[target_column]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

except Exception as e:
    st.error(f"Model Training Error: {e}")
    st.stop()

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("Machine Inputs")

temperature = st.sidebar.slider(
    "Temperature",
    0.0,
    200.0,
    50.0
)

vibration = st.sidebar.slider(
    "Vibration",
    0.0,
    20.0,
    2.0
)

rpm = st.sidebar.slider(
    "RPM",
    0.0,
    10000.0,
    2000.0
)

tool_wear = st.sidebar.slider(
    "Tool Wear",
    0.0,
    100.0,
    10.0
)

# -------------------------
# KPI CARDS
# -------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Temperature", f"{temperature:.1f} °C")
col2.metric("Vibration", f"{vibration:.2f}")
col3.metric("RPM", f"{rpm:.0f}")
col4.metric("Tool Wear", f"{tool_wear:.0f}%")

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict Failure"):

    input_data = np.array([
        [temperature, vibration, rpm, tool_wear]
    ])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠ Failure Predicted")
    else:
        st.success("✅ Machine Healthy")

# -------------------------
# CHART
# -------------------------
st.subheader("Sensor Trend")

chart_col = feature_columns[0]

fig = px.line(
    df,
    y=chart_col,
    title=f"{chart_col} Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -------------------------
# DIGITAL TWIN STATUS
# -------------------------
st.subheader("Digital Twin Status")

health_score = max(
    0,
    100 - tool_wear
)

st.progress(int(health_score))

st.write(
    f"Machine Health Score: {health_score:.0f}%"
)

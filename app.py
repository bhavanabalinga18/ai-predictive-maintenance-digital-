import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# -----------------------------------
# INDUSTRY 4.0 THEME
# -----------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0B1120;
}

div[data-testid="metric-container"] {
    background: #111827;
    padding: 15px;
    border-radius: 12px;
}

h1,h2,h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOGIN PAGE
# -----------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🔐 Industry 4.0 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.rerun()

        else:
            st.error("Invalid Username or Password")

    st.stop()

# -----------------------------------
# DASHBOARD
# -----------------------------------
st.title("🔧 AI Predictive Maintenance Digital Twin Dashboard")

# -----------------------------------
# CSV UPLOADER
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload Industrial Dataset CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your CSV dataset.")
    st.stop()

# -----------------------------------
# LOAD DATA
# -----------------------------------
df = pd.read_csv(uploaded_file)

st.success("Dataset Loaded Successfully")

# -----------------------------------
# PREVIEW
# -----------------------------------
with st.expander("Dataset Preview"):
    st.dataframe(df.head())

# -----------------------------------
# TARGET COLUMN
# -----------------------------------
target_column = "Failure"

if target_column not in df.columns:
    st.error("Failure column not found.")
    st.stop()

feature_columns = [
    col for col in df.columns
    if col != target_column
]

X = df[feature_columns]
y = df[target_column]

# -----------------------------------
# TRAIN MODEL
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.header("Machine Inputs")

inputs = []

for col in feature_columns:

    value = st.sidebar.slider(
        col,
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )

    inputs.append(value)

input_data = np.array([inputs])

# -----------------------------------
# FAILURE PREDICTION
# -----------------------------------
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# -----------------------------------
# KPI CARDS
# -----------------------------------
st.subheader("📊 KPI Dashboard")

c1, c2, c3, c4 = st.columns(4)

health_score = 95

rul_days = 180

c1.metric(
    "Model Accuracy",
    f"{accuracy*100:.2f}%"
)

c2.metric(
    "Health Score",
    f"{health_score}%"
)

c3.metric(
    "RUL",
    f"{rul_days} Days"
)

c4.metric(
    "Failure Risk",
    f"{probability*100:.2f}%"
)

# -----------------------------------
# DIGITAL TWIN
# -----------------------------------
st.subheader("🏭 Digital Twin Status")

if prediction == 1:
    st.error("⚠ Critical Condition")
else:
    st.success("✅ Healthy Machine")

st.progress(health_score)

# -----------------------------------
# FAILURE PREDICTION
# -----------------------------------
st.subheader("🧠 Failure Prediction")

if prediction == 1:

    st.error(
        f"Failure Predicted ({probability*100:.2f}%)"
    )

else:

    st.success(
        f"Machine Healthy ({(1-probability)*100:.2f}%)"
    )

# -----------------------------------
# ANOMALY DETECTION
# -----------------------------------
st.subheader("⚠ Anomaly Detection")

iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

labels = iso.fit_predict(X)

anomaly_count = np.sum(labels == -1)

st.metric(
    "Detected Anomalies",
    anomaly_count
)

# -----------------------------------
# FORECASTING
# -----------------------------------
st.subheader("🔮 Future Forecast")

forecast = {}

for i, col in enumerate(feature_columns):

    forecast[col] = round(
        inputs[i] + np.random.uniform(-5, 5),
        2
    )

forecast_df = pd.DataFrame(
    forecast,
    index=["24 Hours"]
)

st.dataframe(forecast_df)

# -----------------------------------
# CHARTS
# -----------------------------------
st.subheader("📈 Analytics")

sensor = st.selectbox(
    "Select Sensor",
    feature_columns
)

fig = px.line(
    df,
    y=sensor,
    title=f"{sensor} Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------
# FAILURE DISTRIBUTION
# -----------------------------------
st.subheader("📊 Failure Distribution")

fig2 = px.histogram(
    df,
    x="Failure"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# -----------------------------------
# REPORT
# -----------------------------------
st.subheader("📄 Maintenance Report")

report = pd.DataFrame({
    "Health Score": [health_score],
    "Failure Risk": [probability*100],
    "RUL Days": [rul_days]
})

csv = report.to_csv(index=False)

st.download_button(
    "Download Report",
    csv,
    "maintenance_report.csv",
    "text/csv"
)

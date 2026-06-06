
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Predictive Maintenance", page_icon="🔧", layout="wide")

st.markdown("""
<style>
.stApp {background-color:#0B1120;}
h1,h2,h3,p,label {color:white !important;}
div[data-testid="metric-container"]{
background:#111827;
padding:15px;
border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

# Login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Industry 4.0 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == "admin" and p == "admin123":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

st.title("🔧 AI Predictive Maintenance Digital Twin Dashboard")

# Load dataset
df = pd.read_csv("industrial_drilling_dataset.csv")

target = "Failure"
features = ["Temperature","Vibration","RPM","Torque","Tool_Wear","Pressure","Power"]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

# Sidebar inputs
st.sidebar.header("Machine Inputs")

temperature = st.sidebar.slider("Temperature", 0.0, 150.0, 70.0)
vibration = st.sidebar.slider("Vibration", 0.0, 20.0, 2.0)
rpm = st.sidebar.slider("RPM", 0, 10000, 2500)
torque = st.sidebar.slider("Torque", 0.0, 100.0, 40.0)
tool_wear = st.sidebar.slider("Tool Wear", 0.0, 100.0, 20.0)
pressure = st.sidebar.slider("Pressure", 0.0, 50.0, 10.0)
power = st.sidebar.slider("Power", 0.0, 100.0, 20.0)

input_data = np.array([[
    temperature,vibration,rpm,torque,
    tool_wear,pressure,power
]])

prob = model.predict_proba(input_data)[0][1]
pred = model.predict(input_data)[0]

health_score = max(0, int(100-tool_wear))
rul_days = max(1, int((100-tool_wear)*3))

# KPIs
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Accuracy", f"{acc*100:.1f}%")
c2.metric("Health Score", f"{health_score}%")
c3.metric("RUL", f"{rul_days} Days")
c4.metric("Failure Risk", f"{prob*100:.1f}%")
c5.metric("RPM", rpm)

st.subheader("🧠 Failure Prediction")
if pred == 1:
    st.error(f"⚠ Failure Predicted ({prob*100:.1f}%)")
else:
    st.success(f"✅ Machine Healthy ({(1-prob)*100:.1f}%)")

st.subheader("🏭 Digital Twin Status")
if health_score > 80:
    st.success("Machine Status: Healthy")
elif health_score > 50:
    st.warning("Machine Status: Warning")
else:
    st.error("Machine Status: Critical")

st.progress(health_score)

st.subheader("⚠ Anomaly Detection")
iso = IsolationForest(contamination=0.05, random_state=42)
labels = iso.fit_predict(X)
anomaly_count = int((labels == -1).sum())
st.metric("Detected Anomalies", anomaly_count)

st.subheader("🔮 Forecasting")
future_temp = temperature + np.random.uniform(1,5)
future_vib = vibration + np.random.uniform(0.1,0.5)
st.write(f"Predicted Temperature (24h): {future_temp:.2f}")
st.write(f"Predicted Vibration (24h): {future_vib:.2f}")

st.subheader("📈 Analytics")
col1,col2 = st.columns(2)

with col1:
    fig = px.line(df.head(500), y="Temperature", title="Temperature Trend")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = px.line(df.head(500), y="Vibration", title="Vibration Trend")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("📊 Failure Distribution")
fig3 = px.histogram(df, x="Failure")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("📄 Maintenance Report")
report = pd.DataFrame({
    "Health Score":[health_score],
    "Failure Risk":[prob*100],
    "RUL Days":[rul_days]
})
csv = report.to_csv(index=False)
st.download_button("Download Report", csv, "maintenance_report.csv")

with st.expander("Dataset Preview"):
    st.dataframe(df.head(50))
    

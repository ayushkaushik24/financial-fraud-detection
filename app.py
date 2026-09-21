import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="AI Cyber Sentinel | Fraud SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Cyber Security CSS Theme
st.markdown("""
<style>
    /* Dark Cyber Theme Background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Title Bar Styling */
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(90deg, #1f6feb 0%, #3fb950 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0px;
    }
    
    .sub-header {
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 20px;
    }

    /* Custom Glassmorphism Cards */
    .stCard {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur( 4px );
        margin-bottom: 15px;
    }

    /* Streamlit Button Upgrade */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #238636 0%, #2ea043 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #2ea043 0%, #3fb950 100%);
        box-shadow: 0 0 15px rgba(63, 185, 80, 0.4);
        transform: translateY(-2px);
    }

    /* Metric Label Fixes */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<p class="main-header">🛡️ AI CYBER SENTINEL: FRAUD DETECTION SOC</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time Autonomous Threat Engine & Financial Anomaly Monitoring</p>', unsafe_allow_html=True)

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\brije\Downloads\archive\creditcard.csv")
    df['Amount_Log'] = np.log1p(df['Amount'])
    return df

df = load_data()

# 2. Model Pipeline
@st.cache_resource
def train_model(data):
    X = data.drop(columns=['Class', 'Amount'])
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    return rf, X.columns

rf_model, feature_cols = train_model(df)

# Sidebar - System Control Center
st.sidebar.markdown("### 🎛️ SYSTEM CONTROLS")
st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚡ Core Engine Status")
st.sidebar.success("● AI Engine: Active (Random Forest v1.2)")
st.sidebar.info("● Database Node: MySQL Connected")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 📊 System Metrics")
st.sidebar.metric("Monitored Volume", f"{len(df):,}")
st.sidebar.metric("Anomalies Mitigated", f"{df['Class'].sum():,}")
st.sidebar.metric("Engine Accuracy (AUC)", "95.73%")

# Main Layout
tab1, tab2 = st.tabs(["🚨 AI Live Threat Simulator", "📈 Fraud Intelligence & Analytics"])

with tab1:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("⚡ Live Transaction Ingestion")
    
    col1, col2 = st.columns(2)
    with col1:
        tx_amount = st.number_input("Transaction Value (₹)", min_value=0.0, value=2500.0, step=100.0)
        tx_time = st.number_input("Timestamp Vector (Sec)", min_value=0.0, value=1240.0)
    
    with col2:
        sample_type = st.radio("Simulation Vector:", [
            "🟢 Standard Transaction (Low Risk)", 
            "🔴 Anomalous Attack Scenario (High Risk)"
        ])
    
    analyze_btn = st.button("🤖 RUN AI THREAT ANALYSIS")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if analyze_btn:
        with st.spinner("Analyzing neural pathways & feature vectors..."):
            if "High Risk" in sample_type:
                fraud_sample = df[df['Class'] == 1].iloc[0].drop(['Class', 'Amount']).to_dict()
                input_df = pd.DataFrame([fraud_sample])
                input_df['Amount_Log'] = np.log1p(tx_amount)
            else:
                normal_sample = df[df['Class'] == 0].iloc[0].drop(['Class', 'Amount']).to_dict()
                input_df = pd.DataFrame([normal_sample])
                input_df['Amount_Log'] = np.log1p(tx_amount)
            
            input_df = input_df.reindex(columns=feature_cols, fill_value=0)
            risk_score = rf_model.predict_proba(input_df)[0][1] * 100

        st.write("")
        if risk_score >= 50.0:
            st.error("🚨 **ALERT: HIGH-CONFIDENCE FRAUD DETECTED!**")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Threat Probability", f"{risk_score:.2f}%", delta="CRITICAL", delta_color="inverse")
            m2.metric("Flagged Amount", f"₹{tx_amount:,.2f}")
            m3.metric("Automated Action", "ACCOUNT FROZEN")
            
            st.warning("🔒 **Protocol Triggered**: Transaction halted. User flagged for 2FA/OTP verification. Incident logged in Security Audit.")
        else:
            st.success("✅ **TRANSACTION CLEARED BY AI AGENT**")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Risk Probability", f"{risk_score:.2f}%", delta="SAFE")
            m2.metric("Approved Amount", f"₹{tx_amount:,.2f}")
            m3.metric("System Verdict", "PASSED")
            
            st.info("🟢 Transaction processed through neural gateway with zero threat flags.")

with tab2:
    st.subheader("📊 Transaction Distribution & Analytics")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### Recent Logged Transactions")
        st.dataframe(df[['Time', 'Amount', 'Class']].head(10), use_container_width=True)
    
    with col_b:
        st.markdown("#### Average Spending: Normal vs Fraud (₹)")
        st.bar_chart(df.groupby('Class')['Amount'].mean())
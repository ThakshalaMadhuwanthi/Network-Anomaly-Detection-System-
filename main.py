import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import time
import json
from pathlib import Path
import requests

# =========================
# LOAD MODELS
# =========================
rf_model = joblib.load("random_forest.pkl")
xgb_model = joblib.load("xgboost.pkl")

scaler = joblib.load("scaler.pkl")
# Model uses only 13 numeric features (no one-hot encoding)

# =========================
# IP reputation (AbuseIPDB) with simple file cache
# =========================
API_KEY = os.environ.get("ABUSEIPDB_KEY")
IP_CACHE_PATH = Path("ip_rep_cache.json")
IP_CACHE_TTL = 3600  # seconds

try:
    IP_CACHE = json.loads(IP_CACHE_PATH.read_text()) if IP_CACHE_PATH.exists() else {}
except Exception:
    IP_CACHE = {}

def save_ip_cache():
    try:
        IP_CACHE_PATH.write_text(json.dumps(IP_CACHE))
    except Exception:
        pass

def get_ip_reputation(ip, min_age_seconds=IP_CACHE_TTL):
    if not ip:
        return None
    rec = IP_CACHE.get(ip)
    if rec and (time.time() - rec.get("ts", 0) < min_age_seconds):
        return rec.get("score")
    if not API_KEY:
        return None
    url = "https://api.abuseipdb.com/api/v2/check"
    headers = {"Key": API_KEY, "Accept": "application/json"}
    params = {"ipAddress": ip, "maxAgeInDays": 90}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=6)
        r.raise_for_status()
        data = r.json().get("data", {})
        score = data.get("abuseConfidenceScore")
    except Exception:
        score = None
    IP_CACHE[ip] = {"score": score, "ts": time.time()}
    save_ip_cache()
    return score

# =========================
# SELECTED FEATURES (UI ONLY)
# =========================
selected_features = [
    'same_srv_rate',
    'dst_host_srv_count',
    'logged_in',
    'dst_host_same_srv_rate',
    'serror_rate',
    'srv_serror_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'count',
    'dst_host_count',
    'dst_host_rerror_rate',
    'rerror_rate',
    'srv_rerror_rate'
]

# Feature descriptions and ranges for UI guidance
feature_info = {
    'same_srv_rate': {
        'description': '% connections to same service',
        'range': (0.0, 1.0),
        'example': 0.8,
        'help': 'Ratio of connections to the same service in a 2-sec window (0-1)'
    },
    'dst_host_srv_count': {
        'description': 'Connections from dst host to same service',
        'range': (0, 255),
        'example': 50,
        'help': 'Number of connections from destination host to same service'
    },
    'logged_in': {
        'description': 'User logged in (0=No, 1=Yes)',
        'range': (0, 1),
        'example': 1,
        'help': 'Binary: 0=not logged in, 1=logged in'
    },
    'dst_host_same_srv_rate': {
        'description': '% same service from dst host',
        'range': (0.0, 1.0),
        'example': 0.9,
        'help': 'Percentage of connections to same service from destination host (0-1)'
    },
    'serror_rate': {
        'description': '% SYN errors',
        'range': (0.0, 1.0),
        'example': 0.0,
        'help': 'Percentage of connections with SYN errors (0-1). High = suspicious'
    },
    'srv_serror_rate': {
        'description': '% SYN errors to same service',
        'range': (0.0, 1.0),
        'example': 0.0,
        'help': 'Percentage of SYN errors to the same service (0-1)'
    },
    'dst_host_serror_rate': {
        'description': '% SYN errors from dst host',
        'range': (0.0, 1.0),
        'example': 0.0,
        'help': 'Percentage of SYN errors from destination host (0-1)'
    },
    'dst_host_srv_serror_rate': {
        'description': '% SYN errors from dst to service',
        'range': (0.0, 1.0),
        'example': 0.0,
        'help': 'Percentage of SYN errors from dst host to same service (0-1)'
    },
    'count': {
        'description': 'Connections in 2-sec window',
        'range': (0, 511),
        'example': 5,
        'help': 'Number of connections in 2 second window (0-511)'
    },
    'dst_host_count': {
        'description': 'Total connections to dst host',
        'range': (0, 255),
        'example': 100,
        'help': 'Number of connections to destination host'
    },
    'dst_host_rerror_rate': {
        'description': '% rejected connections from dst',
        'range': (0.0, 1.0),
        'example': 0.0,
        'help': 'Percentage of rejected connections from destination host (0-1)'
    },
    'rerror_rate': {
        'description': '% REJ errors',
        'range': (0.0, 1.0),
        'example': 0.0,
        'help': 'Percentage of connections with REJ errors (0-1). High = suspicious'
    },
    'srv_rerror_rate': {
        'description': '% REJ errors to same service',
        'range': (0.0, 1.0),
        'example': 0.0,
        'help': 'Percentage of REJ errors to the same service (0-1)'
    }
}

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="IDS Dashboard", layout="wide")

st.title("🛡 Intrusion Detection System (RF + XGBoost)")

st.markdown("""
### What is this system?
This **Intrusion Detection System (IDS)** analyzes network connections and identifies potential attacks using Machine Learning models.

- 🎯 **Purpose:** Detect anomalous network behavior (attacks vs. normal traffic)
- 🤖 **Models:** Random Forest & XGBoost classifiers
- 📊 **Input:** 13 network connection features
- ✅ **Output:** NORMAL or ATTACK prediction

### How to use:
1. **Select a Model** (Random Forest or XGBoost) from the sidebar
2. **Choose Input Method:**
   - **Manual Input:** Enter feature values one-by-one using interactive controls
   - **CSV Upload:** Batch predict multiple connections from a CSV file
3. **Get Predictions:** System will detect if traffic is normal or anomalous
""")

# =========================
# MODEL SELECTION
# =========================
model_name = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "XGBoost"]
)

model = rf_model if model_name == "Random Forest" else xgb_model

st.sidebar.success(f"Using: {model_name}")



# =========================
# INPUT MODE
# =========================
mode = st.selectbox("Input Type", ["Manual Input", "CSV Upload"])

# =========================
# MANUAL INPUT
# =========================
if mode == "Manual Input":

    st.subheader("📊 Enter Network Connection Features")
    
    st.markdown("""
    **Instructions:** Enter values for the 13 network features below. 
    - Use **sliders** for percentage values (0.0 - 1.0)
    - Use **number inputs** for counts
    - **Hover over feature names** for detailed descriptions
    """)

    values = {}
    
    # Organize inputs in 3 columns for better layout
    col1, col2, col3 = st.columns(3)
    
    for idx, feature in enumerate(selected_features):
        info = feature_info[feature]
        col = [col1, col2, col3][idx % 3]
        
        with col:
            # Friendly description shown first
            st.markdown(f"**{info['description']}**")
            # Show help text and the actual column name so users can map UI → dataset
            st.caption(f"{info['help']} — Column: {feature}")
            
            min_val, max_val = info['range']
            
            # Use slider for percentage features (0-1), number input for counts
            if max_val <= 1.0:
                values[feature] = st.slider(
                    label=f"{info['description']} ({feature})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(info['example']),
                    step=0.01,
                    label_visibility="visible"
                )
            else:
                values[feature] = st.number_input(
                    label=f"{info['description']} ({feature})",
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(info['example']),
                    label_visibility="visible"
                )

    # Optional source IP for enrichment / reputation checks
    src_ip = st.text_input("Source IP (optional)")

    # Prediction button with summary
    st.markdown("---")
    col_left, col_right = st.columns([3, 1])
    
    with col_right:
        predict_btn = st.button("🔍 Predict", use_container_width=True)
    
    with col_left:
        st.info("✅ Enter values above and click Predict to detect anomalies")

    if predict_btn:

        # Create dataframe with 13 selected features
        input_df = pd.DataFrame([values])

        # Scale using training scaler
        X = scaler.transform(input_df)

        # Prediction
        pred = model.predict(X)[0]
        # Reputation check (optional)
        score = None
        try:
            if src_ip:
                score = get_ip_reputation(src_ip)
        except Exception:
            score = None
        
        # Display result with confidence
        st.markdown("---")
        
        # NOTE: label encoding used during training maps 1 -> normal, 0 -> anomaly
        if pred == 1:
            st.success("✅ **NORMAL TRAFFIC DETECTED**")
            st.markdown("""
            This network connection appears to be **legitimate and safe**.
            - No anomalous behavior patterns detected
            - Connection follows normal baseline patterns
            """)
        else:
            st.error("🚨 **POTENTIAL ATTACK DETECTED**")
            st.markdown("""
            This network connection shows **anomalous behavior**.
            - Suspicious patterns identified
            - Recommended: Further investigation required
            - Consider blocking this connection
            """)
        # Display IP reputation if available
        if score is not None:
            st.caption(f"IP reputation score: {score} / 100")
            if score >= 50:
                st.warning("⚠️ High abuse score — increase alert priority")
        elif src_ip:
            st.caption("IP reputation unavailable. Check that ABUSEIPDB_KEY is set in the shell where Streamlit is running.")

# =========================
# CSV UPLOAD
# =========================
elif mode == "CSV Upload":

    st.subheader("📁 Upload CSV File for Batch Prediction")
    
    st.markdown("""
    **Requirements:**
    - CSV must contain the following 13 columns:
    """)
    
    cols_display = st.columns(4)
    for idx, feat in enumerate(selected_features):
        cols_display[idx % 4].markdown(f"• `{feat}`")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        try:
            df = pd.read_csv(file)

            st.write("**File Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Check for missing required columns
            missing_cols = [col for col in selected_features if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
            else:
                st.success(f"✅ All {len(selected_features)} required columns found")

                # If CSV provides src_ip, pre-check reputation for unique IPs (cached)
                if 'src_ip' in df.columns:
                    unique_ips = df['src_ip'].dropna().unique().tolist()
                    for ip in unique_ips:
                        try:
                            get_ip_reputation(ip)
                        except Exception:
                            pass
                    st.info(f"Checked reputation for {len(unique_ips)} unique IPs (cache may be used)")

                if st.button("🚀 Run Prediction on Batch"):

                    # Extract only the 13 selected features
                    input_df = df[selected_features]

                    # Scale using training scaler
                    X = scaler.transform(input_df)

                    # Predict
                    preds = model.predict(X)

                    # Add predictions to dataframe
                    df["Prediction"] = preds
                    # training LabelEncoder mapped 1 -> normal, 0 -> anomaly
                    df["Prediction"] = df["Prediction"].apply(
                        lambda x: "🟢 NORMAL" if x == 1 else "🔴 ATTACK"
                    )

                    # attach IP abuse score if src_ip present
                    if 'src_ip' in df.columns:
                        df['ip_abuse_score'] = df['src_ip'].map(lambda ip: IP_CACHE.get(ip, {}).get('score'))
                    else:
                        df['ip_abuse_score'] = None

                    # final alert severity: HIGH if model says ATTACK or ip score >= 50
                    df['final_alert'] = df.apply(
                        lambda r: 'HIGH' if (r['Prediction'] == '🔴 ATTACK' or (r['ip_abuse_score'] or 0) >= 50) else 'LOW',
                        axis=1
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Prediction Results")
                    
                    # Summary statistics
                    # invert counts to match encoding (1 = normal)
                    normal_count = (preds == 1).sum()
                    attack_count = (preds == 0).sum()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(df))
                    with col2:
                        st.metric("🟢 Normal", normal_count)
                    with col3:
                        st.metric("🔴 Attacks", attack_count)
                    
                    st.dataframe(df, use_container_width=True)

                    st.success("✅ Prediction Completed Successfully")
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
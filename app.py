import streamlit as st
import requests
import numpy as np
import os

import streamlit as st
import requests

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000/predict"
)

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")

st.subheader("📊 PCA Features (V1 – V28)")

v_features = []
for i in range(1, 29):
    v = st.number_input(
        f"V{i}",
        value=0.0,
        format="%.6f"
    )
    v_features.append(v)

st.subheader("💰 Transaction Amount")
amount = st.number_input(
    "Amount",
    value=50.0,
    format="%.2f"
)

# ✅ FINAL FEATURE LIST
features = v_features + [amount]

st.write("🔢 Feature count:", len(features))

st.markdown("---")

if st.button("🚨 Predict Fraud"):

    if len(features) != 29:
        st.error("❌ Feature count must be exactly 29")
    else:
        with st.spinner("🔄 Sending data to backend..."):

            payload = {
                "features": [float(x) for x in features]
            }


            st.write("📦 Payload sent:", payload)

            try:
                response = requests.post(
                    BACKEND_URL,
                    json=payload,
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()

                    prob = result["fraud_probability"]
                    fraud = result["fraud"]

                    st.subheader("📊 Prediction Result")
                    st.metric("Fraud Probability", f"{prob:.4f}")

                    if fraud:
                        st.error("🚨 FRAUD DETECTED")
                    else:
                        st.success("✅ Legitimate Transaction")

                else:
                    st.error(f"Backend error: {response.text}")

            except Exception as e:
                st.error(f"❌ Request failed: {e}")


st.markdown("---")
st.caption("Powered by XGBoost + FastAPI + Streamlit")

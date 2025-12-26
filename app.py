"""
TELCO CUSTOMER CHURN PREDICTION APP
Streamlit App – Compatible with Pipeline Model (.pkl)
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📱",
    layout="wide"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model.pkl")
        model_info = joblib.load("model_info.pkl")
        return model, model_info
    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        return None, None

model, model_info = load_model()

# ================= HEADER =================
st.title("📱 Telco Customer Churn Prediction")
st.subheader("Prediksi churn pelanggan telekomunikasi menggunakan Machine Learning")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📌 Navigasi")
    page = st.radio(
        "Pilih Halaman",
        ["🏠 Home", "🔮 Prediction", "📊 Model Info", "ℹ️ About"]
    )

    st.markdown("---")
    st.info("""
    **Nama:** Daffa Setya Ramadhan  
    **NIM:** A11.2022.14042  
    **Mata Kuliah:** Bengkel Koding Data Science  
    """)

# ================= HOME =================
if page == "🏠 Home":
    st.markdown("""
    ### 📘 Tentang Aplikasi
    Aplikasi ini memprediksi **churn pelanggan Telco** menggunakan model Machine Learning
    yang telah dilatih dengan preprocessing otomatis (Pipeline).

    **Dataset:** Telco Customer Churn – Kaggle  
    **Jumlah data:** 7.043 pelanggan
    """)

    if model_info:
        st.markdown("### 📊 Performa Model")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{model_info['performance']['accuracy']:.2%}")
        col2.metric("Precision", f"{model_info['performance']['precision']:.2%}")
        col3.metric("Recall", f"{model_info['performance']['recall']:.2%}")
        col4.metric("F1 Score", f"{model_info['performance']['f1_score']:.2%}")

# ================= PREDICTION =================
elif page == "🔮 Prediction":

    if model is None:
        st.error("Model belum berhasil dimuat")
    else:
        st.subheader("📝 Input Data Pelanggan")

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
                Partner = st.selectbox("Partner", ["Yes", "No"])
                Dependents = st.selectbox("Dependents", ["Yes", "No"])

            with col2:
                PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
                MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

            with col3:
                OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

            with col2:
                PaymentMethod = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)"
                    ]
                )

            with col3:
                tenure = st.slider("Tenure (bulan)", 0, 72, 12)
                MonthlyCharges = st.number_input("Monthly Charges", 0.0, 150.0, 70.0)
                TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, tenure * MonthlyCharges)

            submit = st.form_submit_button("🔮 Predict")

        # ================= PREDICTION RESULT =================
        if submit:
            input_df = pd.DataFrame([{
                "gender": gender,
                "SeniorCitizen": SeniorCitizen,
                "Partner": Partner,
                "Dependents": Dependents,
                "tenure": tenure,
                "PhoneService": PhoneService,
                "MultipleLines": MultipleLines,
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection,
                "TechSupport": TechSupport,
                "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies,
                "Contract": Contract,
                "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod,
                "MonthlyCharges": MonthlyCharges,
                "TotalCharges": TotalCharges
            }])

            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]

            st.markdown("---")
            st.subheader("📊 Hasil Prediksi")

            if prediction == 1:
                st.error(f"⚠️ **CHURN** (Probabilitas: {proba:.2%})")
            else:
                st.success(f"✅ **TIDAK CHURN** (Probabilitas: {1-proba:.2%})")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Probabilitas Churn (%)"},
                gauge={"axis": {"range": [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

# ================= MODEL INFO =================
elif page == "📊 Model Info":
    if model_info:
        st.json(model_info)
    else:
        st.warning("Model info tidak tersedia")

# ================= ABOUT =================
elif page == "ℹ️ About":
    st.markdown("""
    ### 📚 UAS Bengkel Koding Data Science

    Aplikasi ini dibuat sebagai bagian dari **UAS Bengkel Koding Data Science**
    untuk memprediksi churn pelanggan telekomunikasi menggunakan Machine Learning.

    **Teknologi:**
    - Python
    - Scikit-Learn
    - Streamlit
    - Plotly
    """)

st.markdown("---")
st.caption("© 2025 – Telco Churn Prediction | Daffa Setya Ramadhan")

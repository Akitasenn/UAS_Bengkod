"""
TELCO CUSTOMER CHURN PREDICTION APP
Aplikasi Streamlit untuk prediksi churn pelanggan telekomunikasi
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOAD MODEL & PREPROCESSING OBJECTS ====================
@st.cache_resource
def load_models():
    """Load model dan preprocessing objects"""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        categorical_cols = joblib.load('categorical_cols.pkl')
        model_info = joblib.load('model_info.pkl')
        return model, scaler, feature_names, categorical_cols, model_info
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None

# Load models
model, scaler, feature_names, categorical_cols, model_info = load_models()

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .churn-yes {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .churn-no {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown('<div class="main-header">📱 Telco Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Prediksi Churn Pelanggan Telekomunikasi Menggunakan Machine Learning</div>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2991/2991148.png", width=100)
    st.title("📋 Navigation")
    
    page = st.radio(
        "Pilih Halaman:",
        ["🏠 Home", "🔮 Prediction", "📊 Model Info", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer Info")
    st.info("""
    **Nama:** Daffa Setya Ramadhan 
    **NIM:** A11.2022.14042
    **Mata Kuliah:** Bengkel Koding Data Science  
    **Tahun:** 2025/2026
    """)
    
    if model_info:
        st.markdown("---")
        st.markdown("### Model Performance")
        st.metric("Accuracy", f"{model_info['performance']['accuracy']:.2%}")
        st.metric("F1-Score", f"{model_info['performance']['f1_score']:.2%}")

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.header("🏠 Welcome to Telco Churn Prediction App")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📖 Tentang Aplikasi
        
        Aplikasi ini menggunakan **Machine Learning** untuk memprediksi apakah seorang pelanggan 
        telekomunikasi akan melakukan churn (berhenti berlangganan) atau tidak.
        
        **Fitur Utama:**
        - 🔮 Prediksi churn real-time
        - 📊 Visualisasi probability
        - 📈 Analisis feature importance
        - 🎯 Model performa metrics
        
        **Model yang Digunakan:**
        """)
        if model_info:
            st.success(f"✅ **{model_info['model_name']}**")
            st.write(f"F1-Score: **{model_info['performance']['f1_score']:.2%}**")
    
    with col2:
        st.markdown("""
        ### 🎯 Cara Menggunakan
        
        1. **Navigasi ke halaman Prediction** melalui sidebar
        2. **Isi form** dengan data pelanggan
        3. **Klik tombol Predict** untuk mendapatkan hasil
        4. **Lihat hasil prediksi** dan probability score
        
        ### 📊 Dataset Info
        
        Dataset yang digunakan adalah **Telco Customer Churn** dari Kaggle yang berisi:
        - 7,043 pelanggan
        - 20 fitur prediktor
        - 1 target variable (Churn: Yes/No)
        """)
    
    # Performance Metrics
    if model_info:
        st.markdown("---")
        st.subheader("📊 Model Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Accuracy", f"{model_info['performance']['accuracy']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Precision", f"{model_info['performance']['precision']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Recall", f"{model_info['performance']['recall']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("F1-Score", f"{model_info['performance']['f1_score']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)

# ==================== PREDICTION PAGE ====================
elif page == "🔮 Prediction":
    st.header("🔮 Customer Churn Prediction")
    st.markdown("Masukkan informasi pelanggan untuk memprediksi probabilitas churn")
    
    if model is None:
        st.error("❌ Model tidak dapat dimuat. Pastikan semua file model tersedia.")
    else:
        # Form Input
        with st.form("prediction_form"):
            st.subheader("📝 Data Pelanggan")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Informasi Demografis")
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Partner", ["No", "Yes"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
            
            with col2:
                st.markdown("#### Informasi Layanan")
                phone_service = st.selectbox("Phone Service", ["No", "Yes"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            
            with col3:
                st.markdown("#### Layanan Tambahan")
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Informasi Kontrak")
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
                payment_method = st.selectbox("Payment Method", 
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            
            with col2:
                st.markdown("#### Informasi Finansial")
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0, 0.5)
            
            with col3:
                st.markdown("#### Kalkulasi Total")
                total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, tenure * monthly_charges, 0.5)
            
            # Submit Button
            st.markdown("---")
            submit_button = st.form_submit_button("🔮 Predict Churn")
        
        # Prediction
        if submit_button:
            # Prepare input data
            input_dict = {
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # Preprocessing
            try:
                # One-hot encoding
                input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
                
                # Ensure all features are present
                for col in feature_names:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                
                # Reorder columns
                input_encoded = input_encoded[feature_names]
                
                # Scaling
                input_scaled = scaler.transform(input_encoded)
                
                # Prediction
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                # Display Results
                st.markdown("---")
                st.subheader("📊 Hasil Prediksi")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prediction Result
                    if prediction == 'Yes':
                        st.markdown(
                            f'<div class="prediction-box churn-yes">⚠️ CHURN RISK: HIGH<br>Pelanggan diprediksi akan CHURN</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box churn-no">✅ CHURN RISK: LOW<br>Pelanggan diprediksi TETAP BERLANGGANAN</div>',
                            unsafe_allow_html=True
                        )
                
                with col2:
                    # Probability Gauge
                    churn_prob = prediction_proba[1] if len(prediction_proba) > 1 else (1 if prediction == 'Yes' else 0)
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=churn_prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Churn Probability", 'font': {'size': 24}},
                        delta={'reference': 50, 'increasing': {'color': "red"}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkred" if churn_prob > 0.5 else "darkgreen"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 30], 'color': '#e8f5e9'},
                                {'range': [30, 70], 'color': '#fff9c4'},
                                {'range': [70, 100], 'color': '#ffebee'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Probabilities
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Probability NO CHURN", f"{prediction_proba[0]:.2%}")
                
                with col2:
                    st.metric("Probability CHURN", f"{prediction_proba[1]:.2%}" if len(prediction_proba) > 1 else "N/A")
                
                # Recommendations
                st.markdown("---")
                st.subheader("💡 Rekomendasi Aksi")
                
                if prediction == 'Yes':
                    st.warning("""
                    **⚠️ Pelanggan ini berisiko tinggi untuk churn!**
                    
                    Rekomendasi tindakan:
                    1. 🎁 Tawarkan diskon atau promo khusus
                    2. 📞 Hubungi pelanggan untuk feedback
                    3. 🎯 Berikan paket upgrade dengan harga spesial
                    4. 💳 Tawarkan program loyalitas
                    5. 📱 Tingkatkan kualitas layanan pelanggan
                    """)
                else:
                    st.success("""
                    **✅ Pelanggan ini cenderung tetap berlangganan!**
                    
                    Rekomendasi tindakan:
                    1. 🌟 Pertahankan kualitas layanan
                    2. 📧 Kirim newsletter dengan penawaran menarik
                    3. 🎁 Apresiasi loyalitas pelanggan
                    4. 📊 Monitor kepuasan pelanggan secara berkala
                    """)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.write("Debug Info:", input_encoded.shape if 'input_encoded' in locals() else "No data")

# ==================== MODEL INFO PAGE ====================
elif page == "📊 Model Info":
    st.header("📊 Model Information")
    
    if model_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Model Details")
            st.info(f"""
            **Model Type:** {model_info['model_name']}  
            **Training Date:** {datetime.now().strftime('%Y-%m-%d')}  
            **Total Features:** {len(model_info['feature_names'])}  
            **Categorical Features:** {len(model_info['categorical_cols'])}
            """)
            
            st.subheader("⚙️ Best Parameters")
            st.json(model_info['best_params'])
        
        with col2:
            st.subheader("📈 Performance Metrics")
            
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Score': [
                    model_info['performance']['accuracy'],
                    model_info['performance']['precision'],
                    model_info['performance']['recall'],
                    model_info['performance']['f1_score']
                ]
            })
            
            fig = px.bar(metrics_df, x='Metric', y='Score', 
                        title='Model Performance Metrics',
                        color='Score',
                        color_continuous_scale='Blues')
            fig.update_layout(showlegend=False, height=400)
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature List
        st.markdown("---")
        st.subheader("📋 Feature List")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Categorical Features:**")
            for col in model_info['categorical_cols']:
                st.write(f"- {col}")
        
        with col2:
            st.markdown("**Total Features after Encoding:**")
            st.write(f"Total: {len(model_info['feature_names'])} features")
            with st.expander("Show all features"):
                for idx, feat in enumerate(model_info['feature_names'], 1):
                    st.write(f"{idx}. {feat}")
    
    else:
        st.error("Model information not available")

# ==================== ABOUT PAGE ====================
elif page == "ℹ️ About":
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### 📚 UAS Bengkel Koding Data Science
    
    #### 🎯 Tujuan Project
    
    Project ini merupakan Capstone Project untuk Ujian Akhir Semester (UAS) mata kuliah 
    **Bengkel Koding Data Science** dengan tujuan:
    
    1. Melakukan Exploratory Data Analysis (EDA) yang komprehensif
    2. Membangun model machine learning untuk prediksi churn
    3. Melakukan preprocessing dan hyperparameter tuning
    4. Deploy model ke Streamlit Cloud
    
    #### 📊 Dataset
    
    **Telco Customer Churn Dataset** dari Kaggle berisi informasi tentang:
    - 7,043 pelanggan telekomunikasi
    - 20 fitur prediktor (demografis, layanan, billing)
    - 1 target variable: Churn (Yes/No)
    
    #### 🔧 Teknologi yang Digunakan
    
    - **Python** - Programming language
    - **Pandas & NumPy** - Data manipulation
    - **Scikit-learn** - Machine learning
    - **Streamlit** - Web app framework
    - **Plotly** - Interactive visualization
    - **GitHub** - Version control
    - **Streamlit Cloud** - Deployment platform
    
    #### 📈 Model Development
    
    Project ini mengimplementasikan 3 skenario modeling:
    
    1. **Direct Modeling** - Baseline tanpa preprocessing
    2. **Preprocessing** - Dengan data cleaning dan feature engineering
    3. **Hyperparameter Tuning** - Optimasi model untuk performa terbaik
    
    Model yang digunakan:
    - Logistic Regression (Konvensional)
    - Random Forest (Ensemble Bagging)
    - Voting Classifier (Ensemble Voting)
    
    #### 👨‍💻 Developer
    
    **Nama:** Daffa Setya Ramadhan  
    **NIM:** A11.2022.14042  
    **Program Studi:** Teknik INformatika  
    **Universitas:** Universitas Dian Nuswantoro
    
    #### 📝 License
    
    This project is created for educational purposes as part of UAS requirements.
    
    ---
    
    © 2025 - Bengkel Koding Data Science
    """)
    
    # Project Timeline
    st.markdown("---")
    st.subheader("📅 Project Timeline")
    
    timeline_data = {
        'Phase': ['EDA', 'Direct Modeling', 'Preprocessing', 'Tuning', 'Deployment'],
        'Duration': [1, 1, 2, 1.5, 0.5],
        'Status': ['✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete']
    }
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig = px.bar(timeline_df, x='Phase', y='Duration', 
                 title='Project Development Timeline (weeks)',
                 color='Duration',
                 text='Status')
    st.plotly_chart(fig, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Telco Customer Churn Prediction App</strong></p>
    <p>Bengkel Koding Data Science | Semester Ganjil 2025/2026</p>
    <p>© 2025 - Made with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
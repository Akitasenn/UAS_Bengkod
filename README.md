# 📱 Telco Customer Churn Prediction
## 📋 Deskripsi Project

Project ini merupakan **Capstone Project UAS Bengkel Koding Data Science** Semester Ganjil 2025/2026 yang bertujuan untuk memprediksi churn pelanggan telekomunikasi menggunakan machine learning.

**Churn** adalah kondisi ketika pelanggan berhenti berlangganan layanan. Dengan memprediksi pelanggan yang berpotensi churn, perusahaan dapat mengambil tindakan preventif untuk mempertahankan pelanggan.

## 🎯 Tujuan

1. Melakukan **Exploratory Data Analysis (EDA)** yang komprehensif
2. Membangun model prediksi churn menggunakan 3 kategori model:
   - Model Konvensional (Logistic Regression)
   - Ensemble Bagging (Random Forest)
   - Ensemble Voting (Voting Classifier)
3. Evaluasi model melalui 3 skenario:
   - Direct Modeling (tanpa preprocessing)
   - Modeling dengan Preprocessing
   - Hyperparameter Tuning
4. Deploy model terbaik ke **Streamlit Cloud**

## 📊 Dataset

**Sumber:** [Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Karakteristik:**
- 7,043 records
- 20 fitur prediktor
- 1 target variable (Churn: Yes/No)

**Fitur Dataset:**
- **Demografis:** Gender, SeniorCitizen, Partner, Dependents
- **Layanan:** PhoneService, InternetService, OnlineSecurity, dll.
- **Billing:** Contract, PaymentMethod, MonthlyCharges, TotalCharges
- **Target:** Churn (Yes/No)

## 🛠️ Teknologi

- **Python 3.8+**
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Machine learning
- **Matplotlib & Seaborn** - Data visualization
- **Plotly** - Interactive charts
- **Streamlit** - Web application
- **Joblib** - Model serialization

## 📁 Struktur Project

```
telco-churn-prediction/
│
├── notebooks/
│   └── Telco_Churn_Analysis.ipynb    
│
├── models/
│   ├── best_model.pkl                 # Model terbaik
│   ├── scaler.pkl                     # StandardScaler
│   ├── feature_names.pkl              # Nama fitur
│   ├── categorical_cols.pkl           # Kolom kategorikal
│   └── model_info.pkl                 # Informasi model
│
├── visualizations/
│   ├── missing_values.png
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrices/
│   └── comparison_all_scenarios.png
│
├── app.py                             # Aplikasi Streamlit
├── requirements.txt                   # Dependencies
├── README.md                          # Dokumentasi
└── .gitignore                         # Git ignore file
```

## 🚀 Instalasi & Penggunaan

### 1. Clone Repository

```bash
git clone https://github.com/[username]/telco-churn-prediction.git
cd telco-churn-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download dataset dari [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dan letakkan di root directory dengan nama `WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 4. Run Jupyter Notebook (Opsional)

```bash
jupyter notebook notebooks/Telco_Churn_Analysis.ipynb
```

### 5. Run Streamlit App

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📊 Model Performance

### Model Terbaik: [Nama Model]

| Metric | Score |
|--------|-------|
| Accuracy | XX.XX% |
| Precision | XX.XX% |
| Recall | XX.XX% |
| F1-Score | XX.XX% |

### Perbandingan Skenario

| Model | Direct | Preprocessing | Tuned |
|-------|--------|---------------|-------|
| Logistic Regression | X.XX | X.XX | X.XX |
| Random Forest | X.XX | X.XX | X.XX |
| Voting Classifier | X.XX | X.XX | X.XX |

## 🔍 Tahapan Project

### 1. Exploratory Data Analysis (EDA)
- ✅ Eksplorasi data awal (info, describe)
- ✅ Identifikasi missing values
- ✅ Visualisasi distribusi target
- ✅ Analisis korelasi fitur numerik

### 2. Direct Modeling
- ✅ Train-test split
- ✅ Training 3 model tanpa preprocessing
- ✅ Evaluasi performa (accuracy, precision, recall, F1-score)
- ✅ Confusion matrix visualization

### 3. Modeling dengan Preprocessing
- ✅ Handle missing values
- ✅ Remove duplikasi
- ✅ Handle outliers (IQR method)
- ✅ One-Hot Encoding
- ✅ Feature scaling (StandardScaler)
- ✅ Re-training & evaluasi

### 4. Hyperparameter Tuning
- ✅ GridSearchCV / RandomizedSearchCV
- ✅ Parameter optimization
- ✅ Best estimator selection
- ✅ Final evaluation

### 5. Deployment
- ✅ Save model & preprocessing objects
- ✅ Build Streamlit app
- ✅ Deploy to Streamlit Cloud
- ✅ Testing & validation

## 💻 Fitur Aplikasi

### 🏠 Home
- Overview aplikasi
- Model performance metrics
- Quick start guide

### 🔮 Prediction
- Form input data pelanggan
- Real-time prediction
- Probability visualization (gauge chart)
- Actionable recommendations

### 📊 Model Info
- Model details & parameters
- Performance metrics visualization
- Feature list & importance

### ℹ️ About
- Project information
- Technology stack
- Developer contact
- Project timeline

## 📈 Hasil & Insights

### Key Findings dari EDA:
- Distribusi kelas: XX% No Churn, XX% Churn
- Fitur dengan korelasi tinggi: [list fitur]
- Missing values: [deskripsi]

### Model Insights:
- Model terbaik: [nama model]
- Improvement dari baseline: +X.XX%
- Feature importance top 5: [list fitur]

### Business Recommendations:
1. [Rekomendasi 1]
2. [Rekomendasi 2]
3. [Rekomendasi 3]

## 🔗 Links

- 🌐 **Live Demo:** [https://your-app.streamlit.app](https://your-app.streamlit.app)
- 📊 **GitHub Repository:** [https://github.com/username/telco-churn-prediction](https://github.com/username/telco-churn-prediction)
- 📓 **Kaggle Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 👨‍💻 Developer

**Nama:** [Nama Anda]  
**NIM:** [NIM Anda]  
**Program Studi:** [Prodi Anda]  
**Email:** [email@example.com]  
**LinkedIn:** [linkedin.com/in/username](https://linkedin.com/in/username)

## 📝 License

This project is created for educational purposes as part of UAS requirements for Bengkel Koding Data Science course.

## 🙏 Acknowledgments

- Dataset dari [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Tim Dosen Bengkel Koding Data Science
- Asisten praktikum yang telah membimbing

## 📞 Contact & Support

Jika ada pertanyaan atau masalah, silakan:
- 📧 Email: [email@example.com]
- 💬 Create an issue di GitHub
- 📱 WhatsApp: [nomor]

---

⭐ **Jika project ini bermanfaat, jangan lupa berikan star di GitHub!**

© 2025 - Bengkel Koding Data Science | Semester Ganjil 2025/2026
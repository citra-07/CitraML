
import streamlit as st
import pandas as pd
import numpy as np
import pickle
# Tambahkan import ini agar pickle bisa memuat objek model dengan benar
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Load assets menggunakan pickle
@st.cache_resource
def load_assets():
    try:
        with open('placement_models.pkl', 'rb') as f:
            models = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return models, scaler
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

models, scaler = load_assets()

if models and scaler:
    st.set_page_config(page_title="Placement Predictor", layout="centered")
    st.title("🎓 Student Placement Prediction")
    st.markdown("Input detail mahasiswa untuk prediksi status penempatan.")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        ssc_p = st.number_input("SSC Percentage", 0.0, 100.0, 67.0)
        ssc_b = st.selectbox("SSC Board", ["Central", "Others"])
        hsc_p = st.number_input("HSC Percentage", 0.0, 100.0, 66.0)
        hsc_b = st.selectbox("HSC Board", ["Central", "Others"])
        hsc_s = st.selectbox("HSC Subject", ["Commerce", "Science", "Arts"])

    with col2:
        degree_p = st.number_input("Degree Percentage", 0.0, 100.0, 66.0)
        degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])
        workex = st.selectbox("Work Experience", ["Yes", "No"])
        etest_p = st.number_input("E-Test Percentage", 0.0, 100.0, 72.0)
        specialisation = st.selectbox("MBA Specialisation", ["Mkt&HR", "Mkt&Fin"])
        mba_p = st.number_input("MBA Percentage", 0.0, 100.0, 62.0)

    st.markdown("---")
    selected_algo = st.selectbox("🚀 Pilih Algoritma", list(models.keys()))

    if st.button("Prediksi Penempatan"):
        data = {
            "gender": 1 if gender == "M" else 0,
            "ssc_p": ssc_p,
            "ssc_b": 0 if ssc_b == "Central" else 1,
            "hsc_p": hsc_p,
            "hsc_b": 0 if hsc_b == "Central" else 1,
            "hsc_s": {"Arts": 0, "Commerce": 1, "Science": 2}[hsc_s],
            "degree_p": degree_p,
            "degree_t": {"Comm&Mgmt": 0, "Others": 1, "Sci&Tech": 2}[degree_t],
            "workex": 1 if workex == "Yes" else 0,
            "etest_p": etest_p,
            "specialisation": 0 if specialisation == "Mkt&Fin" else 1,
            "mba_p": mba_p
        }
        
        input_df = pd.DataFrame([data])
        # Pastikan urutan kolom sesuai dengan saat training
        input_scaled = scaler.transform(input_df)
        
        model = models[selected_algo]
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)[0]

        if prediction[0] == 1:
            st.success(f"Hasil: **Placed** (Confidence: {prob[1]*100:.2f}%)")
            st.balloons()
        else:
            st.error(f"Hasil: **Not Placed** (Confidence: {prob[0]*100:.2f}%)")

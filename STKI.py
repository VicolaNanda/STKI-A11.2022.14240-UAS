import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# Fungsi untuk melatih model
def train_model(data):
    # Memisahkan fitur dan target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Normalisasi fitur
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Melatih model Random Forest
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_scaled, y)
    
    return rf_classifier, scaler

# Memuat dataset
data = pd.read_csv('heart.csv')

# Melatih model
rf_classifier, scaler = train_model(data)

# Judul aplikasi
st.title("Prediksi Penyakit Jantung")
st.write(
    """
    Aplikasi ini memprediksi kemungkinan penyakit jantung berdasarkan input fitur.
    Silakan isi form di bawah ini:
    """
)

# Form untuk input fitur
age = st.number_input("Umur (age):", min_value=1, max_value=120, value=50)
sex = st.selectbox("Jenis Kelamin (sex):", options=[0, 1], format_func=lambda x: "Pria" if x == 1 else "Wanita")
cp = st.selectbox("Tipe Nyeri Dada (cp):", options=[0, 1, 2, 3], format_func=lambda x: f"Tipe {x}")
trestbps = st.number_input("Tekanan Darah Istirahat (trestbps):", min_value=50, max_value=200, value=120)
chol = st.number_input("Kolesterol (chol):", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl (fbs):", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
restecg = st.selectbox("Hasil Elektrokardiografi (restecg):", options=[0, 1, 2], format_func=lambda x: f"Kategori {x}")
thalach = st.number_input("Denyut Jantung Maksimal (thalach):", min_value=50, max_value=250, value=150)
exang = st.selectbox("Angina Induksi Olahraga (exang):", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
oldpeak = st.number_input("Depresi ST (oldpeak):", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Kemiringan ST (slope):", options=[0, 1, 2], format_func=lambda x: f"Kategori {x}")
ca = st.selectbox("Jumlah Pembuluh Warna (ca):", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal):", options=[0, 1, 2, 3], format_func=lambda x: f"Kategori {x}")

# Tombol untuk prediksi
if st.button("Prediksi"):
    # Menyusun input pengguna
    input_data = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
        columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    )
    
    # Normalisasi input data
    input_scaled = scaler.transform(input_data)
    
    # Prediksi dengan model
    prediction = rf_classifier.predict(input_scaled)[0]
    prediction_proba = rf_classifier.predict_proba(input_scaled)[0]
    
    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    st.write("Kemungkinan Penyakit Jantung: ", "Positif" if prediction == 1 else "Negatif")
    st.write("Probabilitas:")
    st.write(f"- Negatif: {prediction_proba[0]:.2f}")
    st.write(f"- Positif: {prediction_proba[1]:.2f}")

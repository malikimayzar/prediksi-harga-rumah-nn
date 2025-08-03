import streamlit as st
import pandas as pd
from tensorflow import keras
import pickle
import numpy as np

# Mengubah nama file model menjadi 'model.h5' agar konsisten
# dan kompatibel dengan deployment
try:
    model = keras.models.load_model("model/model.h5")
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model or scaler. Make sure you have run src/train_model.py. Error: {e}")
    st.stop()

# Judul aplikasi
st.title('🏠 Prediksi Harga Rumah')
st.markdown("""
Aplikasi sederhana untuk memprediksi harga rumah menggunakan model *Neural Network*. 
Silakan masukkan fitur-fitur rumah di bawah ini untuk mendapatkan perkiraan harganya.
""")

# Input dari user
st.sidebar.header('Fitur Properti')
luas = st.sidebar.slider('Luas Bangunan (m²)', 30, 500, 100)
kamar = st.sidebar.slider('Jumlah Kamar Tidur', 1, 10, 3) # Memperbaiki nilai default
usia = st.sidebar.slider('Usia Rumah (tahun)', 0, 50, 10)

# Tombol prediksi
if st.sidebar.button('Prediksi Harga'):
    input_data = np.array([[luas, kamar, usia]])
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        price_prediction = prediction[0][0]
        
        st.header('Hasil Prediksi')
        st.success(f"Harga rumah yang diprediksi adalah: *Rp {price_prediction:,.2f} Juta*")
        st.balloons()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")

st.markdown("""
---
Catatan: Model ini adalah model sederhana yang dilatih dengan data buatan. Hasil prediksi mungkin tidak akurat untuk data riil.
""")

st.markdown("---")
st.markdown("""
Dibuat dengan 💓 oleh Pa mey IG: malikimayzar
""")
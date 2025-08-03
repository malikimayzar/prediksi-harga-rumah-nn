import streamlit as st
import pandas as pd
from tensorflow import keras
import pickle
import numpy as np

try:
    model = keras.models.load_model("model/model.keras")
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model or scaler. Make sure you have run src/train_model.py. Error: {e}")
    st.stop()

# application title
st.title('House Price Predictions')
st.markdown(""" Simple application 
            to predict house prices using the 
            **Neural Network** model.
             Please enter the house features 
            below to get an estimated price.""")

# user input
st.sidebar.header('Fitur Properti')
wide = st.sidebar.slider('Building Area (m2)', 30, 500, 100,)
room = st.sidebar.slider('Number of Bedrooms', 1, 10, 30)
age = st.sidebar.slider('Age of House (year)', 0, 50, 10)

# prediction button
if st.sidebar.button('Price Prediction'):
    input_data = np.array([[wide, room, age]])
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        price_prediction = prediction[0][0]
        st.header('Prediction Results')
        st.success(f"The predicted house price is: **Rp{price_prediction:,.2f} million")
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred while predicting the process: {e}")
st.markdown("""
---This model is a simple model 
trained on artificial data. 
Predictions may not be accurate 
on real data.*""")

st.markdown("---")
st.markdown("""
Dibuat dengan 💓 oleh Pa mey IG:malikimayzar""")

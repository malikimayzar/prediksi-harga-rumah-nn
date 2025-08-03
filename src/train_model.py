import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# === Setup path ===
DATA_PATH = 'data/home_data.csv'
MODEL_DIR = 'model'
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')

# === Load dataset ===
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"❌ ERROR: File '{DATA_PATH}' tidak ditemukan. Pastikan nama file sudah benar.")
    exit()

x = df[['building_area', 'numbers_of_rooms', 'age_of_the_house']]
y = df['house_price']

# === Feature normalization ===
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# === Save scaler ===
os.makedirs(MODEL_DIR, exist_ok=True)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler berhasil disimpan di {SCALER_PATH}")

# === Split data ===
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42
)

# === Build model ===
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(3,)),
    layers.Dense(1, activation='linear')
])

# === Compile model ===
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

# === Train model ===
print("\n🚀 Start Model Training.....")
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_data=(x_test, y_test),
    verbose=1
)
print("✅ Training Completed")

# === Save model ===
model.save(MODEL_PATH)
print(f"✅ Model berhasil disimpan di {MODEL_PATH}")

# === Evaluate model ===
loss, mae = model.evaluate(x_test, y_test, verbose=0)
print("\n📊 Model Evaluation on Test Data:")
print(f"Loss (MSE): {loss:.2f}")
print(f"MAE: {mae:.2f}")
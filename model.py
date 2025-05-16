
# Supabase setup
from supabase import create_client, Client
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Initialize Supabase client (replace with your details)
url = "https://lcmsxmciopzkdldekbmw.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxjbXN4bWNpb3B6a2RsZGVrYm13Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MzY5OTYxOCwiZXhwIjoyMDU5Mjc1NjE4fQ.zAA8DtBU2qR12mLE1ldMY0iypQYc1OnJ1ndBHVVlpDk"
supabase = create_client(url, key)

# Fetch recent data from Supabase
response = supabase.table("sensor_data").select("temperature", "humidity", "ph_level").execute()

# Transform data into a DataFrame
data = response.data
df = pd.DataFrame(data)
print(f"✅ Data fetched from Supabase. {len(df)} records found.")

# Check if we have enough data
if len(df) < 46:  # We need at least 46 records for our window size of 45 plus 1 prediction
    print(f"❌ Not enough data for prediction. Found only {len(df)} records, need at least 46.")
    import sys
    sys.exit(1)

# 1. Normalize data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[["temperature", "humidity", "ph_level"]])

# 2. Create sequences: use 45 days to predict the 25th day's temperature and humidity
def create_sequences(data, window_size=45):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])        # Past 45 days
        y.append(data[i+window_size][:2])      # Next day's [temp, humidity]
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

# 3. Split into training and test sets (80% train, 20% test)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Build the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, input_shape=(45, 3)),   # 45 time steps, 3 features
    Dense(32, activation='relu'),
    Dense(2)  # Output: [temperature, humidity]
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 5. Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    verbose=1
)

# 6. Predict on the test set
y_pred = model.predict(X_test)

# Inverse scale the temperature and humidity
temp_hum_scaler = MinMaxScaler()
temp_hum_scaler.fit(df[["temperature", "humidity"]])
y_test_inv = temp_hum_scaler.inverse_transform(y_test)
y_pred_inv = temp_hum_scaler.inverse_transform(y_pred)

# 7. Print evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"✅ MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

# 8. Predict the next day's temperature & humidity (using last 45 days)
last_45_days = df[["temperature", "humidity", "ph_level"]].tail(45)
scaled_input = scaler.transform(last_45_days)
input_seq = scaled_input.reshape(1, 45, 3)

pred_scaled = model.predict(input_seq)
pred_actual = temp_hum_scaler.inverse_transform(pred_scaled)

# Display result
print("📅 Prediction for Tomorrow:")
print(f"🌡️ Temperature: {pred_actual[0][0]:.2f} °C")
print(f"💧 Humidity:    {pred_actual[0][1]:.2f} %")

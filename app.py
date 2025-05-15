from flask import Flask, render_template, request, jsonify, redirect, url_for
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
from supabase import create_client, Client
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import json

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)

# Configure Flask to use custom JSON encoder
app.json_encoder = NumpyEncoder

# Supabase credentials (from model.py)
url = "https://lcmsxmciopzkdldekbmw.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxjbXN4bWNpb3B6a2RsZGVrYm13Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MzY5OTYxOCwiZXhwIjoyMDU5Mjc1NjE4fQ.zAA8DtBU2qR12mLE1ldMY0iypQYc1OnJ1ndBHVVlpDk"
supabase = create_client(url, key)

# Model file path
MODEL_PATH = "plant_growth_model.h5"

# Define crop water needs (mm/day) for different crops at different growth stages
# Format: {crop_type: {early_stage: value, mid_stage: value, late_stage: value}}
CROP_WATER_NEEDS = {
    "Tomato": {"early": 3.8, "mid": 5.2, "late": 4.0},
    "Cucumber": {"early": 3.5, "mid": 5.0, "late": 3.8},
    "Lettuce": {"early": 3.0, "mid": 4.2, "late": 3.5},
    "Spinach": {"early": 2.8, "mid": 3.8, "late": 3.0},
    "Pepper": {"early": 3.2, "mid": 4.8, "late": 3.6},
    "Rice": {"early": 4.5, "mid": 6.8, "late": 5.2},
    "Wheat": {"early": 3.2, "mid": 4.5, "late": 3.8},
    "Corn": {"early": 3.7, "mid": 5.5, "late": 4.2}
}

# Irrigation efficiency factors
IRRIGATION_EFFICIENCY = {
    "Drip": 0.90,
    "Sprinkler": 0.75,
    "Flood": 0.60,
    "Micro-sprinkler": 0.85,
    "Subsurface": 0.95
}

# Soil types and their water holding capacity (mm/m)
SOIL_TYPES = {
    "Sandy": {"capacity": 70, "infiltration_rate": 25},  # mm/h infiltration rate
    "Loamy": {"capacity": 140, "infiltration_rate": 15},
    "Clay": {"capacity": 200, "infiltration_rate": 5},
    "Silt": {"capacity": 180, "infiltration_rate": 10},
    "Peat": {"capacity": 250, "infiltration_rate": 20}
}

# Weather adjustment factors
WEATHER_FACTORS = {
    "Sunny": 1.1,
    "Partly Cloudy": 1.0,
    "Cloudy": 0.9,
    "Rainy": 0.7,
    "Windy": 1.2
}

@app.route('/')
def index():
    # Fetch the latest 10 records from Supabase with all columns according to schema
    # (timestamp, temperature, humidity, ph_level, soil, soil_status, co2_ppm, co2_status, light)
    response = supabase.table("sensor_data").select("*").order("timestamp", desc=True).limit(10).execute()
    latest_data = response.data

    # Get the very latest record for current conditions
    current_conditions = latest_data[0] if latest_data else None

    # Format timestamp for better display
    if latest_data:
        for record in latest_data:
            if record.get('timestamp'):
                # Truncate timestamp to just show date and time (not timezone)
                try:
                    record['timestamp_display'] = record['timestamp'].split('T')[0] + ' ' + record['timestamp'].split('T')[1].split('.')[0]
                except:
                    record['timestamp_display'] = record['timestamp']

    # Get unique crop types for dropdown
    crop_types = list(CROP_WATER_NEEDS.keys())

    # Get irrigation types for dropdown
    irrigation_types = list(IRRIGATION_EFFICIENCY.keys())

    return render_template('index.html',
                          sensor_data=latest_data,
                          current_conditions=current_conditions,
                          crop_types=crop_types,
                          irrigation_types=irrigation_types,
                          model_exists=os.path.exists(MODEL_PATH))

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # Fetch data from Supabase
        response = supabase.table("sensor_data").select("temperature", "humidity", "ph_level").execute()
        data = response.data
        df = pd.DataFrame(data)

        # Check if we have enough data
        if len(df) < 31:  # We need at least 31 records for our window size of 30 plus 1 prediction
            return jsonify({"status": "error", "message": f"Not enough data for training. Found only {len(df)} records, need at least 31."})

        # Normalize data using MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[["temperature", "humidity", "ph_level"]])

        # Create sequences: use 30 days to predict the next day
        def create_sequences(data, window_size=30):
            X, y = [], []
            for i in range(len(data) - window_size):
                X.append(data[i:i+window_size])  # Past 30 days
                y.append(data[i+window_size][:2])  # Next day's [temp, humidity]
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data)

        # Split into training and test sets
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Build the LSTM model
        model = Sequential([
            LSTM(64, input_shape=(30, 3)),  # 30 days window, 3 features
            Dense(32, activation='relu'),
            Dense(2)  # Output: [temperature, humidity]
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,  # Increased for better accuracy (from model.py)
            batch_size=64,  # Increased for faster training (from model.py)
            verbose=1
        )

        # Save the model
        model.save(MODEL_PATH)

        # Save the scaler
        import pickle
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # Get evaluation metrics
        y_pred = model.predict(X_test)

        # Inverse scale the temperature and humidity
        temp_hum_scaler = MinMaxScaler()
        temp_hum_scaler.fit(df[["temperature", "humidity"]])
        y_test_inv = temp_hum_scaler.inverse_transform(y_test)
        y_pred_inv = temp_hum_scaler.inverse_transform(y_pred)

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)

        # Extract training history for visualization
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_mae = history.history['mae']
        val_mae = history.history['val_mae']
        epochs = list(range(1, len(train_loss) + 1))

        # Save training history to a file for future reference
        history_data = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'epochs': epochs
        }

        with open('training_history.json', 'w') as f:
            json.dump(history_data, f)

        return jsonify({
            "status": "success",
            "message": "Model trained successfully",
            "metrics": {
                "mse": round(mse, 2),
                "mae": round(mae, 2),
                "r2": round(r2, 2)
            },
            "history": {
                "epochs": epochs,
                "train_loss": [float(x) for x in train_loss],
                "val_loss": [float(x) for x in val_loss],
                "train_mae": [float(x) for x in train_mae],
                "val_mae": [float(x) for x in val_mae]
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/schedule_daily_collection', methods=['POST'])
def schedule_daily_collection():
    try:
        # Get form data
        data = request.get_json()
        collection_time = data.get('collection_time', '08:00')

        # Store the schedule in Supabase
        schedule_record = {
            "created_at": datetime.now().isoformat(),
            "collection_time": collection_time,
            "active": True
        }

        # Insert or update the schedule
        response = supabase.table("sensor_collection_schedule").upsert(schedule_record).execute()

        return jsonify({
            "status": "success",
            "message": f"Daily sensor data collection scheduled for {collection_time}",
            "schedule": schedule_record
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/predict_growth', methods=['POST'])
def predict_growth():
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({"status": "error", "message": "Model not trained yet. Please train the model first."})

        # Load the model with custom_objects to handle any LSTM parameter issues
        try:
            # Define custom objects to handle 'mse' function and other potential issues
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError()
            }

            # Try to load the model directly first
            try:
                model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
            except ValueError as ve:
                if 'batch_shape' in str(ve):
                    # If we get a batch_shape error, try loading and recreating the model
                    print("Handling batch_shape error by recreating the model")
            
                    # Create a new model with the same architecture
                    new_model = Sequential([
                        LSTM(64, input_shape=(30, 3)),  # 30 days window, 3 features
                        Dense(32, activation='relu'),
                        Dense(2)  # Output: [temperature, humidity]
                    ])
            
                    # Load the original model to get weights
                    original_model = tf.keras.models.load_model(
                        MODEL_PATH,
                        compile=False,
                        custom_objects=custom_objects,
                        # Skip validating the model to avoid batch_shape error
                        options=tf.saved_model.LoadOptions(experimental_skip_checkpoint=True)
                    )
            
                    # Copy weights from original model to new model
                    for i, layer in enumerate(original_model.layers):
                        if i < len(new_model.layers):
                            new_model.layers[i].set_weights(layer.get_weights())
            
                    model = new_model
                else:
                    # If it's a different error, re-raise it
                    raise

            # Recompile the model to ensure compatibility
            model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])
        except Exception as model_error:
            return jsonify({"status": "error", "message": f"Error loading model: {str(model_error)}"})

        # Fetch the last 30 days of data for prediction - select only fields that exist in the table
        response = supabase.table("sensor_data").select("temperature", "humidity", "ph_level").order("timestamp", desc=True).limit(30).execute()
        data = response.data

        if len(data) < 30:
            return jsonify({"status": "error", "message": f"Not enough recent data. Need at least 30 days for prediction, but found only {len(data)}."})

        # Convert to DataFrame and reverse to get chronological order
        df = pd.DataFrame(data[::-1])

        # Load the scaler
        import pickle
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Scale the input data
        scaled_input = scaler.transform(df[["temperature", "humidity", "ph_level"]])
        input_seq = scaled_input.reshape(1, 30, 3)  # Using 30 days window for prediction

        # Predict the next day only
        next_day = model.predict(input_seq)

        # Explicitly convert NumPy arrays to Python lists
        next_day_list = next_day.tolist()

        # Get temperature and humidity prediction
        temp_hum = next_day_list[0]

        # Get last pH (assume constant for prediction)
        last_ph = float(input_seq[0, -1, 2])  # Convert to native Python float

        # Convert prediction to actual values
        temp_hum_scaler = MinMaxScaler()
        temp_hum_scaler.fit(df[["temperature", "humidity"]])
        prediction_actual = temp_hum_scaler.inverse_transform(np.array(next_day_list))

        # Get the predicted temperature and humidity and convert to native Python floats
        temp = float(prediction_actual[0, 0])
        humidity = float(prediction_actual[0, 1])

        # Simple growth rate calculation based on ideal conditions
        # Assume optimal temp is 25°C and optimal humidity is 60%
        temp_factor = float(1 - abs(temp - 25) / 25)  # Closer to 1 is better
        humidity_factor = float(1 - abs(humidity - 60) / 60)  # Closer to 1 is better

        # Calculate daily growth percentage
        daily_growth = float((temp_factor + humidity_factor) / 2 * 15)  # 15% max daily growth

        # Create tomorrow's date
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        # Get current conditions for comparison and convert to native Python floats
        current_temp = float(df["temperature"].iloc[-1])
        current_humidity = float(df["humidity"].iloc[-1])

        # Calculate growth change from today
        current_temp_factor = float(1 - abs(current_temp - 25) / 25)
        current_humidity_factor = float(1 - abs(current_humidity - 60) / 60)
        current_growth = float((current_temp_factor + current_humidity_factor) / 2 * 15)
        growth_change = float(daily_growth - current_growth)

        # Growth trend indicator
        if growth_change > 0.5:
            trend = "Improving"
            trend_icon = "bi-arrow-up-circle-fill text-success"
        elif growth_change < -0.5:
            trend = "Declining"
            trend_icon = "bi-arrow-down-circle-fill text-danger"
        else:
            trend = "Stable"
            trend_icon = "bi-dash-circle-fill text-warning"

        # Store prediction in global variable for water scheduling
        prediction_result = {
            "status": "success",
            "prediction": {
                "date": tomorrow,
                "temperature": round(temp, 1),
                "humidity": round(humidity, 1),
                "daily_growth": round(daily_growth, 1),
                "trend": trend,
                "trend_icon": trend_icon,
                "growth_change": round(growth_change, 1)
            },
            "current": {
                "temperature": round(current_temp, 1),
                "humidity": round(current_humidity, 1),
                "daily_growth": round(current_growth, 1)
            }
        }

        # Save prediction to a file for water scheduling to use
        with open('latest_prediction.json', 'w') as f:
            json.dump(prediction_result, f)

        return jsonify(prediction_result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/calculate_water', methods=['POST'])
def calculate_water():
    try:
        # Get form data
        data = request.get_json()
        irrigation_type = data.get('irrigation_type')
        area = float(data.get('area', 0))
        crop_type = data.get('crop_type')
        growth_stage = data.get('growth_stage', 'mid')
        soil_type = data.get('soil_type', 'Loamy')
        weather_condition = data.get('weather_condition', 'Partly Cloudy')

        # Get current conditions - select only fields that exist in the table
        response = supabase.table("sensor_data").select("temperature", "humidity", "ph_level").order("timestamp", desc=True).limit(1).execute()
        current_data = response.data[0] if response.data else None

        if not current_data:
            return jsonify({"status": "error", "message": "No current sensor data available"})

        # Check if we have temperature and humidity (required fields)
        if "temperature" not in current_data or "humidity" not in current_data:
            return jsonify({"status": "error", "message": "Required sensor data (temperature and humidity) not available"})

        # Get base water need for the crop and growth stage
        base_water_need = CROP_WATER_NEEDS.get(crop_type, {}).get(growth_stage, 4.0)  # mm/day, default 4.0

        # Extract current sensor data (use None for missing values)
        current_temp = float(current_data.get("temperature")) if "temperature" in current_data else None
        current_humidity = float(current_data.get("humidity")) if "humidity" in current_data else None
        current_ph = float(current_data.get("ph_level")) if "ph_level" in current_data else None
        # Check for additional sensor data that might not be in the database yet
        # If these columns don't exist, they'll be None but the code won't break
        current_co2 = float(current_data.get("co2_level")) if "co2_level" in current_data else None
        current_soil = float(current_data.get("soil_moisture")) if "soil_moisture" in current_data else None
        current_light = float(current_data.get("light_intensity")) if "light_intensity" in current_data else None

        # Get soil properties
        soil_properties = SOIL_TYPES.get(soil_type, SOIL_TYPES["Loamy"])

        # Weather adjustment
        weather_factor = WEATHER_FACTORS.get(weather_condition, 1.0)

        # Check if we have a recent growth prediction to use
        has_prediction = False
        next_day_temp = None
        next_day_humidity = None
        prediction_source = None
        growth_data = None

        # Try to load prediction from file first (most recent)
        try:
            if os.path.exists('latest_prediction.json'):
                # Check if the prediction file is recent (less than 1 hour old)
                file_age = time.time() - os.path.getmtime('latest_prediction.json')
                if file_age < 3600:  # 1 hour in seconds
                    with open('latest_prediction.json', 'r') as f:
                        growth_data = json.load(f)

                    if growth_data and growth_data["status"] == "success":
                        next_day_temp = float(growth_data["prediction"]["temperature"])
                        next_day_humidity = float(growth_data["prediction"]["humidity"])
                        prediction_source = "Growth Prediction"
                        has_prediction = True
        except Exception as e:
            print(f"Error loading prediction file: {str(e)}")

        # If no growth prediction available, try to predict using the model directly
        if not has_prediction and os.path.exists(MODEL_PATH) and os.path.exists('scaler.pkl'):
            try:
                # Load model and scaler with custom_objects to handle any LSTM parameter issues
                try:
                    # Define custom objects to handle 'mse' function and other potential issues
                    custom_objects = {
                        'mse': tf.keras.losses.MeanSquaredError(),
                        'mae': tf.keras.metrics.MeanAbsoluteError()
                    }

                    # Try to load the model directly first
                    try:
                        model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
                    except ValueError as ve:
                        if 'batch_shape' in str(ve):
                            # If we get a batch_shape error, try loading and recreating the model
                            print("Handling batch_shape error by recreating the model")

                            # Create a new model with the same architecture
                            new_model = Sequential([
                                LSTM(64, input_shape=(30, 3)),  # 30 days window, 3 features
                                Dense(32, activation='relu'),
                                Dense(2)  # Output: [temperature, humidity]
                            ])

                            # Load the original model to get weights
                            original_model = tf.keras.models.load_model(
                                MODEL_PATH,
                                compile=False,
                                custom_objects=custom_objects,
                                # Skip validating the model to avoid batch_shape error
                                options=tf.saved_model.LoadOptions(experimental_skip_checkpoint=True)
                            )

                            # Copy weights from original model to new model
                            for i, layer in enumerate(original_model.layers):
                                if i < len(new_model.layers):
                                    new_model.layers[i].set_weights(layer.get_weights())

                            model = new_model
                        else:
                            # If it's a different error, re-raise it
                            raise

                    # Recompile the model to ensure compatibility
                    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])
                except Exception as model_error:
                    print(f"Error loading model: {str(model_error)}")
                    raise

                import pickle
                with open('scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)

                # Get last 30 days of data for prediction
                hist_response = supabase.table("sensor_data").select("temperature", "humidity", "ph_level").order("timestamp", desc=True).limit(30).execute()
                if len(hist_response.data) == 30:
                    # Prepare data for prediction
                    hist_data = pd.DataFrame(hist_response.data[::-1])  # Reverse to get chronological order
                    scaled_input = scaler.transform(hist_data[["temperature", "humidity", "ph_level"]])
                    input_seq = scaled_input.reshape(1, 30, 3)  # Using 30 days window for prediction

                    # Predict next day
                    next_day = model.predict(input_seq, verbose=0)
                    next_day_list = next_day.tolist()  # Convert to Python list

                    # Convert prediction to actual values
                    temp_hum_scaler = MinMaxScaler()
                    temp_hum_scaler.fit(hist_data[["temperature", "humidity"]])
                    pred_actual = temp_hum_scaler.inverse_transform(np.array(next_day_list))

                    next_day_temp = float(pred_actual[0, 0])
                    next_day_humidity = float(pred_actual[0, 1])
                    prediction_source = "ML Model"
                    has_prediction = True
            except Exception as e:
                print(f"Prediction error: {str(e)}")

        # If no prediction available, use current values
        if not has_prediction:
            next_day_temp = current_temp
            next_day_humidity = current_humidity
            prediction_source = "Current Data"

        # Create water scheduling plan based on available data
        water_scheduling_plan = create_water_scheduling_plan(
            current_temp=current_temp,
            current_humidity=current_humidity,
            current_ph=current_ph,
            current_soil=current_soil,
            current_co2=current_co2,
            current_light=current_light,
            next_day_temp=next_day_temp,
            next_day_humidity=next_day_humidity,
            crop_type=crop_type,
            growth_stage=growth_stage,
            soil_type=soil_type,
            weather_condition=weather_condition,
            irrigation_type=irrigation_type,
            area=area,
            base_water_need=base_water_need
        )

        # Create tomorrow's date
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        # Growth information if available
        growth_info = None
        if growth_data and prediction_source == "Growth Prediction":
            growth_info = {
                "daily_growth": growth_data["prediction"]["daily_growth"],
                "trend": growth_data["prediction"]["trend"],
                "growth_change": growth_data["prediction"]["growth_change"]
            }

        response_data = {
            "status": "success",
            "prediction": {
                "date": tomorrow,
                "temperature": round(next_day_temp, 1) if next_day_temp is not None else None,
                "humidity": round(next_day_humidity, 1) if next_day_humidity is not None else None,
                "prediction_source": prediction_source
            },
            "irrigation": water_scheduling_plan,
            "soil_properties": {
                "type": soil_type
            },
            "crop_info": {
                "type": crop_type,
                "growth_stage": growth_stage,
                "base_water_need": base_water_need
            },
            "current_conditions": {
                "temperature": round(current_temp, 1) if current_temp is not None else None,
                "humidity": round(current_humidity, 1) if current_humidity is not None else None,
                "ph_level": round(current_ph, 1) if current_ph is not None else None,
                "co2_level": round(current_co2, 1) if current_co2 is not None else None,
                "soil_moisture": round(current_soil, 1) if current_soil is not None else None,
                "light_intensity": round(current_light, 1) if current_light is not None else None,
                "weather": weather_condition
            }
        }

        # Add growth info if available
        if growth_info:
            response_data["growth"] = growth_info

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


def create_water_scheduling_plan(current_temp, current_humidity, current_ph, current_soil, current_co2,
                                current_light, next_day_temp, next_day_humidity, crop_type, growth_stage,
                                soil_type, weather_condition, irrigation_type, area, base_water_need):
    """Create a water scheduling plan based on available sensor data."""
    # Initialize adjustment factors
    temp_factor = 1.0
    humidity_factor = 1.0
    ph_factor = 1.0
    soil_factor = 1.0
    co2_factor = 1.0
    light_factor = 1.0
    weather_factor = WEATHER_FACTORS.get(weather_condition, 1.0)

    # Use prediction data if available, otherwise use current data
    temp = next_day_temp if next_day_temp is not None else current_temp
    humidity = next_day_humidity if next_day_humidity is not None else current_humidity

    # Temperature adjustment (critical factor)
    if temp is not None:
        if temp > 30:  # Hot conditions
            temp_factor = 1.3  # 30% more water
        elif temp > 25:
            temp_factor = 1.15  # 15% more water
        elif temp < 15:
            temp_factor = 0.8  # 20% less water
        elif temp < 20:
            temp_factor = 0.9  # 10% less water

    # Humidity adjustment (critical factor)
    if humidity is not None:
        if humidity < 40:  # Dry conditions
            humidity_factor = 1.25  # 25% more water
        elif humidity < 60:
            humidity_factor = 1.1  # 10% more water
        elif humidity > 80:
            humidity_factor = 0.85  # 15% less water
        elif humidity > 70:
            humidity_factor = 0.95  # 5% less water

    # pH adjustment (if available)
    if current_ph is not None:
        # Optimal pH range depends on crop type, but generally 6.0-7.0 is good
        if current_ph < 5.5 or current_ph > 7.5:
            ph_factor = 1.1  # 10% more water for non-optimal pH
        elif current_ph < 6.0 or current_ph > 7.0:
            ph_factor = 1.05  # 5% more water for slightly non-optimal pH

    # Soil moisture adjustment (if available)
    if current_soil is not None:
        if current_soil < 30:  # Very dry soil
            soil_factor = 1.3  # 30% more water
        elif current_soil < 45:
            soil_factor = 1.15  # 15% more water
        elif current_soil > 75:
            soil_factor = 0.7  # 30% less water
        elif current_soil > 60:
            soil_factor = 0.85  # 15% less water

    # CO2 adjustment (if available)
    if current_co2 is not None:
        if current_co2 > 800:  # High CO2 can improve water efficiency
            co2_factor = 0.95  # 5% less water
        elif current_co2 < 350:  # Low CO2 might reduce efficiency
            co2_factor = 1.05  # 5% more water

    # Light intensity adjustment (if available)
    if current_light is not None:
        if current_light > 8000:  # Bright conditions
            light_factor = 1.1  # 10% more water
        elif current_light < 3000:  # Low light
            light_factor = 0.9  # 10% less water

    # Calculate water need with all available factors
    water_need = base_water_need * temp_factor * humidity_factor * ph_factor * soil_factor * co2_factor * light_factor * weather_factor

    # Convert to liters based on area
    water_volume = water_need * area

    # Adjust based on irrigation efficiency
    required_water = water_volume / IRRIGATION_EFFICIENCY.get(irrigation_type, 0.75)

    # Calculate irrigation duration (minutes)
    flow_rate = 4.0  # liters per minute
    duration = required_water / flow_rate

    # Determine optimal watering time based on temperature
    if temp is not None and temp > 28:
        optimal_time = "Early Morning (5-7 AM) or Evening (After 6 PM)"
    elif temp is not None and temp < 15:
        optimal_time = "Late Morning (10-11 AM)"
    else:
        optimal_time = "Morning (6-9 AM)"

    # Determine watering frequency and sessions based on soil type and moisture
    frequency = "Every 2 days"  # Default
    sessions = 1  # Default

    # Adjust based on soil type
    if soil_type == "Sandy":
        frequency = "Daily"
        sessions = 2  # Multiple shorter sessions for sandy soil
    elif soil_type == "Clay":
        frequency = "Every 3 days"

    # Further adjust based on soil moisture if available
    if current_soil is not None:
        if current_soil < 30:  # Very dry
            if frequency == "Every 3 days":
                frequency = "Every 2 days"
            elif frequency == "Every 2 days":
                frequency = "Daily"
        elif current_soil > 70:  # Very moist
            if frequency == "Daily":
                frequency = "Every 2 days"
            elif frequency == "Every 2 days":
                frequency = "Every 3 days"

    # Calculate session duration
    session_duration = duration / sessions

    # Create watering tips based on available data
    watering_tips = []

    if temp is not None and temp > 30:
        watering_tips.append("Due to high temperatures, consider adding mulch to reduce evaporation")

    if humidity is not None and humidity < 40:
        watering_tips.append("Low humidity increases water needs. Consider misting plants in addition to regular watering")

    if current_soil is not None and current_soil < 30:
        watering_tips.append("Soil is very dry. Consider a thorough initial watering before starting regular schedule")

    if current_ph is not None and (current_ph < 5.5 or current_ph > 7.5):
        watering_tips.append(f"pH level ({current_ph}) is outside optimal range. Consider soil amendments")

    # Return the complete water scheduling plan
    return {
        "water_required_liters": round(required_water, 1),
        "duration_minutes": round(duration, 1),
        "optimal_time": optimal_time,
        "frequency": frequency,
        "sessions": sessions,
        "session_duration": round(session_duration, 1),
        "adjustment_factors": {
            "temperature": round(temp_factor, 2),
            "humidity": round(humidity_factor, 2),
            "ph_level": round(ph_factor, 2),
            "soil_moisture": round(soil_factor, 2) if current_soil is not None else None,
            "co2_level": round(co2_factor, 2) if current_co2 is not None else None,
            "light_intensity": round(light_factor, 2) if current_light is not None else None,
            "weather": round(weather_factor, 2)
        },
        "watering_tips": watering_tips
    }

@app.route('/add_sensor_data', methods=['POST'])
def add_sensor_data():
    try:
        # Get form values
        temperature = float(request.form.get("temperature"))
        humidity = float(request.form.get("humidity"))
        ph_level = float(request.form.get("ph_level"))

        # Create record
        record = {
            "timestamp": datetime.now().isoformat(),
            "temperature": temperature,
            "humidity": humidity,
            "ph_level": ph_level
        }

        # Insert into Supabase
        response = supabase.table("sensor_data").insert(record).execute()
        return redirect(url_for('index'))

    except Exception as e:
        return f"<script>alert('Error: {str(e)}'); window.location.href='/'</script>"

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT dynamically
    app.run(host='0.0.0.0', port=port)


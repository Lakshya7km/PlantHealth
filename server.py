from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime
import random

app = Flask(__name__)
CORS(app)

# Sample plant metadata (normally from Supabase)
plant_info = {
    "plantId": "lettuce001",
    "plantingDate": "2025-03-25",
    "irrigationType": "drip",
    "plantingArea": 50
}

# Sensor data (normally from Supabase)
sensor_data = {
    "temperature": 29.0,
    "humidity": 55.0,
    "tdsValue": 850,
    "phLevel": 6.5
}

# Simulated LSTM model logic
def predict_growth_days(data):
    return 21  # placeholder: return fixed days for now

def calculate_growth_status(planting_date_str, predicted_total_days):
    today = datetime.date.today()
    planting_date = datetime.datetime.strptime(planting_date_str, "%Y-%m-%d").date()
    days_since = (today - planting_date).days
    remaining = max(predicted_total_days - days_since, 0)
    growth_percentage = min(100, round((days_since / predicted_total_days) * 100))

    if growth_percentage < 33:
        stage = "Seedling"
    elif growth_percentage < 66:
        stage = "Vegetative"
    else:
        stage = "Mature/Harvest"

    return {
        "daysSincePlanting": days_since,
        "remainingDays": remaining,
        "growthPercentage": growth_percentage,
        "growthStage": stage,
        "predictedTotalDays": predicted_total_days
    }

def calculate_water_duration(irrigation_type, area):
    rates = {
        "drip": 0.3,
        "sprinkler": 0.5,
        "borewell": 0.7
    }
    flow_rate = rates.get(irrigation_type, 0.5)
    return round(area * flow_rate)

def should_water(temp, humidity):
    return temp > 28 and humidity < 60

@app.route('/plant-status', methods=['GET'])
def get_plant_status():
    predicted_days = predict_growth_days(sensor_data)
    growth = calculate_growth_status(plant_info["plantingDate"], predicted_days)
    duration = calculate_water_duration(plant_info["irrigationType"], plant_info["plantingArea"])
    needs_water = should_water(sensor_data["temperature"], sensor_data["humidity"])

    return jsonify({
        "growthPrediction": growth,
        "wateringInfo": {
            "waterDuration": duration,
            "needsWatering": needs_water
        },
        "plantInfo": plant_info,
        "sensorData": sensor_data
    })

@app.route('/reset-cycle', methods=['POST'])
def reset_cycle():
    data = request.get_json()
    plant_info.update({
        "plantId": data.get("plantId"),
        "plantingDate": data.get("plantingDate"),
        "irrigationType": data.get("irrigationType"),
        "plantingArea": data.get("plantingArea")
    })
    return jsonify({"message": "Cycle reset successfully", "plantInfo": plant_info})

if __name__ == '__main__':
    app.run(debug=True)

from fastapi import FastAPI
from supabase import create_client, Client
from datetime import datetime
import os

app = FastAPI()

# Supabase connection
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

@app.get("/")
def home():
    return {"message": "✅ API working!"}

@app.get("/irrigation")
def calculate_irrigation():
    # 1. Fetch latest sensor data
    sensor_res = supabase.table("sensor_data").select("*").order("timestamp", desc=True).limit(1).execute()
    sensor = sensor_res.data[0]

    # 2. Fetch latest planting info
    planting_res = supabase.table("planting_info").select("*").order("planting_date", desc=True).limit(1).execute()
    planting = planting_res.data[0]

    # 3. Calculate growth stage
    planting_date = datetime.fromisoformat(planting["planting_date"])
    now = datetime.now()
    growth_days = (now - planting_date).days

    if growth_days <= 7:
        stage = "Germination"
    elif growth_days <= 15:
        stage = "Early Growth"
    elif growth_days <= 25:
        stage = "Mid Growth"
    else:
        stage = "Maturity"

    # 4. Calculate irrigation need
    temperature = sensor["temperature"]
    humidity = sensor["humidity"]
    irrigation_type = planting["irrigation_type"]
    area = planting["planting_area"]

    irrigation_needed = humidity < 65 or temperature > 28

    time_per_m2 = {
        "Drip": 0.3,
        "Sprinkler": 0.5,
        "Borewell": 0.6,
        "Tap": 0.4
    }

    stage_multiplier = {
        "Germination": 0.5,
        "Early Growth": 0.8,
        "Mid Growth": 1.0,
        "Maturity": 0.6
    }

    base_time = time_per_m2.get(irrigation_type, 0.5)
    adjusted_time = base_time * area * stage_multiplier.get(stage, 1.0)

    return {
        "irrigation_needed": irrigation_needed,
        "duration_minutes": round(adjusted_time, 2),
        "growth_stage": stage
    }

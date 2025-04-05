from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client, Client
from datetime import datetime
import uuid
import os

app = FastAPI()

# Supabase Setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------- Data Models ------------------------
class PlantingInput(BaseModel):
    irrigation_type: str
    planting_area: float

# ---------------------- Routes -----------------------------

@app.get("/")
def home():
    return {"message": "✅ API working!"}

# ✅ Combined Dashboard Info API
@app.get("/dashboard-info")
def dashboard_info():
    sensor_res = supabase.table("sensor_data").select("*").order("timestamp", desc=True).limit(1).execute()
    planting_res = supabase.table("planting_info").select("*").order("planting_date", desc=True).limit(1).execute()

    if not sensor_res.data or not planting_res.data:
        return {"error": "No data available."}

    sensor = sensor_res.data[0]
    planting = planting_res.data[0]

    # Step 1: Prediction part (replace with actual LSTM later)
    predicted_growth_days = 25  # Example fixed value for now

    planting_date = datetime.fromisoformat(planting["planting_date"])
    now = datetime.now()
    days_passed = (now - planting_date).days
    remaining_days = max(predicted_growth_days - days_passed, 0)

    # Step 2: Growth stage
    if days_passed <= 7:
        stage = "Germination"
    elif days_passed <= 15:
        stage = "Early Growth"
    elif days_passed <= 25:
        stage = "Mid Growth"
    else:
        stage = "Maturity"

    # Step 3: Irrigation logic
    temperature = sensor["temperature"]
    humidity = sensor["humidity"]
    irrigation_type = planting["irrigation_type"]
    area = float(planting["planting_area"])

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
    duration_minutes = round(base_time * area * stage_multiplier.get(stage, 1.0), 2)

    return {
        "predicted_growth_days": predicted_growth_days,
        "days_passed": days_passed,
        "remaining_days": remaining_days,
        "growth_stage": stage,
        "irrigation_needed": irrigation_needed,
        "duration_minutes": duration_minutes,
        "planting_info": planting,
        "sensor_data": sensor,
        "last_updated": now.isoformat()
    }

# ✅ Reset functionality - Save new planting info
@app.post("/reset")
def reset_planting_info(data: PlantingInput):
    new_entry = {
        "id": str(uuid.uuid4()),
        "plant_id": "plant_1",
        "planting_date": datetime.now().date().isoformat(),
        "irrigation_type": data.irrigation_type,
        "planting_area": data.planting_area
    }

    supabase.table("planting_info").insert(new_entry).execute()
    return {"message": "✅ New planting data inserted and reset successful!"}

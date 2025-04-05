from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client, Client
from datetime import datetime
import uuid
import os

app = FastAPI()

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Test route
@app.get("/")
def home():
    return {"message": "✅ API working!"}

# Irrigation calculation route
@app.get("/irrigation")
def calculate_irrigation():
    sensor_res = supabase.table("sensor_data").select("*").order("timestamp", desc=True).limit(1).execute()
    planting_res = supabase.table("planting_info").select("*").order("planting_date", desc=True).limit(1).execute()

    if not sensor_res.data or not planting_res.data:
        return {"error": "No data available for irrigation calculation."}

    sensor = sensor_res.data[0]
    planting = planting_res.data[0]

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
    adjusted_time = base_time * area * stage_multiplier.get(stage, 1.0)

    return {
        "irrigation_needed": irrigation_needed,
        "duration_minutes": round(adjusted_time, 2),
        "growth_stage": stage
    }

# Manual input data model
class PlantingInput(BaseModel):
    irrigation_type: str
    planting_area: float  # Use float because numeric in Supabase allows decimals

# Save manual input to planting_info
@app.post("/manual-input")
def manual_input(data: PlantingInput):
    new_entry = {
        "id": str(uuid.uuid4()),
        "plant_id": "plant_1",
        "planting_date": datetime.now().date().isoformat(),
        "irrigation_type": data.irrigation_type,
        "planting_area": data.planting_area
    }

    supabase.table("planting_info").insert(new_entry).execute()
    return {"message": "✅ Manual input saved!"}


from fastapi import FastAPI
from supabase import create_client, Client
from datetime import datetime

app = FastAPI()

# Supabase connection
import os

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(url, key)

@app.get("/")
def home():
    return {"message": "✅ API working!"}

@app.get("/irrigation")
def calculate_irrigation():
    # fetch + compute logic
    return {
        "irrigation_needed": True,
        "duration_minutes": 3.2,
        "growth_stage": "Mid Growth"
    }

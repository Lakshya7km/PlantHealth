
from fastapi import FastAPI
from supabase import create_client, Client
from datetime import datetime

app = FastAPI()

# Supabase connection
url = "https://lcmsxmciopzkdldekbmw.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxjbXN4bWNpb3B6a2RsZGVrYm13Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM2OTk2MTgsImV4cCI6MjA1OTI3NTYxOH0.mPPJbRQNN-rNLnZfMEpJ2siWgD9TtnWnEYj91broSMA"

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

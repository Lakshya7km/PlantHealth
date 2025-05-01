import pandas as pd
from supabase import create_client, Client
from datetime import datetime

# Supabase credentials
url = "https://lcmsxmciopzkdldekbmw.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxjbXN4bWNpb3B6a2RsZGVrYm13Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM2OTk2MTgsImV4cCI6MjA1OTI3NTYxOH0.mPPJbRQNN-rNLnZfMEpJ2siWgD9TtnWnEYj91broSMA"
supabase: Client = create_client(url, key)

# Read the CSV file
filename = "data1.csv"
data = pd.read_csv(filename, header=None)

# Extract header and data rows
header_row = data.iloc[6]
data.columns = header_row
filtered_data = data.iloc[7]

# Prepare record dictionary
record = {
    "timestamp": datetime.now().isoformat(),  # current timestamp in ISO format
    "temperature": float(filtered_data['tem(C)']),
    "humidity": float(filtered_data['Hum(%)']),
    "ph_level": float(filtered_data['pH']),
    "soil": float(filtered_data['Soil']),
    "soil_status": int(filtered_data['Soil status']),
    "co2_ppm": int(filtered_data['CO2(ppm)']),
    "co2_status": int(filtered_data['CO2 status']),
    "light": int(filtered_data['Light'])
}

# Replace NaN with None
record = {k: None if pd.isna(v) else v for k, v in record.items()}

# Insert the record into Supabase
response = supabase.table("sensor_data").insert(record).execute()
print("Insert response:", response)

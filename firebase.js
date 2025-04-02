import React, { useState, useEffect } from "react";
import { db } from "./firebase";
import { collection, addDoc, getDocs, deleteDoc, doc } from "firebase/firestore";
import { Button, Input } from "@/components/ui/button";

export default function SensorDataUI() {
  const [temperature, setTemperature] = useState("");
  const [humidity, setHumidity] = useState("");
  const [sensorData, setSensorData] = useState([]);

  // Fetch data from Firestore
  useEffect(() => {
    const fetchData = async () => {
      const querySnapshot = await getDocs(collection(db, "sensor_data"));
      setSensorData(querySnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() })));
    };
    fetchData();
  }, []);

  // Add Data to Firestore
  const addData = async () => {
    if (!temperature || !humidity) return;
    await addDoc(collection(db, "sensor_data"), { temperature, humidity });
    setTemperature("");
    setHumidity("");
    window.location.reload();
  };

  // Delete Data from Firestore
  const deleteData = async (id) => {
    await deleteDoc(doc(db, "sensor_data", id));
    window.location.reload();
  };

  return (
    <div className="p-4 max-w-md mx-auto bg-white shadow-md rounded-lg">
      <h2 className="text-xl font-bold mb-4">Sensor Data</h2>
      <div className="flex space-x-2 mb-4">
        <Input placeholder="Temperature" value={temperature} onChange={(e) => setTemperature(e.target.value)} />
        <Input placeholder="Humidity" value={humidity} onChange={(e) => setHumidity(e.target.value)} />
        <Button onClick={addData}>Add</Button>
      </div>
      <ul>
        {sensorData.map((data) => (
          <li key={data.id} className="flex justify-between p-2 border-b">
            {data.temperature}°C - {data.humidity}% 
            <Button variant="destructive" onClick={() => deleteData(data.id)}>Delete</Button>
          </li>
        ))}
      </ul>
    </div>
  );
}

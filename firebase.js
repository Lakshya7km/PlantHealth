<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sensor Data UI</title>
    <script type="module" src="firebaseConfig.js"></script>
    <script type="module" defer>
        import { initializeApp } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-app.js";
        import { getFirestore, collection, addDoc, getDocs, deleteDoc, doc } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-firestore.js";
        import { firebaseConfig } from "./firebaseConfig.js";

        const app = initializeApp(firebaseConfig);
        const db = getFirestore(app);

        async function fetchData() {
            const querySnapshot = await getDocs(collection(db, "sensor_data"));
            const sensorList = document.getElementById("sensor-list");
            sensorList.innerHTML = "";
            querySnapshot.forEach((doc) => {
                const data = doc.data();
                const li = document.createElement("li");
                li.innerHTML = `${data.temperature}°C - ${data.humidity}% <button onclick="deleteData('${doc.id}')">Delete</button>`;
                sensorList.appendChild(li);
            });
        }

        async function addData() {
            const temperature = document.getElementById("temperature").value;
            const humidity = document.getElementById("humidity").value;
            if (!temperature || !humidity) return;
            await addDoc(collection(db, "sensor_data"), { temperature, humidity });
            document.getElementById("temperature").value = "";
            document.getElementById("humidity").value = "";
            fetchData();
        }

        async function deleteData(id) {
            await deleteDoc(doc(db, "sensor_data", id));
            fetchData();
        }

        window.onload = fetchData;
    </script>
</head>
<body>
    <div>
        <h2>Sensor Data</h2>
        <input type="text" id="temperature" placeholder="Temperature" />
        <input type="text" id="humidity" placeholder="Humidity" />
        <button onclick="addData()">Add</button>
        <ul id="sensor-list"></ul>
    </div>
</body>
</html>

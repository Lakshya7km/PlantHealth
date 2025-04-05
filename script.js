const API_URL = 'https://your-render-app.onrender.com';

function formatDate(dateString) {
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    return new Date(dateString).toLocaleDateString(undefined, options);
}

function updateDashboard(data) {
    document.getElementById('remainingDays').textContent = data.growthPrediction.remainingDays;
    document.getElementById('growthPercentage').textContent = `${data.growthPrediction.growthPercentage}%`;
    document.getElementById('growthProgressBar').style.width = `${data.growthPrediction.growthPercentage}%`;
    document.getElementById('totalGrowthDays').textContent = data.growthPrediction.predictedTotalDays;
    document.getElementById('daysSincePlanting').textContent = data.growthPrediction.daysSincePlanting;
    document.getElementById('currentStage').textContent = data.growthPrediction.growthStage;

    const stageMap = { 'Seedling': 1, 'Vegetative': 2, 'Mature/Harvest': 3 };
    const currentStageNum = stageMap[data.growthPrediction.growthStage] || 1;
    for (let i = 1; i <= 3; i++) {
        const dot = document.getElementById(`stage${i}Dot`);
        dot.classList.toggle('active', i <= currentStageNum);
    }

    const waterStatus = document.getElementById('waterStatus');
    waterStatus.textContent = data.wateringInfo.needsWatering ? '💧 Watering Needed Today' : '✅ No Watering Needed Today';
    waterStatus.className = data.wateringInfo.needsWatering ? 'water-needed' : 'no-water-needed';

    document.getElementById('waterDuration').textContent = data.wateringInfo.waterDuration;
    document.getElementById('irrigationType').textContent = data.plantInfo.irrigationType;
    document.getElementById('plantingArea').textContent = data.plantInfo.plantingArea;
    document.getElementById('plantingDate').textContent = formatDate(data.plantInfo.plantingDate);
    document.getElementById('temperature').textContent = data.sensorData.temperature;
    document.getElementById('humidity').textContent = data.sensorData.humidity;
    document.getElementById('tdsValue').textContent = data.sensorData.tdsValue;
    document.getElementById('phLevel').textContent = data.sensorData.phLevel;
    document.getElementById('lastUpdated').textContent = `Last updated: ${new Date().toLocaleTimeString()}`;

    document.getElementById('growthLoader').style.display = 'none';
    document.getElementById('growthContent').style.display = 'block';
    document.getElementById('irrigationLoader').style.display = 'none';
    document.getElementById('irrigationContent').style.display = 'block';
    document.getElementById('sensorLoader').style.display = 'none';
    document.getElementById('sensorContent').style.display = 'block';
}

async function fetchPlantStatus() {
    try {
        document.getElementById('growthLoader').style.display = 'block';
        document.getElementById('growthContent').style.display = 'none';
        document.getElementById('irrigationLoader').style.display = 'block';
        document.getElementById('irrigationContent').style.display = 'none';
        document.getElementById('sensorLoader').style.display = 'block';
        document.getElementById('sensorContent').style.display = 'none';

        const response = await fetch(`${API_URL}/plant-status`);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        const data = await response.json();
        updateDashboard(data);
    } catch (error) {
        console.error('Error fetching plant status:', error);
        alert('Failed to fetch plant status. Please try again later.');
    }
}

async function resetGrowthCycle(formData) {
    try {
        const response = await fetch(`${API_URL}/reset-cycle`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        const data = await response.json();
        console.log('Reset successful:', data);

        const successAlert = document.getElementById('resetSuccess');
        successAlert.style.display = 'block';
        setTimeout(() => successAlert.style.display = 'none', 3000);

        fetchPlantStatus();
    } catch (error) {
        console.error('Error resetting growth cycle:', error);
        const errorAlert = document.getElementById('resetError');
        errorAlert.textContent = `Error: ${error.message}`;
        errorAlert.style.display = 'block';
        setTimeout(() => errorAlert.style.display = 'none', 3000);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    fetchPlantStatus();
    document.getElementById('refreshBtn').addEventListener('click', fetchPlantStatus);
    document.getElementById('resetForm').addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = {
            plantingDate: document.getElementById('plantingDate').value,
            irrigationType: document.getElementById('irrigationType').value,
            plantingArea: parseFloat(document.getElementById('plantingArea').value),
            plantId: document.getElementById('plantId').value
        };
        resetGrowthCycle(formData);
    });
});

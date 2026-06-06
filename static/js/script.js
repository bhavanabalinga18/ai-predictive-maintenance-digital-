const ctx = document.getElementById('sensorChart');

const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Temperature',
            data: [],
            borderWidth: 2,
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false
    }
});

function loadData() {

    fetch('/get_latest_data')
    .then(response => response.json())
    .then(data => {

        document.getElementById('temp').innerHTML =
            data.temperature + ' °C';

        document.getElementById('vibration').innerHTML =
            data.vibration;

        document.getElementById('rpm').innerHTML =
            data.rpm;

        document.getElementById('toolwear').innerHTML =
            data.tool_wear;

        chart.data.labels.push(
            new Date().toLocaleTimeString()
        );

        chart.data.datasets[0].data.push(
            data.temperature
        );

        if (chart.data.labels.length > 10) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }

        chart.update();
    })
    .catch(error => {
        console.log("Error loading data:", error);
    });
}

function predictFailure() {

    fetch('/get_latest_data')
    .then(response => response.json())
    .then(sensor => {

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                temperature: sensor.temperature,
                vibration: sensor.vibration,
                rpm: sensor.rpm,
                tool_wear: sensor.tool_wear
            })
        })

        .then(response => response.json())
        .then(result => {

            document.getElementById('predictionResult').innerHTML =
                result.status;

            let alertBox =
                document.getElementById('alertBox');

            if (result.prediction === 1) {

                alertBox.className =
                    'alert-box danger';

                alertBox.innerHTML =
                    '⚠ Failure Predicted';

            } else {

                alertBox.className =
                    'alert-box safe';

                alertBox.innerHTML =
                    '✅ Machine Healthy';
            }
        });
    });
}

function futurePrediction() {

    fetch('/future_prediction')
    .then(response => response.json())
    .then(data => {

        document.getElementById('futureTemp').innerHTML =
            data.future_temperature;

        document.getElementById('futureVibration').innerHTML =
            data.future_vibration;

        document.getElementById('futureRPM').innerHTML =
            data.future_rpm;
    });
}

setInterval(loadData, 3000);

loadData();

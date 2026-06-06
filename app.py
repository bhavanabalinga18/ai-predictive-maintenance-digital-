from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open('failure_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load dataset
data = pd.read_csv('industrial_drilling_dataset.csv')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_latest_data')
def get_latest_data():

    latest = data.iloc[-1]

    response = {
        "temperature": float(latest["Temperature"]),
        "vibration": float(latest["Vibration"]),
        "rpm": float(latest["RPM"]),
        "tool_wear": float(latest["Tool_Wear"])
    }

    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():

    input_data = request.json

    features = np.array([
        [
            float(input_data['temperature']),
            float(input_data['vibration']),
            float(input_data['rpm']),
            float(input_data['tool_wear'])
        ]
    ])

    prediction = model.predict(features)[0]

    if prediction == 1:
        status = "Failure Predicted"
    else:
        status = "Machine Healthy"

    return jsonify({
        "prediction": int(prediction),
        "status": status
    })


@app.route('/future_prediction')
def future_prediction():

    latest = data.iloc[-1]

    future_temp = round(float(latest["Temperature"]) + np.random.uniform(-2, 3), 2)
    future_vibration = round(float(latest["Vibration"]) + np.random.uniform(-0.5, 0.5), 2)
    future_rpm = round(float(latest["RPM"]) + np.random.uniform(-50, 50), 2)

    return jsonify({
        "future_temperature": future_temp,
        "future_vibration": future_vibration,
        "future_rpm": future_rpm
    })


if __name__ == '__main__':
    app.run(debug=True)

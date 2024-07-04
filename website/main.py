from flask import Flask, render_template
from flask import request
import pickle
import xgboost as xgb
import numpy as np

# Default dari template_folder adalah 'templates'
app = Flask(__name__, template_folder='templates', static_folder='staticFiles')

model = xgb.XGBRegressor(n_estimators=100)
model.load_model('model/model.json')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inputDuration = request.form['duration']
    inputDuration = float(inputDuration)
    inputDurationSec = inputDuration*60
    inputDistance = request.form['distance']
    inputDistance = float(inputDistance)
    inputPassenger = request.form['passenger']
    inputPassenger = float(inputPassenger)

    inputData = np.array([[inputDurationSec, inputDistance, inputPassenger]])

    pred = model.predict(inputData)
    res = round(pred[0], 2)

    return render_template('result.html', prediction=res, duration=inputDuration, distance=inputDistance, passenger=int(inputPassenger))

app.run(debug=True)
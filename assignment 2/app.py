
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model/crop_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorus'])
        K = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)

        return render_template('result.html', prediction_text=f'Recommended Crop: {prediction[0]}')

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model
model = joblib.load('model/crop_recommendation.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [
            float(request.form['nitrogen']),
            float(request.form['phosphorus']),
            float(request.form['potassium']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        return render_template('result.html', prediction=prediction)
        
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)
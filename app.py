from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the model
with open('addclick1.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the LabelEncoder
label_enc = LabelEncoder()

# Initialize Flask app
app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        daily_time_spent = float(request.form['daily_time_spent'])
        age = int(request.form['age'])
        area_income = float(request.form['area_income'])
        daily_internet_usage = float(request.form['daily_internet_usage'])
        ad_topic_line = request.form['ad_topic_line']
        city = request.form['city']
        country = request.form['country']
        
        # Get the value for the 'Male' feature (1 for male, 0 for female)
        male = int(request.form['male'])  # Ensure this matches your HTML form input
        
        # Encode categorical variables
        ad_topic_line_encoded = label_enc.fit_transform([ad_topic_line])[0]
        city_encoded = label_enc.fit_transform([city])[0]
        country_encoded = label_enc.fit_transform([country])[0]
        
        # Prepare the input for the model including 'Male'
        input_data = np.array([[daily_time_spent, age, area_income, daily_internet_usage, ad_topic_line_encoded, city_encoded, country_encoded, male]])
        
        # Make a prediction
        prediction = model.predict(input_data)
        output = prediction[0]
        
        # Prepare prediction text
        prediction_text = f'Likelihood of clicking on the ad: {"Yes" if output == 1 else "No"}'
        
        return render_template('result.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)

# Load the trained TensorFlow model with custom objects
best_model = tf.keras.models.load_model('ANN.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling the form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form
        features = [float(x) for x in request.form.values()]
        
        # Convert features to a numpy array
        input_data = np.array([features])

        # Scale the input data using the same scaler used during training
        input_data_scaled = scaler.transform(input_data)

        # Make prediction using the loaded model
        prediction = best_model.predict(input_data_scaled)[0][0]

        return render_template('index.html', prediction_text=f'Predicted Median Value in Thousands of Dollars: {prediction:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
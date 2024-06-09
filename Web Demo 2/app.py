from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained RandomForestRegressor model (assuming it's saved as a pickle file)
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

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

        # Make prediction using the loaded model
        prediction = best_model.predict(input_data)[0]

        return render_template('index.html', prediction_text=f'Predicted Median Value in Thousands of Dollars: {prediction:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
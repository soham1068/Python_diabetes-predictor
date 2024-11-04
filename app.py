from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained models
with open('naive_bayes_model.pkl', 'rb') as nb_file:
    loaded_nb_model = pickle.load(nb_file)

with open('perceptron_model_.pkl', 'rb') as perceptron_file:
    loaded_perceptron_model = pickle.load(perceptron_file)


# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_type = data.get('model_type', 'naive_bayes')  # Default to Naive Bayes

    # Prepare input features array
    input_features = np.array([[data['Age'], data['Glucose'], data['Insulin'], data['BMI']]])

    # Choose the model based on input
    if model_type == 'naive_bayes':
        prediction = loaded_nb_model.predict(input_features)
    elif model_type == 'perceptron':
        prediction = loaded_perceptron_model.predict(input_features)
    else:
        return jsonify({'error': 'Invalid model type'}), 400
    print(int(prediction[0]))
    return jsonify({'diabetes_type': int(prediction[0])})  # Return prediction as JSON


# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import requests
from PIL import Image
import io
import numpy as np
import os

app = Flask(__name__)

# GitHub Model URL (Replace with your actual release link)
MODEL_URL = "https://github.com/brandonowens24/PSA/releases/download/v0.0.1/PSA_grading_model.h5"
MODEL_PATH = "PSA_grading_model.h5"

def download_model():
    """Download the model from GitHub if it doesn’t exist locally."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from GitHub...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Model downloaded successfully.")

def preprocess_image(image):
    """Preprocesses the input image for model inference."""
    image = image.convert("RGB")  # Ensure image is in RGB mode
    image = image.resize((600, 600))  # Adjust size based on model training
    image = np.array(image) / 255.0  # Normalize pixel values
    return image

def format_prediction(prediction):
    """Extracts class label from model prediction."""
    return np.argmax(prediction, axis=1)[0]

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image uploads and returns a PSA grading prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        # Predict PSA grade
        prediction = model.predict(np.expand_dims(processed_image, axis=0))
        response = format_prediction(prediction)
        
        return jsonify({'message': f'We predict your card is rated PSA: {response}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    download_model()  # Download model at startup
    model = tf.keras.models.load_model(MODEL_PATH)  # Load model
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import requests
from PIL import Image
import io
import numpy as np
import zipfile
import os

app = Flask(__name__)

# GitHub Model URL
MODEL_URL = "https://github.com/brandonowens24/PSA/releases/download/v0.0.2/PSA_grading_model.h5.zip"
ZIP_PATH = "PSA_grading_model.zip"
MODEL_PATH = "PSA_grading_model.h5"

def download_and_extract_model():
    """Download and extract the model if it's not already present."""
    if not os.path.exists(MODEL_PATH):  # Check if model is already extracted
        print("Downloading model from GitHub...")
        response = requests.get(MODEL_URL, stream=True)
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Download complete. Extracting model...")

        # Extract files from ZIP
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall()  # Extract to current folder
        os.remove(ZIP_PATH)  # Cleanup ZIP file

        print("Model extracted successfully.")

def preprocess_image(image):
    """Preprocesses the input image for model inference."""
    image = image.convert("RGB")
    image = image.resize((600, 600))  # Resize image to match model input
    image = np.array(image) / 255.0  # Normalize pixel values
    return image

def format_prediction(prediction):
    """Extracts class label from model prediction."""
    return np.argmax(prediction, axis=1)[0]  # Get the class with the highest probability

@app.route('/')
def home():
    """Root route that serves the HTML page."""
    return render_template('index.html')  # Ensure this HTML file is in the 'templates' folder

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
    download_and_extract_model()  # Ensure the model is downloaded and extracted
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # Load the model without compiling
    app.run(debug=True)  # Start the Flask app

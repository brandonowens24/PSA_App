from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("../modeling/PSA_grading_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Image not found.'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Open the image file
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Preprocess new image
        processed_image = preprocess_image(image)
        
        # Predict distribution
        prediction = model.predict(np.expand_dims(processed_image, axis=0))
        
        # Obtain class number
        response = format_prediction(prediction)
        
        # Return message
        return jsonify({'message': f'We predict your card is rated PSA: {response}'})
    except Exception as e:
        return jsonify({'error': str(e)})

def preprocess_image(image):
    image = image.resize((600, 600))
    image = np.array(image) / 255.0 
    return image

def format_prediction(prediction):
    return np.argmax(prediction, axis=1)[0] 

if __name__ == '__main__':
    app.run(debug=True)

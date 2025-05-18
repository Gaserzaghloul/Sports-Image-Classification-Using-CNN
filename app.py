from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import traceback
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Load model and setup preprocessing
model = tf.keras.models.load_model('final_model_edited.keras')
train_df = pd.read_csv('dataset/train.csv')
le = LabelEncoder().fit(train_df['label'])

# Normalization stats (must match your train-time preprocessing)
TRAIN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
TRAIN_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(img_bytes):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image (corrupt or wrong format)")
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = (img - TRAIN_MEAN) / TRAIN_STD
    return img.astype(np.float32)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No images provided'}), 400

        # Build batch
        batch = np.stack([preprocess_image(f.read()) for f in files], axis=0)

        # Model inference
        preds = model.predict(batch, verbose=0)
        idxs = preds.argmax(axis=1)

        # Map back to your actual labels
        labels = le.inverse_transform(idxs)

        return jsonify({'predictions': labels.tolist()})

    except Exception as e:
        traceback.print_exc()
        tb = traceback.format_exc().splitlines()[-5:]
        return jsonify({'error': str(e), 'traceback': tb}), 500

@app.route('/')
def home():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False) 
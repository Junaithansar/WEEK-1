# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

st.set_page_config(page_title="AGRISCAN – Crop Disease Detection", layout="centered")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Suppress TensorFlow logging
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ---------------- Load Model and Labels ----------------
@st.cache_resource
def load_model():
    model_path = os.path.join(script_dir, "model.h5")
    labels_path = os.path.join(script_dir, "labels.txt")
    
    try:
        # Try loading with safe_mode disabled for better compatibility
        model = tf.keras.models.load_model(model_path, safe_mode=False)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Attempting alternative loading method...")
        try:
            # Fallback: load with custom_objects
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e2:
            st.error(f"Failed to load model: {str(e2)}")
            return None, None
    
    try:
        with open(labels_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error(f"Labels file not found at {labels_path}")
        return model, None
    
    return model, class_names

model, class_names = load_model()

# Check if model loaded successfully
if model is None or class_names is None:
    st.error("Failed to load the model or labels. Please check the model files.")
    st.stop()

# ---------------- Preprocess Image ----------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- UI ----------------
st.title("🌿 AGRISCAN – Intelligent Crop Disease Detection System")
st.write("Upload a crop leaf image to identify whether it is healthy or diseased.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Analyzing...")

    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

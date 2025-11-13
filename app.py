# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.layers import DepthwiseConv2D as KDepthwiseConv2D
import os

st.set_page_config(page_title="AGRISCAN – Crop Disease Detection", layout="centered")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Suppress TensorFlow logging
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Compatibility wrapper for models saved with a different Keras/TensorFlow version
# Some older models serialize a 'groups' kwarg into DepthwiseConv2D config which
# newer Keras DepthwiseConv2D does not accept. This wrapper accepts and ignores
# the 'groups' argument so the layer can be deserialized.
class DepthwiseConv2DCompat(KDepthwiseConv2D):
    def __init__(self, *args, groups=None, **kwargs):
        # Accept 'groups' for backward compatibility and ignore it
        super().__init__(*args, **kwargs)

# ---------------- Load Model and Labels ----------------
@st.cache_resource
def load_model():
    # Prefer repaired model if it exists
    fixed_path = os.path.join(script_dir, "model_fixed.h5")
    model_path = os.path.join(script_dir, "model.h5")
    labels_path = os.path.join(script_dir, "labels.txt")
    
    # Try repaired model first
    try_paths = []
    if os.path.exists(fixed_path):
        try_paths.append((fixed_path, 'repaired'))
    try_paths.append((model_path, 'original'))

    model = None
    for p, kind in try_paths:
        try:
            model = tf.keras.models.load_model(
                p,
                custom_objects={"DepthwiseConv2D": DepthwiseConv2DCompat},
                compile=False,
            )
            if model is not None:
                if kind == 'repaired':
                    st.info(f'Loaded repaired model from {p}')
                else:
                    st.info(f'Loaded model from {p}')
                break
        except Exception as e:
            st.warning(f'Could not load {kind} model ({p}): {e}')

    if model is None:
        st.error('Failed to load model from both repaired and original files.')
        return None, None
    
    try:
        with open(labels_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error(f"Labels file not found at {labels_path}")
        return model, None
    
    return model, class_names

model, class_names = load_model()

# Allow app to run even if model is unavailable (for deployment)
model_available = model is not None and class_names is not None
if not model_available:
    st.warning("⚠️ Model is currently unavailable. You can still upload images, but predictions are disabled.")

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
    
    if model_available:
        st.write("Analyzing...")
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")
    else:
        st.info("Model predictions are currently unavailable. Check back soon!")

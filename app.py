# app.py
# AGRISCAN – Intelligent Crop Disease Detection System
# Developed by: Junaith Ansar

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.layers import DepthwiseConv2D as KDepthwiseConv2D
import os
import logging
import random
import time

# ================== PAGE CONFIGURATION ==================
st.set_page_config(page_title="AGRISCAN – Crop Disease Detection", layout="centered")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ================== COMPATIBILITY WRAPPER ==================
class DepthwiseConv2DCompat(KDepthwiseConv2D):
    """Wrapper to handle models saved with different Keras/TensorFlow versions."""
    def __init__(self, *args, groups=None, **kwargs):
        super().__init__(*args, **kwargs)

# ================== LOAD MODEL AND LABELS ==================
@st.cache_resource
def load_model():
    """Load model with fallback support for repaired and original models."""
    fixed_path = os.path.join(script_dir, "model_fixed.h5")
    model_path = os.path.join(script_dir, "model.h5")
    labels_path = os.path.join(script_dir, "labels.txt")
    
    try_paths = []
    if os.path.exists(fixed_path):
        try_paths.append((fixed_path, 'repaired'))
    if os.path.exists(model_path):
        try_paths.append((model_path, 'original'))

    model = None
    loaded_from = None
    
    for p, kind in try_paths:
        try:
            model = tf.keras.models.load_model(
                p,
                custom_objects={"DepthwiseConv2D": DepthwiseConv2DCompat},
                compile=False,
            )
            if model is not None:
                loaded_from = kind
                break
        except Exception as e:
            continue

    class_names = None
    if model is not None:
        try:
            with open(labels_path, "r") as f:
                class_names = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            pass
    
    return model, class_names, loaded_from

model, class_names, model_source = load_model()

# ================== MODEL AVAILABILITY CHECK ==================
model_available = model is not None and class_names is not None
if not model_available:
    st.warning("⚠️ Model is currently unavailable. You can still upload images, but predictions are disabled. Check back soon!")

# ================== PREPROCESS IMAGE ==================
def preprocess_image(img):
    """Preprocess image for model input (224x224, normalized)."""
    img = img.convert("RGB")
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ================== ANIMATED PREDICTION REVEAL ==================
def show_animated_prediction(predicted_class, confidence, class_names):
    """Show animated prediction reveal with random disease names cycling."""
    
    # Get random disease names for animation (excluding the actual prediction)
    other_diseases = [name for name in class_names if name != predicted_class]
    random.shuffle(other_diseases)
    
    # Create placeholder for animated reveal
    placeholder = st.empty()
    
    # Animate through random predictions with confidence in 75-90% range
    animation_steps = 8
    for step in range(animation_steps):
        random_disease = random.choice(other_diseases)
        random_confidence = random.uniform(75, 90)
        with placeholder.container():
            st.success(f"**Prediction:** {random_disease}")
            st.info(f"**Confidence:** {random_confidence:.2f}%")
        time.sleep(0.3)
    
    # Reveal the actual prediction
    with placeholder.container():
        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")
    
    return predicted_class

# ================== DISEASE INFORMATION & SUGGESTIONS ==================
disease_info = {
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "cause": "Caused by a fungus (Alternaria solani) that thrives in warm, humid conditions.",
        "solution": [
            "Remove infected leaves to prevent spread.",
            "Avoid overhead watering to keep leaves dry.",
            "Use organic fungicides like neem oil or copper-based sprays."
        ],
        "sustainability_tip": "Use crop rotation and organic compost to naturally strengthen soil health."
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "cause": "Caused by Phytophthora infestans fungus, especially during wet weather.",
        "solution": [
            "Destroy infected plants immediately.",
            "Use resistant tomato varieties.",
            "Apply bio-fungicides and maintain good air circulation."
        ],
        "sustainability_tip": "Avoid excessive moisture and use natural disease-resistant crops."
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "cause": "Plant appears healthy with no visible disease symptoms.",
        "solution": [
            "Maintain regular watering and sunlight exposure.",
            "Continue monitoring leaf color and texture.",
            "Use organic fertilizers to support plant growth."
        ],
        "sustainability_tip": "Keep soil rich in organic matter and use natural compost regularly."
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "cause": "Fungal infection due to Alternaria solani.",
        "solution": [
            "Use disease-free certified seeds.",
            "Apply neem oil weekly to control fungal spores.",
            "Remove old crop debris to reduce infection sources."
        ],
        "sustainability_tip": "Follow a crop rotation system to maintain soil biodiversity."
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "cause": "Caused by Phytophthora infestans, spreads rapidly in moist conditions.",
        "solution": [
            "Use resistant potato varieties.",
            "Improve drainage and air circulation.",
            "Apply bio-fungicides during growing season."
        ],
        "sustainability_tip": "Practice intercropping and maintain biodiversity in fields."
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "cause": "Plant appears healthy with no visible disease symptoms.",
        "solution": [
            "Maintain consistent watering schedule.",
            "Monitor soil pH and nutrient levels.",
            "Use organic mulch to retain moisture."
        ],
        "sustainability_tip": "Use natural pest management techniques to preserve soil ecosystem."
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "cause": "Caused by Guignardia bidwellii fungus during humid weather.",
        "solution": [
            "Prune infected leaves and ensure good sunlight exposure.",
            "Use sulfur-based organic fungicides.",
            "Avoid overwatering and dense planting."
        ],
        "sustainability_tip": "Encourage beneficial insects and avoid chemical overuse."
    },
    "Grape___Esca": {
        "plant": "Grape",
        "disease": "Esca",
        "cause": "Fungal wood disease that affects grapevines, causing leaf discoloration.",
        "solution": [
            "Prune affected wood and remove diseased branches.",
            "Apply wound-healing compounds to cut surfaces.",
            "Ensure proper canopy management for air circulation."
        ],
        "sustainability_tip": "Maintain healthy vineyard soil with balanced nutrient management."
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "cause": "Plant appears healthy with no visible disease symptoms.",
        "solution": [
            "Maintain regular pruning for optimal growth.",
            "Ensure adequate water and sunlight.",
            "Monitor for early signs of disease."
        ],
        "sustainability_tip": "Use organic fertilizers and encourage natural predators of pests."
    }
}

# ================== WEB INTERFACE ==================
st.title("🌿 AGRISCAN – Intelligent Crop Disease Detection System")
st.write("Upload a crop leaf image to identify whether it is healthy or diseased, and get expert suggestions for treatment.")

uploaded_file = st.file_uploader("📷 Upload a Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if model_available:
        img_array = preprocess_image(image)
        predictions = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])

        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        # Show animated prediction reveal
        show_animated_prediction(predicted_class, confidence, class_names)

        # Display additional info if available
        if predicted_class in disease_info:
            info = disease_info[predicted_class]
            st.markdown(f"### 🌱 **Plant:** {info['plant']}")
            st.markdown(f"### 🦠 **Disease/Status:** {info['disease']}")
            st.markdown(f"**Cause:** {info['cause']}")
            st.markdown("**✅ Recommended Actions:**")
            for step in info['solution']:
                st.markdown(f"- {step}")
            st.markdown(f"**♻️ Sustainability Tip:** {info['sustainability_tip']}")
        else:
            st.warning(f"No specific suggestion available for '{predicted_class}'. Please consult an agricultural expert.")
    else:
        st.info("📌 Model predictions are currently unavailable. You can still view your uploaded image, but disease detection is disabled.")

else:
    st.info("👆 Please upload a leaf image to start the detection.")

# ================== FOOTER / CREDITS ==================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 16px; margin-top: 20px;">
        🌿 This application is developed under 
        <a href="https://github.com/Junaithansar" target="_blank" style="text-decoration: none; color: #2e8b57; font-weight: bold;">
            GitHub: Junaith Ansar
        </a><br>
        <span style="font-size: 14px; color: gray;">© 2025 AGRISCAN – Intelligent Crop Disease Detection System</span>
    </div>
    """,
    unsafe_allow_html=True
)

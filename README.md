# AGRISCAN - Intelligent Crop Disease Detection System

An AI-powered Streamlit application that uses deep learning to detect crop diseases from leaf images.

## Features
- Upload crop leaf images (JPG, JPEG, PNG)
- Real-time disease classification using TensorFlow/Keras
- Confidence score for predictions
- Support for multiple crop disease categories

## Installation

### Local Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deployment

### Streamlit Community Cloud
1. Push this repository to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app" and select this repository
5. Your app will be live in minutes!

## Model Files Required
- `model.h5` - Trained TensorFlow/Keras model
- `labels.txt` - Class labels (one per line)

## Usage
1. Open the app in your browser (localhost:8501 for local, or the Streamlit Cloud URL)
2. Click "Upload Image" and select a crop leaf image
3. Wait for the analysis
4. View the disease prediction and confidence score

## Technologies
- **Streamlit** - Web framework
- **TensorFlow/Keras** - Deep learning framework
- **Python** - Programming language
- **PIL** - Image processing

## Author
AGRISCAN Development Team

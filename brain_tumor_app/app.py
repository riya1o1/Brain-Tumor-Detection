import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- App Config ---
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# --- Custom CSS for better UI ---
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            padding: 10px;
        }
        .subtitle {
            font-size: 20px;
            color: #555;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-box {
            border: 2px dashed #aaa;
            padding: 20px;
            border-radius: 15px;
            background-color: #fafafa;
            margin-bottom: 20px;
        }
        .predict-box {
            border: 2px solid #1f77b4;
            padding: 20px;
            border-radius: 15px;
            background-color: #f0f8ff;
            text-align: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="title">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an MRI image to check for presence of tumor</div>', unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_brain_tumor_model():
    return load_model("brain_tumor_model.h5")

model = load_brain_tumor_model()

# --- Upload Image ---
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction ---
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼 Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((150, 150)).convert("RGB")  # ensure 3 channels
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize

    # Predict
    prediction = model.predict(img_array)
    score = float(prediction[0][0])
    label = "🧠 Tumor Detected" if score > 0.5 else "✅ No Tumor Detected"
    confidence = score if score > 0.5 else 1 - score

    # Show Result
    st.markdown('<div class="predict-box">', unsafe_allow_html=True)
    st.subheader("🔍 Prediction Result")
    st.write(f"**Result:** {label}")
    st.write(f"**Confidence:** {confidence:.2%}")
    st.progress(confidence)
    st.markdown('</div>', unsafe_allow_html=True)

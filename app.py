import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.title("🤟 Indian Sign Language Recognition")
st.write("Upload a hand sign image (A–Z, 0–9) and let the AI predict it.")

# File info
MODEL_PATH = "resnet50_full_model.keras"
DRIVE_FILE_ID = "1T2cbk4txFnKDJnLsZZzbGClhPjEqkcPG"

# Download model from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("⏬ Downloading model from Google Drive..."):
        gdown.download(
            id=DRIVE_FILE_ID,
            output=MODEL_PATH,
            quiet=False
        )

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = prediction[0][index] * 100

    st.markdown(f"### 🧠 Prediction: `{class_labels[index]}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")

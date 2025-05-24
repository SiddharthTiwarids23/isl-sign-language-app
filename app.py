import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --- Download model and class_order if not present ---
MODEL_PATH = "resnet50_full_model.keras"
CLASS_ORDER_PATH = "class_order.npy"

MODEL_ID = "1-3DpM6EckF3pAb-JGS6nUE1pmCEwnFpr"
CLASS_ORDER_ID = "1-6RDPKAELZo5JlOkIjtf2FJFcnHB8Ktr"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)

if not os.path.exists(CLASS_ORDER_PATH):
    with st.spinner("Downloading class labels..."):
        gdown.download(id=CLASS_ORDER_ID, output=CLASS_ORDER_PATH, quiet=False)

# --- Load model and class labels ---
model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASS_ORDER_PATH)

# --- Preprocessing function ---
def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize((224, 224))
    arr = tf.keras.utils.img_to_array(image)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# --- Streamlit UI ---
st.set_page_config(page_title="Indian Sign Language Recognition", layout="centered")
st.title("🤟 Indian Sign Language Recognition")
st.write("Upload an image of a hand gesture to identify the corresponding ISL class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    input_tensor = preprocess_image(image)
    prediction = model.predict(input_tensor, verbose=0)
    confidence = np.max(prediction)
    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]

    # Display result
    st.markdown("###  Prediction:")
    st.write(f"**Class:** `{predicted_class}`")
    st.write(f"**Confidence:** `{confidence:.2%}`")

    if confidence < 0.5:
        st.warning(" Low confidence. The prediction may not be reliable.")

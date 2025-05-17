import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('resnet50_full_model.h5')
    return model

model = load_model()

# Class labels (adjust based on your dataset)
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

st.title("Indian Sign Language Recognition")
st.write("Upload an image of a hand sign to identify the letter or number.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    pred_label = class_labels[pred_index]
    confidence = prediction[0][pred_index] * 100

    st.markdown(f"### Prediction: `{pred_label}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from config import IMAGE_SIZE, EMOTION_LABELS

st.title("🖼️ Image Emotion Detection")

model = load_model("models/image_model.keras")

img_file = st.file_uploader("Upload Face Image", type=["jpg", "png"])

if img_file:
    img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    img = img.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    preds = model.predict(img)[0]

    emotion = EMOTION_LABELS[np.argmax(preds)]
    confidence = np.max(preds)

    st.image(img, caption="Uploaded Image", clamp=True)
    st.success(f"Emotion: {emotion}")
    st.info(f"Confidence: {confidence:.2f}")

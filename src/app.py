# src/app.py

import streamlit as st
import numpy as np
import cv2
import librosa
import tempfile

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import IMAGE_SIZE, EMOTION_LABELS, MAX_TEXT_LEN, SAMPLE_RATE, N_MFCC


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    fusion = load_model("models/fusion_model.keras")
    image_model = load_model("models/image_model.keras")
    audio_model = load_model("models/audio_model.keras")
    text_model  = load_model("models/text_model.keras")
    tokenizer   = np.load("models/tokenizer.npy", allow_pickle=True).item()
    return fusion, image_model, audio_model, text_model, tokenizer


fusion_model, image_model, audio_model, text_model, tokenizer = load_models()


# =========================
# UI
# =========================
st.title("🎭 Multimodal Emotion Detection")


# =====================================================
# 📷 IMAGE SECTION
# =====================================================
st.header("📷 Image Emotion Detection")

img_file = st.file_uploader("Upload Face Image", type=["jpg", "png"], key="img")

img_pred = None

if img_file:

    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # SHOW INPUT IMAGE
    st.image(img, caption="Your Uploaded Image", use_column_width=True)

    # PREPROCESS
    processed_img = cv2.resize(img, IMAGE_SIZE)
    processed_img = processed_img / 255.0
    processed_img = processed_img.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    img_pred = image_model.predict(processed_img)

    # PREDICT BUTTON
    if st.button("Predict Emotion from Image"):
        emotion = EMOTION_LABELS[np.argmax(img_pred)]
        confidence = np.max(img_pred)

        st.success(f"Emotion: {emotion}")
        st.info(f"Confidence: {confidence:.2f}")


# =====================================================
# 🎤 AUDIO SECTION
# =====================================================
st.header("🎤 Audio Emotion Detection")

audio_file = st.file_uploader("Upload Audio (.wav)", type=["wav"], key="audio")

audio_pred = None

if audio_file:

    # PLAY INPUT AUDIO
    st.audio(audio_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = mfcc.reshape(1, -1)

    audio_pred = audio_model.predict(mfcc)

    # PREDICT BUTTON
    if st.button("Predict Emotion from Audio"):
        emotion = EMOTION_LABELS[np.argmax(audio_pred)]
        confidence = np.max(audio_pred)

        st.success(f"Emotion: {emotion}")
        st.info(f"Confidence: {confidence:.2f}")


# =====================================================
# 📝 TEXT SECTION
# =====================================================
st.header("📝 Text Emotion Detection")

text_input = st.text_input("Enter text:")

text_pred = None

if text_input.strip():

    # SHOW INPUT TEXT
    st.write("Your Text:", text_input)

    seq = tokenizer.texts_to_sequences([text_input])
    padded = pad_sequences(seq, maxlen=MAX_TEXT_LEN)

    text_pred = text_model.predict(padded)

    # PREDICT BUTTON
    if st.button("Predict Emotion from Text"):
        emotion = EMOTION_LABELS[np.argmax(text_pred)]
        confidence = np.max(text_pred)

        st.success(f"Emotion: {emotion}")
        st.info(f"Confidence: {confidence:.2f}")

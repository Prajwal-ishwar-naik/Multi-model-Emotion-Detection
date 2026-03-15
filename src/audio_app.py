import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from config import SAMPLE_RATE, N_MFCC, EMOTION_LABELS

st.title("🎤 Audio Emotion Detection")

model = load_model("models/audio_model.keras")

audio_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])

if audio_file:
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = mfcc.reshape(1, -1)

    preds = model.predict(mfcc)[0]

    emotion = EMOTION_LABELS[np.argmax(preds)]
    confidence = np.max(preds)

    st.audio(audio_file)
    st.success(f"Emotion: {emotion}")
    st.info(f"Confidence: {confidence:.2f}")

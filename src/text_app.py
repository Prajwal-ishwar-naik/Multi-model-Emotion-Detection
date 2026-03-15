import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import EMOTION_LABELS, MAX_TEXT_LEN

st.title("📝 Text Emotion Detection")

model = load_model("models/text_model.keras")
tokenizer = np.load("models/tokenizer.npy", allow_pickle=True).item()

text = st.text_area(
    "Enter text expressing emotion",
    placeholder="I am feeling very happy today"
)

if st.button("Predict Emotion"):
    if text.strip():
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_TEXT_LEN)

        preds = model.predict(padded)[0]

        emotion = EMOTION_LABELS[np.argmax(preds)]
        confidence = np.max(preds)

        st.success(f"Emotion: {emotion}")
        st.info(f"Confidence: {confidence}")
    else:
        st.warning("Please enter some text")

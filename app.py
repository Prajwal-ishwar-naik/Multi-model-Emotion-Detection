import streamlit as st
import numpy as np
import tensorflow as tf

# ---------------- CONFIG ----------------
NUM_CLASSES = 11

EMOTION_LABELS = [
    "Angry", "Disgust", "Fear", "Happy", "Sad",
    "Surprise", "Neutral", "Calm", "Bored",
    "Excited", "Frustrated"
]

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/fusion_model.keras")

model = load_model()

# ---------------- LOAD FEATURES ----------------
@st.cache_data
def load_features():
    img = np.load("data/processed/fusion/image_features.npy")
    aud = np.load("data/processed/fusion/audio_features.npy")
    txt = np.load("data/processed/fusion/text_features.npy")
    return img, aud, txt

img_feat, aud_feat, txt_feat = load_features()

# ---------------- UI ----------------
st.set_page_config(
    page_title="Multimodal Emotion Detection",
    page_icon="🎭",
    layout="centered"
)

st.markdown("""
<style>
.big-font {
    font-size:40px !important;
    font-weight:700;
    color:#ff4b4b;
}
.card {
    padding:20px;
    border-radius:15px;
    background-color:#f5f5f5;
}
</style>
""", unsafe_allow_html=True)

st.title("🎭 Multimodal Emotion Detection")
st.caption("Image + Audio + Text Fusion Model")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")
sample_id = st.sidebar.slider(
    "Select Sample",
    0,
    min(len(img_feat), len(aud_feat), len(txt_feat)) - 1,
    0
)

predict_btn = st.sidebar.button("🚀 Predict Emotion")

# ---------------- MAIN ----------------
st.markdown("### 📊 Selected Sample")
st.write(f"Sample ID: **{sample_id}**")

if predict_btn:

    img = img_feat[sample_id].reshape(1, -1)
    aud = aud_feat[sample_id].reshape(1, -1)
    txt = txt_feat[sample_id].reshape(1, -1)

    preds = model.predict([img, aud, txt])
    pred_class = int(np.argmax(preds))
    confidence = float(np.max(preds))

    st.markdown("---")

    st.markdown(
        f"""
        <div class="card">
            <div class="big-font">🎯 {EMOTION_LABELS[pred_class]}</div>
            <p>Confidence</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(confidence)
    st.write(f"**{confidence*100:.2f}% confidence**")

    st.markdown("### 🔎 All Emotion Probabilities")
    for i, prob in enumerate(preds[0]):
        st.write(f"**{EMOTION_LABELS[i]}**")
        st.progress(float(prob))

else:
    st.info("👈 Select a sample and click **Predict Emotion**")

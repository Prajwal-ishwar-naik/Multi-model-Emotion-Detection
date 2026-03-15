import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================
FUSION_MODEL_PATH = "models/fusion_model.keras"

IMAGE_FEAT_PATH = "data/processed/fusion/image_features.npy"
AUDIO_FEAT_PATH = "data/processed/fusion/audio_features.npy"
TEXT_FEAT_PATH  = "data/processed/fusion/text_features.npy"

# ⚠️ Put MAX labels you know (can be fewer than model outputs)
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# =========================
# LOAD MODEL
# =========================
print("🔹 Loading fusion model...")
fusion_model = load_model(FUSION_MODEL_PATH)
print("✅ Fusion model loaded")

num_classes = fusion_model.output_shape[-1]
print(f"🎯 Model predicts {num_classes} classes")

# =========================
# LOAD FEATURES
# =========================
img_feat   = np.load(IMAGE_FEAT_PATH)
audio_feat = np.load(AUDIO_FEAT_PATH)
text_feat  = np.load(TEXT_FEAT_PATH)

# =========================
# ALIGN SAMPLE SIZE
# =========================
min_len = min(len(img_feat), len(audio_feat), len(text_feat))

img_feat   = img_feat[:min_len]
audio_feat = audio_feat[:min_len]
text_feat  = text_feat[:min_len]

# =========================
# PREDICT
# =========================
preds = fusion_model.predict([img_feat, audio_feat, text_feat])
pred_classes = np.argmax(preds, axis=1)

# =========================
# DISPLAY RESULTS (SAFE)
# =========================
print("\n🎯 Predictions:")
for i in range(min(10, min_len)):
    idx = pred_classes[i]
    if idx < len(EMOTION_LABELS):
        label = EMOTION_LABELS[idx]
    else:
        label = f"Class_{idx}"

    print(f"Sample {i+1}: {label}")

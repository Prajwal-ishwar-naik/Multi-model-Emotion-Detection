import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

# Paths
IMAGE_MODEL_PATH = "models/image_model.keras"
AUDIO_MODEL_PATH = "models/audio_model.keras"
TEXT_MODEL_PATH  = "models/text_model.keras"

SAVE_DIR = "data/processed/fusion"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    print("🔹 Loading models...")

    image_model = load_model(IMAGE_MODEL_PATH, compile=False)
    audio_model = load_model(AUDIO_MODEL_PATH, compile=False)
    text_model  = load_model(TEXT_MODEL_PATH, compile=False)

    # ---------------- IMAGE FEATURES ----------------
    X_img = np.load("data/processed/image/x_train.npy")

    img_input = Input(shape=X_img.shape[1:])
    img_features = image_model(img_input, training=False)
    image_feat_model = Model(img_input, img_features)

    img_feat = image_feat_model.predict(X_img, batch_size=32)
    np.save(f"{SAVE_DIR}/image_features.npy", img_feat)

    # ---------------- AUDIO FEATURES ----------------
    X_audio = np.load("data/processed/audio/X_audio.npy")

    audio_input = Input(shape=X_audio.shape[1:])
    audio_features = audio_model(audio_input, training=False)
    audio_feat_model = Model(audio_input, audio_features)

    audio_feat = audio_feat_model.predict(X_audio, batch_size=32)
    np.save(f"{SAVE_DIR}/audio_features.npy", audio_feat)

    # ---------------- TEXT FEATURES ----------------
    X_text = np.load("data/processed/text/x_train.npy")

    text_input = Input(shape=X_text.shape[1:])
    text_features = text_model(text_input, training=False)
    text_feat_model = Model(text_input, text_features)

    text_feat = text_feat_model.predict(X_text, batch_size=32)
    np.save(f"{SAVE_DIR}/text_features.npy", text_feat)

    # ---------------- LABELS ----------------
    y = np.load("data/processed/text/y_train.npy")
    np.save(f"{SAVE_DIR}/labels.npy", y)

    print("✅ Feature extraction COMPLETE")
    print("Saved to:", SAVE_DIR)

if __name__ == "__main__":
    main()

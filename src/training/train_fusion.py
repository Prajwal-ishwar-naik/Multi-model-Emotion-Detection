import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras.utils import to_categorical

def train_fusion_model():

    print("🔹 Loading extracted fusion features...")

    img_feat   = np.load("data/processed/fusion/image_features.npy")
    audio_feat = np.load("data/processed/fusion/audio_features.npy")
    text_feat  = np.load("data/processed/fusion/text_features.npy")
    y          = np.load("data/processed/fusion/labels.npy")

    # -----------------------------
    # ALIGN SAMPLE COUNTS
    # -----------------------------
    min_len = min(
        img_feat.shape[0],
        audio_feat.shape[0],
        text_feat.shape[0],
        y.shape[0]
    )

    img_feat   = img_feat[:min_len]
    audio_feat = audio_feat[:min_len]
    text_feat  = text_feat[:min_len]
    y          = y[:min_len]

    print("✅ Aligned shapes:")
    print("Image:", img_feat.shape)
    print("Audio:", audio_feat.shape)
    print("Text :", text_feat.shape)
    print("Labels:", y.shape)

    # -----------------------------
    # FIX LABELS 🔥
    # -----------------------------
    unique_labels = np.unique(y)
    NUM_CLASSES = len(unique_labels)

    print("🧠 Detected classes:", unique_labels)
    print("🧠 Num classes:", NUM_CLASSES)

    # Map labels to 0..N-1 if needed
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])

    y = to_categorical(y, NUM_CLASSES)

    # -----------------------------
    # MODEL INPUTS
    # -----------------------------
    img_input   = Input(shape=(img_feat.shape[1],), name="image_input")
    audio_input = Input(shape=(audio_feat.shape[1],), name="audio_input")
    text_input  = Input(shape=(text_feat.shape[1],), name="text_input")

    fused = Concatenate()([img_input, audio_input, text_input])

    x = Dense(128, activation="relu")(fused)
    x = Dense(64, activation="relu")(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    fusion_model = Model(
        inputs=[img_input, audio_input, text_input],
        outputs=output
    )

    fusion_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    fusion_model.summary()

    # -----------------------------
    # TRAIN
    # -----------------------------
    print("🚀 Training fusion model...")
    fusion_model.fit(
        [img_feat, audio_feat, text_feat],
        y,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        shuffle=True
    )

    fusion_model.save("models/fusion_model.keras")
    print("✅ Fusion model trained & saved successfully!")

if __name__ == "__main__":
    train_fusion_model()

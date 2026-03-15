import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    MAX_VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
    NUM_CLASSES
)

def train_text_model():
    print("🔹 Loading preprocessed numpy data...")

    X_train = np.load("data/processed/text/x_train.npy")
    X_test  = np.load("data/processed/text/x_test.npy")
    y_train = np.load("data/processed/text/y_train.npy")
    y_test  = np.load("data/processed/text/y_test.npy")

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # =========================
    # CLASS WEIGHTS (IMPORTANT)
    # =========================
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))
    print("🧮 Class weights:", class_weights)

    # =========================
    # MODEL
    # =========================
    model = Sequential([
        Embedding(
            input_dim=MAX_VOCAB_SIZE + 1,
            output_dim=128,
            input_length=MAX_SEQUENCE_LENGTH,
            mask_zero=True   # 🔥 CRITICAL FIX
        ),

        Bidirectional(LSTM(64)),
        Dropout(0.5),

        Dense(64, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # =========================
    # CALLBACKS
    # =========================
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    # =========================
    # TRAIN
    # =========================
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[early_stop]
    )

    # =========================
    # SAVE MODEL
    # =========================
    os.makedirs("models", exist_ok=True)
    model.save("models/text_model.keras")

    print("✅ Text model trained and saved successfully!")

if __name__ == "__main__":
    train_text_model()

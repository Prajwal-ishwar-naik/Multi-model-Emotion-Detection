import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import AUDIO_PROCESSED_DIR, NUM_CLASSES


def train_audio_model():
    print("🔹 Loading preprocessed audio data...")

    X = np.load(os.path.join(AUDIO_PROCESSED_DIR, "X_audio.npy"))
    y = np.load(os.path.join(AUDIO_PROCESSED_DIR, "y_audio.npy"))

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # =========================
    # FEATURE NORMALIZATION
    # =========================
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # =========================
    # TRAIN / TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # ONE-HOT ENCODE LABELS
    # =========================
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test  = to_categorical(y_test, NUM_CLASSES)

    # =========================
    # MODEL ARCHITECTURE
    # =========================
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.5),

        Dense(256, activation='relu'),
        Dropout(0.4),

        Dense(128, activation='relu'),
        Dropout(0.3),

        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # =========================
    # CALLBACK (PREVENT OVERFITTING)
    # =========================
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # =========================
    # TRAIN
    # =========================
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop]
    )

    # =========================
    # SAVE MODEL
    # =========================
    os.makedirs("models", exist_ok=True)
    model.save("models/audio_model.keras")

    print("✅ Audio model trained and saved successfully!")


if __name__ == "__main__":
    train_audio_model()

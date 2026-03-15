import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    Flatten, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import IMAGE_PROCESSED_DIR, IMAGE_SIZE, NUM_CLASSES


def train_image_model():
    print("🔹 Loading preprocessed image data...")

    X_train = np.load(os.path.join(IMAGE_PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(IMAGE_PROCESSED_DIR, "y_train.npy"))
    X_test  = np.load(os.path.join(IMAGE_PROCESSED_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(IMAGE_PROCESSED_DIR, "y_test.npy"))

    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    # =========================
    # NORMALIZE
    # =========================
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32") / 255.0

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test  = to_categorical(y_test, NUM_CLASSES)

    # =========================
    # DATA AUGMENTATION
    # =========================
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # =========================
    # CNN MODEL (STABLE)
    # =========================
    model = Sequential([

        Conv2D(32, (3,3), activation='relu', input_shape=(*IMAGE_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # =========================
    # CALLBACKS (VERY IMPORTANT)
    # =========================
    os.makedirs("models", exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            "models/image_model.keras",
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    # =========================
    # TRAIN
    # =========================
    model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        validation_data=(X_test, y_test),
        epochs=40,
        callbacks=callbacks
    )

    print("✅ Image model trained & best model saved!")


if __name__ == "__main__":
    train_image_model()

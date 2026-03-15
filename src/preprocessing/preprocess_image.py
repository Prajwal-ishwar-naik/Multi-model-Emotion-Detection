import os
import cv2
import numpy as np
from tqdm import tqdm

from src.config import (
    IMAGE_RAW_DIR,
    IMAGE_PROCESSED_DIR,
    IMAGE_SIZE,
    EMOTION_MAP
)

def load_images(split):
    X, y = [], []

    split_dir = os.path.join(IMAGE_RAW_DIR, "fer2013", split)

    for emotion, label in EMOTION_MAP.items():
        emotion_dir = os.path.join(split_dir, emotion)

        if not os.path.exists(emotion_dir):
            print(f"⚠️ Skipping missing folder: {emotion_dir}")
            continue

        for img_name in tqdm(os.listdir(emotion_dir), desc=f"{split}-{emotion}"):
            img_path = os.path.join(emotion_dir, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, IMAGE_SIZE)
            img = img / 255.0  # normalize

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)

def main():
    print("🔹 Preprocessing image data...")

    X_train, y_train = load_images("train")
    X_test, y_test = load_images("test")

    # Add channel dimension (CNN requirement)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    os.makedirs(IMAGE_PROCESSED_DIR, exist_ok=True)

    np.save(os.path.join(IMAGE_PROCESSED_DIR, "x_train.npy"), X_train)
    np.save(os.path.join(IMAGE_PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(IMAGE_PROCESSED_DIR, "x_test.npy"), X_test)
    np.save(os.path.join(IMAGE_PROCESSED_DIR, "y_test.npy"), y_test)

    print("✅ Image preprocessing completed!")
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)

if __name__ == "__main__":
    main()

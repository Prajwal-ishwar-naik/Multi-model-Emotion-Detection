import os
import numpy as np
import librosa
from tqdm import tqdm

from src.config import AUDIO_RAW_DIR, AUDIO_PROCESSED_DIR, SAMPLE_RATE

# =========================
# RAVDESS EMOTION MAP
# =========================
# RAVDESS emotion codes:
# 01 = neutral
# 02 = calm (treated as neutral)
# 03 = happy
# 04 = sad
# 05 = angry
# 06 = fear
# 07 = disgust
# 08 = surprise

RAVDESS_EMOTION_MAP = {
    1: 6,  # neutral
    2: 6,  # calm -> neutral
    3: 3,  # happy
    4: 4,  # sad
    5: 0,  # angry
    6: 2,  # fear
    7: 1,  # disgust
    8: 5   # surprise
}

N_MFCC = 40


def extract_features(file_path):
    """Extract MFCC features from audio file"""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc = np.mean(mfcc.T, axis=0)
        return mfcc
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


def preprocess_ravdess():
    print("🔹 Preprocessing audio data...")

    X, y = [], []

    ravdess_dir = os.path.join(AUDIO_RAW_DIR, "ravdess")

    if not os.path.exists(ravdess_dir):
        print(f"❌ RAVDESS folder not found at {ravdess_dir}")
        return

    # Loop through Actor folders
    for actor in tqdm(os.listdir(ravdess_dir), desc="ravdess"):
        actor_path = os.path.join(ravdess_dir, actor)

        if not os.path.isdir(actor_path):
            continue

        for filename in os.listdir(actor_path):

            if not filename.endswith(".wav"):
                continue

            file_path = os.path.join(actor_path, filename)

            # Filename format: 03-01-05-01-02-01-12.wav
            # emotion is 3rd index
            try:
                emotion_code = int(filename.split("-")[2])
            except:
                continue

            if emotion_code not in RAVDESS_EMOTION_MAP:
                continue

            label = RAVDESS_EMOTION_MAP[emotion_code]

            features = extract_features(file_path)
            if features is None:
                continue

            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Save processed data
    os.makedirs(AUDIO_PROCESSED_DIR, exist_ok=True)

    np.save(os.path.join(AUDIO_PROCESSED_DIR, "X_audio.npy"), X)
    np.save(os.path.join(AUDIO_PROCESSED_DIR, "y_audio.npy"), y)

    print("✅ Audio preprocessing completed!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    preprocess_ravdess()

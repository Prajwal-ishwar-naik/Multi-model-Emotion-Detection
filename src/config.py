# src/config.py
import os

# =========================
# PROJECT ROOT
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# DATA DIRECTORIES
# =========================
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

IMAGE_RAW_DIR = os.path.join(RAW_DATA_DIR, "image")
AUDIO_RAW_DIR = os.path.join(RAW_DATA_DIR, "audio")
TEXT_RAW_DIR  = os.path.join(RAW_DATA_DIR, "text")

IMAGE_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, "image")
AUDIO_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, "audio")
TEXT_PROCESSED_DIR  = os.path.join(PROCESSED_DATA_DIR, "text")

os.makedirs(IMAGE_PROCESSED_DIR, exist_ok=True)
os.makedirs(AUDIO_PROCESSED_DIR, exist_ok=True)
os.makedirs(TEXT_PROCESSED_DIR, exist_ok=True)

# =========================
# IMAGE SETTINGS (FER2013)
# =========================
IMAGE_SIZE = (48, 48)
IMAGE_CHANNELS = 1

# Image emotion map (used ONLY for image training)
EMOTION_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

# =========================
# 🔥 FUSION OUTPUT LABELS (11 CLASSES)
# MUST MATCH FUSION MODEL OUTPUT
# =========================
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
    "Calm",
    "Excited",
    "Bored",
    "Confused"
]

NUM_CLASSES = len(EMOTION_LABELS)  # ✅ 11

# =========================
# AUDIO SETTINGS
# =========================
SAMPLE_RATE = 22050
N_MFCC = 40

# =========================
# TEXT SETTINGS
# =========================
MAX_TEXT_LEN = 100
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000

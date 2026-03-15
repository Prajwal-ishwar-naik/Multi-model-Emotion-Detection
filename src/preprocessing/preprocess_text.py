import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import TEXT_RAW_DIR, TEXT_PROCESSED_DIR, MAX_SEQUENCE_LENGTH, MAX_VOCAB_SIZE

def preprocess_text():
    print("🔹 Preprocessing text dataset...")

    df = pd.read_csv(f"{TEXT_RAW_DIR}/text_emotion.csv")
    df = df[['text', 'emotion']].dropna()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['emotion'])

    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])

    sequences = tokenizer.texts_to_sequences(df['text'])
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    np.save(f"{TEXT_PROCESSED_DIR}/x_train.npy", X_train)
    np.save(f"{TEXT_PROCESSED_DIR}/x_test.npy", X_test)
    np.save(f"{TEXT_PROCESSED_DIR}/y_train.npy", y_train)
    np.save(f"{TEXT_PROCESSED_DIR}/y_test.npy", y_test)

    with open(f"{TEXT_PROCESSED_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("✅ Text preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_text()

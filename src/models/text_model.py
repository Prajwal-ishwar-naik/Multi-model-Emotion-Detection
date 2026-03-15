import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def build_text_model(vocab_size, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=100),
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

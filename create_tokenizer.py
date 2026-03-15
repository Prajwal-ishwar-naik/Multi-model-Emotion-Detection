import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

texts = [
    "I am happy",
    "I am sad",
    "I feel angry",
    "I am scared",
    "I feel calm",
    "This is amazing",
    "I am feeling bad"
]

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

np.save("models/tokenizer.npy", tokenizer)

print("✅ tokenizer saved at models/tokenizer.npy")

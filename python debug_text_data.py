import numpy as np

X_train = np.load("data/processed/text/x_train.npy")
y_train = np.load("data/processed/text/y_train.npy")

print("========== DATA SHAPE ==========")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\n========== CLASS DISTRIBUTION ==========")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

print("\n========== INPUT QUALITY ==========")
print("Sample sequence (first 20 tokens):", X_train[0][:20])
print("Zero ratio (padding %):", (X_train == 0).mean())

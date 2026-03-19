# ==========================================
# GRU — MULTIWOZ SEQUENCE MODEL
# ==========================================

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Fix Pylance by mapping keras from tensorflow
Sequential = tf.keras.models.Sequential
GRU = tf.keras.layers.GRU
Dense = tf.keras.layers.Dense
Masking = tf.keras.layers.Masking
Input = tf.keras.layers.Input
to_categorical = tf.keras.utils.to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ==========================================
# LOAD MULTIWOZ FILE
# ==========================================

file_path = r"C:\Users\mansi\Downloads\SNLP MODEL\archive (2)\MultiWOZ_2.2\train\dialogues_003.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Dialogues loaded:", len(data))


# ==========================================
# CREATE SEQUENCE DATA
# ==========================================

sequences = []
labels = []
max_len = 20

for dialog in data[:300]:  # subset for faster training
    turns = len(dialog["turns"])
    seq = list(range(1, turns + 1))

    # Pad or truncate
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]

    sequences.append(seq)

    # Label dialogue complexity
    if turns < 5:
        labels.append("low")
    elif turns < 10:
        labels.append("medium")
    else:
        labels.append("high")


# Convert to numpy
X = np.array(sequences, dtype="float32")

# Normalize (helps convergence)
X = X / max_len

# Reshape for GRU [samples, timesteps, features]
X = X.reshape(len(X), max_len, 1)


# Encode labels
le = LabelEncoder()
y = to_categorical(le.fit_transform(labels))


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)


# ==========================================
# BUILD GRU MODEL
# ==========================================

model = Sequential([
    Input(shape=(max_len, 1)),
    Masking(mask_value=0),
    GRU(64),
    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# TRAIN MODEL
# ==========================================

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)


# ==========================================
# VISUALIZATION
# ==========================================

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("GRU Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("GRU Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# ==========================================
# TEST PREDICTION
# ==========================================

sample = X_test[0].reshape(1, max_len, 1)
pred = model.predict(sample)
pred_class = le.inverse_transform([np.argmax(pred)])

print("Predicted Dialogue Complexity:", pred_class[0])

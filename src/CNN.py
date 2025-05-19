import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Make sure TensorFlow / Keras is installed:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# — Load and sample 1 000 entries —
df = pd.read_csv("../train.csv").sample(n=1000, random_state=42).reset_index(drop=True)

# — Parameters —
img_size = (64, 64)
batch_size = 32
epochs = 10

# — Load images & labels —
X, y = [], []
base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
for row in df.itertuples():
    path = os.path.join(base, row.file_name)
    img = cv2.imread(path)
    if img is None:
        continue
    img = cv2.resize(img, img_size)
    img = img.astype("float32") / 255.0
    X.append(img)
    y.append(row.label)

X = np.array(X)
y = np.array(y)
print(f"Loaded {len(X)} images, shape={X.shape}")

# — Train/Test Split —
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training on {len(X_train)} samples; validating on {len(X_val)} samples")

# — One-hot encode labels —
num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat   = to_categorical(y_val, num_classes)

# — Build CNN model —
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=img_size + (3,)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# — Train —
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=epochs,
    batch_size=batch_size
)

# — Evaluate —
y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)
acc = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {acc:.2%}\n")

print("Classification Report:")
print(classification_report(y_val, y_pred, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# — Save model & history —
out_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
model.save(os.path.join(out_dir, "cnn_model.h5"))
joblib.dump(history.history, os.path.join(out_dir, "cnn_history.joblib"))
print(f"\nSaved CNN model and history to: {out_dir}")

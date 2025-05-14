import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

# Load only the first 1000 entries for quick testing
df = pd.read_csv("../train.csv").iloc[:1000]

# Constants
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

# Load images into arrays
def load_images(df):
    images = []
    labels = []
    paths = []
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    for row in tqdm(df.itertuples(), total=len(df), desc="Loading images"):
        fullpath = os.path.join(base, row.file_name)
        img = cv2.imread(fullpath)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img.astype("float32") / 255.0)  # normalize to [0,1]
        labels.append(row.label)
        paths.append(row.file_name)
    return np.array(images), np.array(labels), np.array(paths)

X, y, paths = load_images(df)
print(f"Loaded {X.shape[0]} images of shape {X.shape[1:]}")

# Train/validation split
X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
    X, y, paths, test_size=0.2, random_state=42
)
print(f"Training on {len(X_train)} samples; validating on {len(X_val)} samples")

# Build a simple CNN
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the CNN
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Evaluate on validation set
y_val_prob = model.predict(X_val).ravel()
y_val_pred = (y_val_prob >= 0.5).astype(int)
acc = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {acc:.2%}\n")
print(classification_report(y_val, y_val_pred, digits=4))

# Save the trained model
out_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
model_path = os.path.join(out_dir, "cnn_model.h5")
model.save(model_path)
print(f"Saved CNN model to: {model_path}")

# Save misclassified examples for inspection
wrong_mask = (y_val_pred != y_val)
wrong_df = pd.DataFrame({
    "file_name": paths_val[wrong_mask],
    "true_label": y_val[wrong_mask],
    "predicted_label": y_val_pred[wrong_mask]
})
wrong_csv = os.path.join(out_dir, "cnn_wrong_predictions.csv")
wrong_df.to_csv(wrong_csv, index=False)
print(f"Saved misclassified image list to: {wrong_csv}")

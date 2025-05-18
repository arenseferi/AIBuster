from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis  # Import QDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load training CSV (relative to src/)
train_df = pd.read_csv("./train.csv")

# Function to load and preprocess images
def load_images(df, col, size=(32, 32), labels=False):
    X, y = [], []
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    for row in tqdm(df.itertuples(), total=len(df), desc=f"Loading {col}"):
        relpath = getattr(row, col)
        fullpath = os.path.join(base, relpath)
        img = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, size)
        X.append(img.flatten())
        if labels:
            y.append(row.label)
    X = np.array(X)
    return (X, np.array(y)) if labels else X

# Split into 80% train / 20% validation
train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=42
)

# Load image data
X_train, y_train = load_images(train_split, "file_name", labels=True)
X_val, y_val     = load_images(val_split,   "file_name", labels=True)

print(f"Loaded data → Train: {X_train.shape}, Validation: {X_val.shape}")

# Initialize and train QDA
qda = QuadraticDiscriminantAnalysis()
print("Training QDA model...")
qda.fit(X_train, y_train)

# Evaluate on validation set
print("Evaluating on validation set...")
y_val_pred = qda.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {accuracy:.2%}\n")
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

# Save the trained model
model_path = os.path.abspath(
    os.path.join(__file__, os.pardir, os.pardir, "qda_model.joblib")
)
joblib.dump(qda, model_path)
print(f"\nTrained model saved to: {model_path}")
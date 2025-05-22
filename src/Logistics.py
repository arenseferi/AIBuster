from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load training CSV
train_df = pd.read_csv("./train.csv")  # Read CSV file into DataFrame

# Function to load and preprocess images
def load_images(df, col, size=(32, 32), labels=False):
    X, y = [], []  # Lists to hold image data and labels
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))  # Project root directory
    for row in tqdm(df.itertuples(), total=len(df), desc=f"Loading {col}"):
        relpath = getattr(row, col)  # Get file path from the specified column
        fullpath = os.path.join(base, relpath)  # Build full file path
        img = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
        if img is None:
            continue  # Skip if image can't be loaded
        img = cv2.resize(img, size)  # Resize to target size
        X.append(img.flatten())  # Flatten image to 1D array
        if labels:
            y.append(row.label)  # Append label if requested
    X = np.array(X)  # Convert list of images to NumPy array
    return (X, np.array(y)) if labels else X  # Return images and labels if needed

# Split into 80% train / 20% validation
train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=42  # Reserve 20% for validation
)

# Load image data for training and validation
X_train, y_train = load_images(train_split, "file_name", labels=True)  # Load training data
X_val, y_val     = load_images(val_split,   "file_name", labels=True)  # Load validation data

print(f"Loaded data → Train: {X_train.shape}, Validation: {X_val.shape}")  # Report shapes

# Initialize and train Logistic Regression
logreg = LogisticRegression(max_iter=1000)  # Create LR model with enough iterations
print("Training Logistic Regression model...")
logreg.fit(X_train, y_train)  # Fit model to training data

# Evaluate on validation set
print("Evaluating on validation set...")
y_val_pred = logreg.predict(X_val)  # Predict labels for validation data
accuracy = accuracy_score(y_val, y_val_pred)  # Calculate accuracy
print(f"\nValidation Accuracy: {accuracy:.2%}\n")  # Display accuracy

print("Classification Report:")
print(classification_report(y_val, y_val_pred))  # Show precision, recall, f1-score

# Save the trained model
model_path = os.path.abspath(
    os.path.join(__file__, os.pardir, os.pardir, "logreg_model.joblib")  # Path to save model
)
joblib.dump(logreg, model_path)  # Write model to disk
print(f"\nTrained model saved to: {model_path}")  # Confirm save location

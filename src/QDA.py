from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load training CSV (relative to src/)
train_df = pd.read_csv("./train.csv")  # Read the CSV file into a DataFrame

# Function to load and preprocess images
def load_images(df, col, size=(32, 32), labels=False):
    X, y = [], []  # Lists to hold flattened images and labels
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))  # Project root directory
    for row in tqdm(df.itertuples(), total=len(df), desc=f"Loading {col}"):
        relpath = getattr(row, col)  # Get the relative file path from the row
        fullpath = os.path.join(base, relpath)  # Build the full path to the image
        img = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
        if img is None:
            continue  # Skip if the image cant be loaded
        img = cv2.resize(img, size)  # Resize image to the target size
        X.append(img.flatten())  # Flatten the image into a 1D array
        if labels:
            y.append(row.label)  # Append the label if requested
    X = np.array(X)  # Convert list of images to NumPy array
    return (X, np.array(y)) if labels else X  # Return both X and y if labels=True

# Split into 80% train / 20% validation
train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=42  # Reserve 20% of data for validation
)

# Load image data for training and validation
X_train, y_train = load_images(train_split, "file_name", labels=True)  # Get training data
X_val, y_val     = load_images(val_split,   "file_name", labels=True)  # Get validation data

print(f"Loaded data → Train: {X_train.shape}, Validation: {X_val.shape}")  # Print data shapes

# Initialize and train QDA classifier
qda = QuadraticDiscriminantAnalysis()  # Create a QDA model instance
print("Training QDA model...")
qda.fit(X_train, y_train)  # Fit the model to the training data

# Evaluate on validation set
print("Evaluating on validation set...")
y_val_pred = qda.predict(X_val)  # Predict labels for validation data
accuracy = accuracy_score(y_val, y_val_pred)  # Compute accuracy score
print(f"\nValidation Accuracy: {accuracy:.2%}\n")  # Print accuracy as percentage

print("Classification Report:")
print(classification_report(y_val, y_val_pred))  # Show precision, recall, and F1-score

# Save the trained model to disk
model_path = os.path.abspath(
    os.path.join(__file__, os.pardir, os.pardir, "qda_model.joblib")  # Path for saving QDA model
)
joblib.dump(qda, model_path)  # Write model object to a .joblib file
print(f"\nTrained model saved to: {model_path}")  # Confirm save location

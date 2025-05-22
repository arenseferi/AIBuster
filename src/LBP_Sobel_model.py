# Needed imports
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def extract_features(img, lbp_points=8, lbp_radius=1):
    # Compute LBP image with given points and radius
    lbp = local_binary_pattern(img, lbp_points, lbp_radius, method="uniform")
    # Determine number of histogram bins for LBP
    n_bins = lbp_points + 2
    # Build and normalize histogram of LBP values
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(float) / (hist.sum() + 1e-7)
    # Compute horizontal and vertical gradients using Sobel
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # Compute gradient magnitude and its mean and standard deviation
    mag = np.sqrt(sobelx**2 + sobely**2)
    mean_mag = mag.mean()
    std_mag = mag.std()
    # Return combined feature vector (histogram and gradient stats)
    return np.hstack([hist, mean_mag, std_mag])

def load_features(df, size=(64, 64)):
    # lists to hold features and labels
    X, y = [], []
    # Base directory for image files
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    # Loop over each row with a progress bar
    for row in tqdm(df.itertuples(), total=len(df), desc="Extracting features"):
        fullpath = os.path.join(base, row.file_name)  # Full path to the image
        img = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        if img is None:
            continue  # Skip if the image cant be loaded
        img = cv2.resize(img, size)  # Resize to target size
        X.append(extract_features(img))  # Extract and store features
        y.append(row.label)  # Store corresponding label
    # Convert lists to NumPy arrays and return
    return np.array(X), np.array(y)

# Read and sample data for quick testing
train_df = pd.read_csv("../train.csv").sample(n=1000, random_state=42).reset_index(drop=True)

# Extract features and labels from the sampled images
X, y = load_features(train_df)
print(f"Extracted features for {X.shape[0]} images, each with {X.shape[1]} dims")  # Report extraction status

# Split features into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training on {X_train.shape[0]} samples; validating on {X_val.shape[0]} samples")  # Report split sizes

# Initialize and train the LDA classifier
lda = LinearDiscriminantAnalysis()  # Create LDA model
lda.fit(X_train, y_train)  # Train on training data

# Predict on validation set and compute accuracy
y_pred = lda.predict(X_val)  # Predict labels
acc = accuracy_score(y_val, y_pred)  # Calculate accuracy
print(f"\nValidation Accuracy: {acc:.2%}\n")  # Display accuracy

print("Classification Report:")
print(classification_report(y_val, y_pred, digits=4))  # Show precision, recall, f1-score

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))  # Display confusion matrix

# Save the trained LDA model for later use
model_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "lda_model.joblib"))
joblib.dump(lda, model_path)  # Write model to disk
print(f"\nModel saved to: {model_path}")  # Confirm save location

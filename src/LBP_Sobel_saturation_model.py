# Needed imports
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def extract_features(img_bgr):
    # Convert BGR image to grayscale for texture and edge features
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Compute LBP histogram for texture
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    n_bins = 10  # Number of histogram bins
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(float) / (hist.sum() + 1e-7)  # Normalize LBP histogram

    # Compute Sobel edge magnitude stats for sharpness
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sx, sy)  # Edge magnitude
    mean_sobel, std_sobel = mag.mean(), mag.std()  # Mean and std of edges

    # Apply high-pass filter to capture fine detail
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1)
    highpass = cv2.subtract(gray, blurred)  # Difference image
    mean_hp, std_hp = highpass.mean(), highpass.std()  # Stats of high-pass

    # Convert to HSV and compute color/lighting stats
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)  # Separate channels
    mean_h, std_h = h.mean(), h.std()  # Hue stats
    mean_s, std_s = s.mean(), s.std()  # Saturation stats
    mean_v, std_v = v.mean(), v.std()  # Value stats

    # Return concatenated feature vector
    return np.hstack([
        hist,
        mean_sobel, std_sobel,
        mean_hp, std_hp,
        mean_h, std_h,
        mean_s, std_s,
        mean_v, std_v
    ])

def load_features(df, size=(64, 64)):
    X, y, paths = [], [], []  # Lists to hold features, labels, and file names
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))  # Base directory for images
    for row in tqdm(df.itertuples(), total=len(df), desc="Extracting features"):
        fullpath = os.path.join(base, row.file_name)  # Full path to image file
        img = cv2.imread(fullpath)  # Read color image
        if img is None:
            continue  # Skip missing images
        img = cv2.resize(img, size)  # Resize to target size
        X.append(extract_features(img))  # Extract and store features
        y.append(row.label)  # Store label
        paths.append(row.file_name)  # Store file name
    return np.array(X), np.array(y), np.array(paths)  # Convert lists to arrays

# Load first 1000 entries for quick testing
df = pd.read_csv("../train.csv").iloc[:1000]  # Read CSV and take first 1000 rows
print(f"Loaded {len(df)} entries from train.csv")

# Extract features and labels
X, y, paths = load_features(df)
print(f"Extracted features for {X.shape[0]} samples × {X.shape[1]} features")

# Split into 80% train / 20% validation sets
X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
    X, y, paths, test_size=0.2, random_state=42
)
print(f"Training on {len(X_train)} samples; validating on {len(X_val)} samples")

# Standardize features to zero mean and unit variance
scaler = StandardScaler()  # Create scaler instance
X_train = scaler.fit_transform(X_train)  # Fit on train then transform
X_val = scaler.transform(X_val)  # Transform validation data

# Train LDA classifier
lda = LinearDiscriminantAnalysis()  # Create LDA model
print("Training LDA classifier...")
lda.fit(X_train, y_train)  # Fit model to training data

# Evaluate on validation set
y_pred = lda.predict(X_val)  # Predict labels for validation data
acc = accuracy_score(y_val, y_pred)  # Calculate accuracy
print(f"\nValidation Accuracy: {acc:.2%}\n")  # Print accuracy

print("Classification Report:")
print(classification_report(y_val, y_pred, digits=4))  # Show precision/recall/f1

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))  # Show confusion matrix

# Save scaler and trained model for future use
out_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))  # Output directory
joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))  # Save scaler
joblib.dump(lda,    os.path.join(out_dir, "lda_model.joblib"))  # Save LDA model
print(f"Saved scaler & LDA model to {out_dir}")

# Identify and save misclassified examples
wrong_mask = (y_pred != y_val)  # Boolean mask of wrong predictions
wrong_df = pd.DataFrame({
    "file_name": paths_val[wrong_mask],  # File names of errors
    "true_label": y_val[wrong_mask],     # Actual labels
    "predicted_label": y_pred[wrong_mask]  # Predicted labels
})
wrong_csv = os.path.join(out_dir, "wrong_predictions.csv")  # Path to CSV
wrong_df.to_csv(wrong_csv, index=False)  # Write errors to CSV
print(f"Saved misclassified image list to {wrong_csv}")

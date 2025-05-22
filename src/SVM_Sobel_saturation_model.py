from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import joblib

def extract_features(img_bgr):
    # Convert BGR image to grayscale for texture and edge features
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Compute LBP histogram for texture
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    n_bins = 10  # Number of histogram bins
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(float) / (hist.sum() + 1e-7)  # Normalize histogram

    # Compute Sobel edge magnitude stats
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sx, sy)
    mean_sobel, std_sobel = mag.mean(), mag.std()  # Mean and std of edge magnitudes

    # Apply high-pass filter for fine detail
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1)
    highpass = cv2.subtract(gray, blurred)
    mean_hp, std_hp = highpass.mean(), highpass.std()  # Mean and std of high-pass image

    # Compute HSV channel statistics for color and lighting
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_h, std_h = h.mean(), h.std()  # Hue stats
    mean_s, std_s = s.mean(), s.std()  # Saturation stats
    mean_v, std_v = v.mean(), v.std()  # Value stats

    # Return combined feature vector
    return np.hstack([
        hist,
        mean_sobel, std_sobel,
        mean_hp, std_hp,
        mean_h, std_h,
        mean_s, std_s,
        mean_v, std_v
    ])

def load_features(df, size=(64, 64)):
    X, y, paths = [], [], []  # Lists to hold features, labels, and file paths
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))  # Base directory for images
    for row in tqdm(df.itertuples(), total=len(df), desc="Extracting features"):
        fullpath = os.path.join(base, row.file_name)  # Full path to image
        img = cv2.imread(fullpath)  # Read the image
        if img is None:
            continue  # Skip if image cant be loaded
        img = cv2.resize(img, size)  # Resize to desired dimensions
        X.append(extract_features(img))  # Extract and store features
        y.append(row.label)  # Store label
        paths.append(row.file_name)  # Store file name
    return np.array(X), np.array(y), np.array(paths)  # Return arrays

# Load and limit to 1,000 samples 
df = pd.read_csv("../train.csv").iloc[:1000].reset_index(drop=True)
print(f"Loaded {len(df)} entries from train.csv")  # Report load count

# Feature extraction
X, y, paths = load_features(df)
print(f"Extracted features for {X.shape[0]} samples × {X.shape[1]} features")  # Report feature shape

# Split 80/20 into train and validation sets
X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
    X, y, paths, test_size=0.2, random_state=42
)
print(f"Training on {len(X_train)} samples; validating on {len(X_val)} samples")  # Report split sizes

# Standardize features to zero mean and unit variance 
scaler = StandardScaler()  # Create scaler
X_train = scaler.fit_transform(X_train)  # Fit on train and transform
X_val   = scaler.transform(X_val)       # Transform validation data

# Train SVM classifier with RBF kernel
svm = SVC(kernel='rbf', C=1.0, gamma='scale')  # Create SVM model
print("\nTraining SVM classifier with RBF kernel...")
svm.fit(X_train, y_train)  # Train the model

# Evaluate on validation set 
y_pred = svm.predict(X_val)  # Predict labels
acc    = accuracy_score(y_val, y_pred)  # Compute accuracy
print(f"\nValidation Accuracy: {acc:.2%}\n")  # Display accuracy

# Print classification report in a table format
cr = classification_report(y_val, y_pred, output_dict=True, digits=4)
cr_df = pd.DataFrame(cr).T  # Convert report to DataFrame
print("Classification Report:")
print(cr_df)  # Display classification metrics

# Print confusion matrix
cm = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:")
print(cm)  # Show confusion matrix

# Save scaler and SVM model for later use 
out_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))  # Output directory
joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))  # Save scaler
joblib.dump(svm,    os.path.join(out_dir, "svm_model.joblib"))  # Save SVM model
print(f"\nSaved scaler & SVM model to {out_dir}")  # Confirm save path

# Save list of misclassified images
wrong_mask = (y_pred != y_val)  # Mask for incorrect predictions
wrong_df = pd.DataFrame({
    "file_name": paths_val[wrong_mask],      # Misclassified file names
    "true_label": y_val[wrong_mask],         # Actual labels
    "predicted_label": y_pred[wrong_mask]    # Predicted labels
})
wrong_csv = os.path.join(out_dir, "wrong_predictions_svm.csv")  # CSV path
wrong_df.to_csv(wrong_csv, index=False)  # Write to CSV
print(f"Saved misclassified image list to {wrong_csv}")  # Confirm CSV save

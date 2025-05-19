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
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # LBP texture
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    n_bins = 10
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(float) / (hist.sum() + 1e-7)

    # Sobel edge magnitude
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sx, sy)
    mean_sobel, std_sobel = mag.mean(), mag.std()

    # High-pass filter for fine detail
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1)
    highpass = cv2.subtract(gray, blurred)
    mean_hp, std_hp = highpass.mean(), highpass.std()

    # HSV color & lighting stats
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_h, std_h = h.mean(), h.std()
    mean_s, std_s = s.mean(), s.std()
    mean_v, std_v = v.mean(), v.std()

    return np.hstack([
        hist,
        mean_sobel, std_sobel,
        mean_hp, std_hp,
        mean_h, std_h,
        mean_s, std_s,
        mean_v, std_v
    ])

def load_features(df, size=(64, 64)):
    X, y, paths = [], [], []
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    for row in tqdm(df.itertuples(), total=len(df), desc="Extracting features"):
        fullpath = os.path.join(base, row.file_name)
        img = cv2.imread(fullpath)
        if img is None:
            continue
        img = cv2.resize(img, size)
        X.append(extract_features(img))
        y.append(row.label)
        paths.append(row.file_name)
    return np.array(X), np.array(y), np.array(paths)

# — Load and limit to 1 000 samples —
df = pd.read_csv("../train.csv").iloc[:1000].reset_index(drop=True)
print(f"Loaded {len(df)} entries from train.csv")

# — Feature extraction —
X, y, paths = load_features(df)
print(f"Extracted features for {X.shape[0]} samples × {X.shape[1]} features")

# — Split 80/20 —
X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
    X, y, paths, test_size=0.2, random_state=42
)
print(f"Training on {len(X_train)} samples; validating on {len(X_val)} samples")

# — Standardize —
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# — Train SVM —
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False)
print("\nTraining SVM classifier with RBF kernel...")
svm.fit(X_train, y_train)

# — Evaluate —
y_pred = svm.predict(X_val)
acc    = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {acc:.2%}\n")

# Tabular classification report
cr = classification_report(y_val, y_pred, output_dict=True, digits=4)
cr_df = pd.DataFrame(cr).T
print("Classification Report:")
print(cr_df)

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:")
print(cm)

# — Save models & scaler —
out_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
joblib.dump(svm,    os.path.join(out_dir, "svm_model.joblib"))
print(f"\nSaved scaler & SVM model to {out_dir}")

# — Save misclassified list —
wrong_mask = (y_pred != y_val)
wrong_df = pd.DataFrame({
    "file_name": paths_val[wrong_mask],
    "true_label": y_val[wrong_mask],
    "predicted_label": y_pred[wrong_mask]
})
wrong_csv = os.path.join(out_dir, "wrong_predictions_svm.csv")
wrong_df.to_csv(wrong_csv, index=False)
print(f"Saved misclassified image list to {wrong_csv}")

from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
    mag = np.hypot(sx, sy)
    mean_sobel, std_sobel = mag.mean(), mag.std()

    # Apply high-pass filter to capture fine detail
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1)
    highpass = cv2.subtract(gray, blurred)
    mean_hp, std_hp = highpass.mean(), highpass.std()

    # Convert to HSV and compute color/lighting stats
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_h, std_h = h.mean(), h.std()
    mean_s, std_s = s.mean(), s.std()
    mean_v, std_v = v.mean(), v.std()

    # Return concatenated feature vector
    return np.hstack([
        hist,
        mean_sobel, std_sobel,
        mean_hp, std_hp,
        mean_h, std_h,
        mean_s, std_s,
        mean_v, std_v
    ])

def train(n_samples, progress_cb):
    # Base directory for project root (where train.csv lives)
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

    # Load full CSV and sample n entries (with replacement if needed)
    csv_path = os.path.join(base, "train.csv")
    df_all   = pd.read_csv(csv_path)
    total    = len(df_all)

    if n_samples > total:
        progress_cb(0, n_samples,
                    f"Requested {n_samples} > {total} available; sampling with replacement")
        df = df_all.sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)
    else:
        df = df_all.sample(n=n_samples, replace=False, random_state=42).reset_index(drop=True)
        progress_cb(0, n_samples, f"Sampled {n_samples} entries")

    # Feature extraction
    X, y = [], []
    for i, row in enumerate(df.itertuples(), start=1):
        fullpath = os.path.join(base, row.file_name)
        img = cv2.imread(fullpath)
        if img is None:
            continue  # Skip if the image can't be loaded
        img = cv2.resize(img, (64, 64))
        feats = extract_features(img)
        X.append(feats)
        y.append(row.label)

        # report progress every 10%
        if i % max(1, n_samples // 10) == 0:
            progress_cb(i, n_samples, f"Extracted features {i}/{n_samples}")

    X = np.array(X)
    y = np.array(y)
    progress_cb(len(X), n_samples, f"Extracted {len(X)} feature vectors")

    # Split into 80% train / 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    progress_cb(0, 1, f"Training on {len(X_train)} samples; validating on {len(X_val)} samples")

    # Standardize features to zero mean and unit variance
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)

    # Train LDA classifier
    lda = LinearDiscriminantAnalysis()
    progress_cb(0, 1, "Training LDA classifier")
    lda.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = lda.predict(X_val)
    acc    = accuracy_score(y_val, y_pred)
    progress_cb(1, 1, f"Validation Accuracy: {acc:.2%}")

    # Save scaler and trained model for later use
    out_dir = os.path.join(base, "trained_models")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, "lbp2_scaler.joblib"))
    joblib.dump(lda,    os.path.join(out_dir, "lbp2_model.joblib"))
    progress_cb(1, 1, f"Saved scaler & LDA model to: {out_dir}")

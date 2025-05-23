from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
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
        img = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # skip if the image can't be loaded
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

    # Split 80/20 into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    progress_cb(0, 1, f"Training on {len(X_train)} samples; validating on {len(X_val)} samples")

    # Initialize and train the LDA classifier
    lda = LinearDiscriminantAnalysis()
    progress_cb(0, 1, "Training LDA classifier")
    lda.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = lda.predict(X_val)
    acc    = accuracy_score(y_val, y_pred)
    progress_cb(1, 1, f"Validation Accuracy: {acc:.2%}")

    # Save the trained LDA model for later use
    out_dir = os.path.join(base, "trained_models")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(lda, os.path.join(out_dir, "lbp1_model.joblib"))
    progress_cb(1, 1, f"Saved LDA model to: {out_dir}")

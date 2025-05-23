import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def extract_features(img_bgr):
    # Convert BGR image to grayscale for feature extraction
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Compute LBP histogram for texture
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    n_bins = 10  # Number of bins for LBP histogram
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(float) / (hist.sum() + 1e-7)  # Normalize histogram

    # Compute Sobel edge magnitude stats for sharpness
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sx, sy)
    mean_sobel, std_sobel = mag.mean(), mag.std()  # Mean and std of edges

    # Apply high-pass filter to capture fine detail
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1)
    highpass = cv2.subtract(gray, blurred)
    mean_hp, std_hp = highpass.mean(), highpass.std()  # Stats of high pass result

    # Convert to HSV and compute color channel stats
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

def load_features(df, size=(64,64), progress_cb=None):
    # Prepare lists to hold features and labels
    X, y = [], []
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))  # Base image directory
    total = len(df)
    step = max(1, total // 10)
    for idx, row in enumerate(df.itertuples(), start=1):
        fullpath = os.path.join(base, row.file_name)  # Full path to image
        img = cv2.imread(fullpath)  # Read the image
        if img is None:
            continue  # Skip if image cant be loaded
        img = cv2.resize(img, size)  # Resize to target dimensions
        X.append(extract_features(img))  # Extract and store features
        y.append(row.label)  # Store corresponding label
        if progress_cb is not None and (idx % step == 0 or idx == total):
            progress_cb(idx, total, f"Extracted features {idx}/{total}")
    return np.array(X), np.array(y)  # Return arrays of features and labels

def visualize(n_samples, progress_cb):
    """
    Sample n images, extract features, run PCA,
    save explained‐variance CSV, scree plot, and 2D scatter.
    """
    # Load and sample n entries from train.csv for quick testing
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    csv_path = os.path.join(base, "train.csv")
    df = pd.read_csv(csv_path).sample(n=n_samples, random_state=42).reset_index(drop=True)
    progress_cb(0, n_samples, f"Sampled {n_samples} entries")

    # Extract features and labels
    X, y = load_features(df, size=(64,64), progress_cb=progress_cb)

    # Fit PCA to all features and get explained variance ratios
    pca_full = PCA(n_components=min(X.shape[1], X.shape[0]))  # Limit components
    pca_full.fit(X)  # Compute PCA
    explained = pca_full.explained_variance_ratio_  # Variance explained by each component
    for i, var in enumerate(explained[:10], start=1):
        progress_cb(i, 10, f"PC{i}: {var:.4f}")

    # Save explained variance ratios to CSV
    out_dir = base  # Output directory
    np.savetxt(os.path.join(out_dir, "pca_explained_variance.csv"), explained, delimiter=",")
    progress_cb(10, 10, f"Saved explained variance ratios to {out_dir}/pca_explained_variance.csv")

    # Create and save scree plot of explained variance
    plt.figure(figsize=(8,5))  # New figure
    plt.plot(np.arange(1, len(explained)+1), explained, marker='o')  # Plot variance ratio
    plt.xlabel("Principal Component")  # X-axis label
    plt.ylabel("Explained Variance Ratio")  # Y-axis label
    plt.title("PCA Scree Plot")  # Title
    plt.tight_layout()  # Adjust layout
    plt.savefig(os.path.join(out_dir, "pca_scree_plot.png"))  # Save figure
    progress_cb(10, 10, f"Scree plot saved to {out_dir}/pca_scree_plot.png")

    # Define label mapping for scatter plot
    label_names = {0: "Real", 1: "AI"}  # Map numeric labels to names

    # Compute 2D PCA projection for visualization
    pca2 = PCA(n_components=2, random_state=42)  # Two principal components
    X2 = pca2.fit_transform(X)  # Transform feature set

    # Plot and save 2D PCA scatter
    plt.figure(figsize=(8,6))  # New figure
    for lbl in np.unique(y):
        mask = (y == lbl)  # Select points of current label
        plt.scatter(
            X2[mask, 0], X2[mask, 1],  # Plot PC1 vs PC2
            label=label_names[lbl],  # Use mapped label name
            alpha=0.6  # Semi-transparent points
        )
    plt.xlabel("PC1")  # X-axis label
    plt.ylabel("PC2")  # Y-axis label
    plt.title("PCA 2D Projection of All Samples")  # Title
    plt.legend(title="True class")  # Legend with title
    plt.tight_layout()  # Adjust layout
    plt.savefig(os.path.join(out_dir, "pca_2d_scatter.png"))  # Save figure
    progress_cb(10, 10, f"2D PCA scatter saved to {out_dir}/pca_2d_scatter.png")

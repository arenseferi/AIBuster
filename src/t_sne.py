import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time

def visualize(n_samples, progress_cb):
    # Base directory for project root (where train.csv lives)
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

    # Load and sample n entries from the CSV for quick testing
    df = (
        pd.read_csv(os.path.join(base, "train.csv"))
          .sample(n=n_samples, random_state=42)
          .reset_index(drop=True)
    )
    progress_cb(0, n_samples, f"Sampled {n_samples} entries")

    # Extract features and labels
    X, y = [], []
    for i, row in enumerate(df.itertuples(), start=1):
        fullpath = os.path.join(base, row.file_name)  # Full path to image
        img = cv2.imread(fullpath)                    # Read color image
        if img is None:
            continue  # Skip missing images
        img = cv2.resize(img, (64,64))                # Resize to target size

        # Convert BGR image to grayscale for feature extraction
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute LBP histogram for texture
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        n_bins = 10  # Number of bins for LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float) / (hist.sum() + 1e-7)  # Normalize histogram

        # Compute Sobel edge magnitude stats for sharpness
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(sx, sy)
        mean_sobel, std_sobel = mag.mean(), mag.std()  # Mean and std of edge magnitudes

        # Apply high-pass filter to capture fine detail
        blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1)
        highpass = cv2.subtract(gray, blurred)
        mean_hp, std_hp = highpass.mean(), highpass.std()  # Mean and std of high-pass image

        # Convert to HSV and compute color channel stats
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)  # Split into H, S, V channels
        mean_h, std_h = h.mean(), h.std()  # Hue stats
        mean_s, std_s = s.mean(), s.std()  # Saturation stats
        mean_v, std_v = v.mean(), v.std()  # Value stats

        # Return concatenated feature vector
        feats = np.hstack([
            hist,
            mean_sobel, std_sobel,
            mean_hp, std_hp,
            mean_h, std_h,
            mean_s, std_s,
            mean_v, std_v
        ])
        X.append(feats)
        y.append(row.label)

        # report progress every 10%
        if i % max(1, n_samples // 10) == 0:
            progress_cb(i, n_samples, f"Extracted features {i}/{n_samples}")

    X = np.array(X)
    y = np.array(y)
    progress_cb(len(X), n_samples, f"Extracted features for {X.shape[0]} samples × {X.shape[1]} features")

    # Run t-SNE for 2D embedding
    progress_cb(0, 1, "Running t-SNE (wait)…")
    start = time.time()  # Start timer
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)  # Configure t-SNE
    X2 = tsne.fit_transform(X)  # Compute 2D embedding
    elapsed = time.time() - start
    progress_cb(1, 1, f"t-SNE done in {elapsed:.1f}s")

    # Save embedding with labels to CSV
    out_dir = base
    embed_df = pd.DataFrame({
        "Dim1": X2[:,0],  # First t-SNE dimension
        "Dim2": X2[:,1],  # Second t-SNE dimension
        "label": y        # True class labels
    })
    embed_df.to_csv(os.path.join(out_dir, "tsne_embedding.csv"), index=False)  # Save CSV
    progress_cb(1, 1, f"Saved t-SNE embedding to {out_dir}/tsne_embedding.csv")

    # Plot and save t-SNE scatter
    label_names = {0: "Real", 1: "AI"}  # Map numeric labels to names
    plt.figure(figsize=(8,6))  # New figure
    for lbl in np.unique(y):
        mask = (y == lbl)  # Mask for current label
        plt.scatter(
            X2[mask,0], X2[mask,1],  # Plot points
            label=label_names[lbl],  # Use label name
            alpha=0.6  # Semi-transparent
        )
    plt.xlabel("t-SNE Dim 1")  # X-axis label
    plt.ylabel("t-SNE Dim 2")  # Y-axis label
    plt.title("t-SNE Visualization of All Samples")  # Title
    plt.legend(title="True class")  # Legend with title
    plt.tight_layout()  # Adjust layout
    plt.savefig(os.path.join(out_dir, "tsne_scatter.png"))  # Save figure
    progress_cb(1, 1, f"Saved t-SNE scatter to {out_dir}/tsne_scatter.png")

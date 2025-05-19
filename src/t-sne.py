import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time

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

    # High-pass filter
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1)
    highpass = cv2.subtract(gray, blurred)
    mean_hp, std_hp = highpass.mean(), highpass.std()

    # HSV stats
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

def load_features(df, size=(64,64)):
    X, y = [], []
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    for row in tqdm(df.itertuples(), total=len(df), desc="Extracting features"):
        fullpath = os.path.join(base, row.file_name)
        img = cv2.imread(fullpath)
        if img is None:
            continue
        img = cv2.resize(img, size)
        X.append(extract_features(img))
        y.append(row.label)
    return np.array(X), np.array(y)

# Load and sample 1000 entries
df = pd.read_csv("../train.csv").sample(n=1000, random_state=42).reset_index(drop=True)

# Extract features
X, y = load_features(df)
print(f"Extracted features for {X.shape[0]} samples × {X.shape[1]} features")

# Run t-SNE
print("Running t-SNE (this may take a minute)…")
start = time.time()
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X2 = tsne.fit_transform(X)
print(f"t-SNE done in {time.time() - start:.1f}s")

# Save embedding + labels
out_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
embed_df = pd.DataFrame({
    "PC1": X2[:,0],
    "PC2": X2[:,1],
    "label": y
})
embed_df.to_csv(os.path.join(out_dir, "tsne_embedding.csv"), index=False)
print(f"Saved t-SNE embedding to {out_dir}/tsne_embedding.csv")

# Plot
label_names = {0: "Real", 1: "AI"}
plt.figure(figsize=(8,6))
for lbl in np.unique(y):
    mask = (y == lbl)
    plt.scatter(
        X2[mask,0],
        X2[mask,1],
        label=label_names[lbl],
        alpha=0.6
    )
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.title("t-SNE Visualization of All Samples")
plt.legend(title="True class")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "tsne_scatter.png"))
print(f"Saved t-SNE scatter to {out_dir}/tsne_scatter.png")

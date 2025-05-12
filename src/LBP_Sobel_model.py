from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt

# Load all training data paths and labels
train_df = pd.read_csv("../train.csv")

# Compute texture (LBP) and edge (Sobel) features for a single image
def extract_features(img, lbp_points=8, lbp_radius=1):
    lbp = local_binary_pattern(img, lbp_points, lbp_radius, method="uniform")
    n_bins = lbp_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(float) / (hist.sum() + 1e-7)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    mean_mag = mag.mean()
    std_mag = mag.std()

    return np.hstack([hist, mean_mag, std_mag])

# Iterate over every image, resize, extract features, and gather labels
def load_features(df, size=(64, 64)):
    X, y = [], []
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    for row in tqdm(df.itertuples(), total=len(df), desc="Extracting features"):
        fullpath = os.path.join(base, row.file_name)
        img = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, size)
        X.append(extract_features(img))
        y.append(row.label)
    return np.array(X), np.array(y)

# Build feature matrix and label vector for all data
X, y = load_features(train_df)
print(f"Extracted features for {X.shape[0]} images, each with {X.shape[1]} dims")

# Hold out 20% of the data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training on {X_train.shape[0]} samples; validating on {X_val.shape[0]} samples")

# Train an LDA classifier on the training split
lda = LinearDiscriminantAnalysis()
print("\nFitting LDA model...")
lda.fit(X_train, y_train)

# Evaluate on the validation split
print("Evaluating on validation set...")
y_pred = lda.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {acc:.2%}\n")
print("Classification Report:")
print(classification_report(y_val, y_pred, digits=4))

# Persist the trained model for later use
model_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "lda_model.joblib"))
joblib.dump(lda, model_path)
print(f"\nModel saved to: {model_path}")

# Project validation features into 2D with PCA and plot Real vs. AI clusters
pca = PCA(n_components=2, random_state=42)
X2d = pca.fit_transform(X_val)

label_names = {0: "Real", 1: "AI"}
plt.figure(figsize=(8, 6))
for cls in np.unique(y_val):
    mask = y_val == cls
    plt.scatter(X2d[mask, 0], X2d[mask, 1], label=label_names[cls], alpha=0.6)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of Validation Features (LBP + Sobel)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the scatter plot to file for review
plt.savefig("../lda_pca_visualization.png")
print("PCA visualization saved to ../lda_pca_visualization.png")

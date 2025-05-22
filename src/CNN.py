# src/CNN.py

# Needed imports
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# to make sure TensorFlow/Keras is installed
from tensorflow.keras.models import Sequential  # this to build a sequential neural network
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # CNN building blocks
from tensorflow.keras.utils import to_categorical  # to turn labels into one-hot vectors
import tensorflow as tf  # for callbacks

def train(n_samples, progress_cb):
    # Base directory for project root (where train.csv lives)
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

    # Load and sample n entries from the CSV for quick experimentation
    df = (
        pd.read_csv(os.path.join(base, "train.csv"))
          .sample(n=n_samples, random_state=42)
          .reset_index(drop=True)
    )
    progress_cb(0, n_samples, f"Sampled {n_samples} entries")

    # Parameters for training
    img_size = (64, 64)  # target image size
    batch_size = 32      # number of samples per gradient update
    epochs = 10          # number of times to iterate over the dataset

    # Load images and labels into arrays
    X, y = [], []
    for i, row in enumerate(df.itertuples(), start=1):
        path = os.path.join(base, row.file_name)  # full path to the image file
        img = cv2.imread(path)                    # read the image from disk
        if img is None:
            continue  # skip if the image can't be loaded
        img = cv2.resize(img, img_size)           # resize image to the wanted size
        img = img.astype("float32") / 255.0       # normalize pixel values to [0,1]
        X.append(img)                             # add image to list
        y.append(row.label)                       # add corresponding label

        # report progress every 10%
        if i % max(1, n_samples // 10) == 0:
            progress_cb(i, n_samples, f"Loaded {i}/{n_samples} images")

    X = np.array(X)  # convert list of images to NumPy array
    y = np.array(y)
    progress_cb(len(X), n_samples, f"Loaded {len(X)} images in total")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    progress_cb(0, epochs, f"Training on {len(X_train)} samples; validating on {len(X_val)} samples")

    # One-hot encode labels for categorical crossentropy loss
    num_classes = len(np.unique(y))                       # find how many unique classes there are
    y_train_cat = to_categorical(y_train, num_classes)   # turn labels into one-hot vectors
    y_val_cat   = to_categorical(y_val, num_classes)

    # Build a simple CNN
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=img_size + (3,)),  # first conv layer
        MaxPooling2D(),  # downsample feature maps
        Conv2D(64, (3,3), activation='relu'),  # second conv layer
        MaxPooling2D(),  # downsample again
        Conv2D(128, (3,3), activation='relu'),  # third conv layer
        MaxPooling2D(),  # final downsampling
        Flatten(),       # flatten 2D features to 1D
        Dense(128, activation='relu'),  # fully connected layer
        Dropout(0.5),    # randomly drop units to prevent overfitting
        Dense(num_classes, activation='softmax')  # output layer with class probabilities
    ])
    model.compile(
        optimizer='adam',               # use the Adam optimizer
        loss='categorical_crossentropy',  # loss for multi-class classification
        metrics=['accuracy']            # track accuracy during training
    )

    # Train the CNN on the training data
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: progress_cb(
                    epoch+1, epochs,
                    f"Epoch {epoch+1}/{epochs} — loss: {logs['loss']:.4f}, val_acc: {logs['val_accuracy']:.2%}"
                )
            )
        ]
    )

    # Evaluate model performance on validation set
    y_pred_prob = model.predict(X_val)       # get predicted probabilities
    y_pred = np.argmax(y_pred_prob, axis=1)  # convert probabilities to class labels
    acc = accuracy_score(y_val, y_pred)      # compute accuracy
    progress_cb(epochs, epochs, f"Validation Accuracy: {acc:.2%}")

    # Save the trained model and training history for later use
    out_dir = os.path.join(base, "trained_models")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "cnn_model.h5"))  # save the CNN model
    joblib.dump(history.history, os.path.join(out_dir, "cnn_history.joblib"))  # save training history
    progress_cb(epochs, epochs, f"Saved CNN model and history to: {out_dir}")

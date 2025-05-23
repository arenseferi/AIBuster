AI Buster
AI Buster is a lightweight GUI application that lets you:
Train different image classification models (CNN, SVM, LBP variants)
Test trained models on individual images
Visualize feature distributions using PCA and t-SNE

Note: the very first time you launch the app it may take a minute or two to initialize all libraries and load resources. Please be patient.

Features

Train Models
Choose from four methods (CNN, SVM, LBP1, LBP2), enter how many images to use, and watch progress in real time.

Test Models
Load any previously saved model and run it on a single image. Results are printed to the window.

Visualize PCA
Sample a set of images, extract features, run PCA, and view both the scree plot and a 2D scatter of your data.

Visualize t-SNE
Sample a set of images, run t-SNE on the extracted features, and view the resulting scatter plot.

Requirements

Python 3.7 or higher
FreeSimpleGUI
OpenCV (cv2)
NumPy
pandas
scikit-learn
scikit-image
matplotlib
tqdm
joblib

You can install most dependencies with the following command:

pip install numpy pandas opencv-python scikit-learn scikit-image matplotlib tqdm joblib FreeSimpleGUI


Getting Started:

Clone or download this repository.

Be sure to have a file named train.csv at the project root that lists images file paths and labels. Download the zip file of images and extract inside the root, and only keep the train_data and train.csv


Run the GUI by executing:

python src/main_gui.py

In the app window, choose Train Models, Test Models, Visualize PCA, or Visualize t-SNE, then follow the on screen prompts.
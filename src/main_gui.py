# src/main_gui.py

import os
import threading
import cv2
import joblib
import numpy as np
import FreeSimpleGUI as sg
import sys


# --- IMPORT YOUR REAL TRAINING FUNCTIONS HERE ---
from CNN import train as train_cnn

# Stub trainers for your other models until you wire them in:
def train_lda(n, progress_cb):
    for i in range(n):
        threading.Event().wait(0.005)
        progress_cb(i+1, n, f"LDA: processed {i+1}/{n}")
    progress_cb(n, n, "LDA training complete. Model saved.")

def train_logreg(n, progress_cb):
    for i in range(n):
        threading.Event().wait(0.007)
        progress_cb(i+1, n, f"LogReg: processed {i+1}/{n}")
    progress_cb(n, n, "LogReg training complete. Model saved.")

def train_qda(n, progress_cb):
    for i in range(n):
        threading.Event().wait(0.006)
        progress_cb(i+1, n, f"QDA: processed {i+1}/{n}")
    progress_cb(n, n, "QDA training complete. Model saved.")

def train_svm(n, progress_cb):
    for i in range(n):
        threading.Event().wait(0.008)
        progress_cb(i+1, n, f"SVM: processed {i+1}/{n}")
    progress_cb(n, n, "SVM training complete. Model saved.")


MODEL_FUNCS = {
    "CNN":       train_cnn,
    "LDA":       train_lda,
    "Logistic":  train_logreg,
    "QDA":       train_qda,
    "SVM":       train_svm,
}


def long_running_train(model_name, n_samples, window):
    """Invoke the selected trainer in a background thread."""
    def progress_cb(done, total, msg):
        pct = int(done/total*100)
        window['PROGRESS'].update_bar(pct)
        window['OUTPUT'].print(msg)

    MODEL_FUNCS[model_name](n_samples, progress_cb)
    window['TRAIN'].update(disabled=False)


def train_window():
    """Show the Train Models window."""
    layout = [
        [sg.Text('Select model to train:'), 
         sg.Combo(list(MODEL_FUNCS), default_value='CNN', key='MODEL')],
        [sg.Text('Number of images:'), 
         sg.InputText('1000', key='NUM')],
        [sg.Button('Train', key='TRAIN'), sg.Button('Back')],
        [sg.ProgressBar(100, orientation='h', size=(40, 20), key='PROGRESS')],
        [sg.Multiline(size=(60,10), key='OUTPUT', autoscroll=True, disabled=True)]
    ]
    win = sg.Window('Train Models', layout, modal=True)

    while True:
        event, values = win.read(timeout=100)
        if event == sg.WIN_CLOSED:
            win.close()
            sys.exit(0)
        if event == 'Back':
            break

        if event == 'TRAIN':
            win['TRAIN'].update(disabled=True)
            win['PROGRESS'].update_bar(0)
            win['OUTPUT'].update('')
            model = values['MODEL']
            try:
                n = int(values['NUM'])
            except ValueError:
                win['OUTPUT'].print("Enter a valid integer number of images.")
                win['TRAIN'].update(disabled=False)
                continue

            threading.Thread(
                target=long_running_train,
                args=(model, n, win),
                daemon=True
            ).start()

    win.close()


def test_window():
    """Show the Test Models window."""
    # list all .h5 and .joblib files in trained_models
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    tm_dir = os.path.join(base, 'trained_models')
    model_files = [
        f for f in os.listdir(tm_dir)
        if f.endswith('.h5') or f.endswith('_model.joblib')
    ]
    layout = [
        [sg.Text('Select trained model:'), 
         sg.Combo(model_files, key='MODELFILE')],
        [sg.Text('Select image to test:'), 
        sg.InputText(key='IMGPATH'), 
        sg.FileBrowse(file_types=(("Image Files", "*.png *.jpg *.jpeg"),), initial_folder=os.getcwd())],
        [sg.Button('Run', key='RUN'), sg.Button('Back')],
        [sg.Multiline(size=(60,10), key='OUT', autoscroll=True, disabled=True)]
    ]
    win = sg.Window('Test Models', layout, modal=True)

    while True:
        event, values = win.read()
        if event == sg.WIN_CLOSED:
            win.close()
            sys.exit(0)
        if event == 'Back':
            break

        if event == 'RUN':
            modelfile = values['MODELFILE']
            imgpath   = values['IMGPATH']
            if not modelfile or not imgpath:
                win['OUT'].print("Please choose both a model and an image.")
                continue

            full_model_path = os.path.join(tm_dir, modelfile)
            win['OUT'].print(f"Loading model {modelfile}…")
            # **TODO**: adapt to how each model is loaded and predict
            if modelfile.endswith('.h5'):
                from tensorflow.keras.models import load_model
                mdl = load_model(full_model_path)
            else:
                mdl = joblib.load(full_model_path)

            win['OUT'].print("Reading image…")
            img = cv2.imread(imgpath)
            # **TODO**: preprocess just like in training (resize/normalize/extract features)
            img = cv2.resize(img, (64,64)).astype('float32')/255.0
            x = np.expand_dims(img, 0)  # batch of 1

            win['OUT'].print("Running prediction…")

            # CNN case
            if modelfile.endswith('.h5'):
                probs = mdl.predict(x)[0]            # [real_prob, ai_prob]
                label_names = {0: "Real", 1: "AI"}
                winner = int(np.argmax(probs))       # 0 or 1
                confidence = probs[winner] * 100

                win['OUT'].print("Class probabilities:")
                win['OUT'].print(f" Real photo : {probs[0]*100:5.1f}%")
                win['OUT'].print(f" AI generated: {probs[1]*100:5.1f}%")
                win['OUT'].print(f"Verdict: {label_names[winner]} "
                                f"(confidence {confidence:.1f}%)")

            # scikit-learn case
            else:
                if hasattr(mdl, 'predict_proba'):
                    proba = mdl.predict_proba([x.flatten()])[0]
                    classes = mdl.classes_
                    winner = int(np.argmax(proba))
                    confidence = proba[winner] * 100

                    win['OUT'].print("📊 Class probabilities:")
                    for cls, p in zip(classes, proba):
                        win['OUT'].print(f"   • {cls:12s}: {p*100:5.1f}%")
                    win['OUT'].print(f"🎯 Verdict: **{classes[winner]}** "
                                    f"(confidence {confidence:.1f}%)")
                else:
                    label = mdl.predict([x.flatten()])[0]
                    win['OUT'].print(f"🎯 Predicted label: **{label}**")


    win.close()


def main():
    sg.theme('LightBlue2')
    layout = [
        [sg.Text('Welcome to AI Buster', font=('Helvetica', 16))],
        [sg.VerticalSeparator(), sg.Button('Train Models'), sg.Button('Test Models')],
    ]
    win = sg.Window('AI Buster', layout)

    while True:
        event, _ = win.read()
        if event in (sg.WIN_CLOSED, None):
            win.close()
            sys.exit(0)
        elif event == 'Train Models':
            train_window()
        elif event == 'Test Models':
            test_window()


    win.close()


if __name__ == '__main__':
    main()

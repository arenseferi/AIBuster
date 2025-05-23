import os
import sys
import threading
import cv2
import joblib
import numpy as np
import FreeSimpleGUI as sg
from PIL import Image
import io

# import training and visualization functions
from CNN import train as train_cnn
from SVM_Sobel_saturation_model import train as train_svm, extract_features as svm_extract_features
from LBP_Sobel_model import train as train_lbp1, extract_features as lbp1_extract_features
from LBP_Sobel_saturation_model import train as train_lbp2, extract_features as lbp2_extract_features
from PCA import visualize as visualize_pca
from t_sne import visualize as visualize_tsne

# map model names to functions
MODEL_FUNCS = {
    "CNN": train_cnn,
    "SVM": train_svm,
    "LBP1 (no saturation)": train_lbp1,
    "LBP2 (with saturation)": train_lbp2,
}

# run training in background thread
def long_running_train(model_name, n_samples, window):
    def progress_cb(done, total, msg):
        pct = int(done/total*100)
        window['PROGRESS'].update_bar(pct)
        window['OUTPUT'].print(msg)

    MODEL_FUNCS[model_name](n_samples, progress_cb)
    window['TRAIN'].update(disabled=False)

# show training dialog
def train_window():
    layout = [
        [sg.Text('Select model to train:'), sg.Combo(list(MODEL_FUNCS), default_value='CNN', key='MODEL')],
        [sg.Text('Number of images:'), sg.InputText('1000', key='NUM')],
        [sg.Button('Train', key='TRAIN'), sg.Button('Back')],
        [sg.ProgressBar(100, orientation='h', size=(40,20), key='PROGRESS')],
        [sg.Multiline(size=(60,10), key='OUTPUT', autoscroll=True, disabled=True)],
    ]
    win = sg.Window('Train Models', layout, modal=True)

    while True:
        event, values = win.read(timeout=100)
        if event in (sg.WIN_CLOSED, 'Back'):
            break

        if event == 'TRAIN':
            win['TRAIN'].update(disabled=True)
            win['PROGRESS'].update_bar(0)
            win['OUTPUT'].update('')
            try:
                n = int(values['NUM'])
            except ValueError:
                win['OUTPUT'].print("Enter a valid integer for number of images.")
                win['TRAIN'].update(disabled=False)
                continue

            threading.Thread(
                target=long_running_train,
                args=(values['MODEL'], n, win),
                daemon=True
            ).start()

    win.close()

# load model, run prediction, send output

def do_test(modelfile, imgpath, window):
    # Background job for loading a model, running a prediction, and streaming output back into the Test Models window
    
    tm_dir = os.path.join(
        os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)),
        'trained_models'
    )
    full_model_path = os.path.join(tm_dir, modelfile)

    def print_cb(msg):
        window['OUT'].print(msg)
        window.refresh()

    print_cb(f"Loading model {modelfile}…")

    # load model and scaler if needed
    if modelfile.endswith('.h5'):
        from tensorflow.keras.models import load_model
        mdl = load_model(full_model_path)
        scaler = None
    elif modelfile == 'svm_model.joblib':
        scaler = joblib.load(os.path.join(tm_dir, 'svm_scaler.joblib'))
        mdl    = joblib.load(full_model_path)
    elif modelfile == 'lbp1_model.joblib':
        scaler = None
        mdl    = joblib.load(full_model_path)
    elif modelfile == 'lbp2_model.joblib':
        scaler = joblib.load(os.path.join(tm_dir, 'lbp2_scaler.joblib'))
        mdl    = joblib.load(full_model_path)
    else:
        scaler = None
        mdl    = joblib.load(full_model_path)

    print_cb("Reading image…")
    img = cv2.imread(imgpath)
    if img is None:
        print_cb("Failed to read image.")
        window.write_event_value('-TEST-DONE-', None)
        return

    print_cb("Running prediction…")

    # CNN case
    if modelfile.endswith('.h5'):
        img_norm = cv2.resize(img, (64,64)).astype('float32') / 255.0
        probs = mdl.predict(np.expand_dims(img_norm, 0))[0]
        # both scores
        print_cb(f"Real: {probs[0]*100:.1f}%, AI: {probs[1]*100:.1f}%")
        winner = int(np.argmax(probs))
        labels = {0: "Real", 1: "AI"}
        print_cb(f"Verdict: {labels[winner]} ({probs[winner]*100:.1f}% confidence)")

    # all other sklearn based models
    else:
        # build feature vector Xf
        if modelfile == 'svm_model.joblib':
            img_r = cv2.resize(img, (64,64))
            feat = svm_extract_features(img_r)
            Xf   = scaler.transform([feat])
        elif modelfile == 'lbp1_model.joblib':
            img_g = cv2.resize(
                cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE), 
                (64,64)
            )
            feat = lbp1_extract_features(img_g)
            Xf   = [feat]
        elif modelfile == 'lbp2_model.joblib':
            img_r = cv2.resize(img, (64,64))
            feat = lbp2_extract_features(img_r)
            Xf   = scaler.transform([feat])
        else:
            img_n = cv2.resize(img, (64,64)).astype('float32') / 255.0
            Xf    = [img_n.flatten()]

        # probability based models
        if hasattr(mdl, 'predict_proba'):
            proba = mdl.predict_proba(Xf)[0]
            # show both
            print_cb(f"Real: {proba[0]*100:.1f}%, AI: {proba[1]*100:.1f}%")
            winner = int(np.argmax(proba))
            labels = {0: "Real", 1: "AI"}
            print_cb(f"Verdict: {labels[winner]} ({proba[winner]*100:.1f}% confidence)")
        else:
            label = mdl.predict(Xf)[0]
            print_cb(f"Predicted label: {label}")

    # notify GUI thread to re enable Run
    window.write_event_value('-TEST-DONE-', None)

# show test dialog with preview
def test_window():
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    tm_dir = os.path.join(base, 'trained_models')
    model_files = [f for f in os.listdir(tm_dir) if f.endswith('.h5') or f.endswith('_model.joblib')]

    layout = [
        [sg.Text('Select trained model:'), sg.Combo(model_files, key='MODELFILE')],
        [
            sg.Column([
                [sg.Text('Select image to test:')],
                [sg.InputText(key='IMGPATH', enable_events=True, size=(40,1)),
                 sg.FileBrowse(file_types=(("Image Files","*.png *.jpg *.jpeg"),))],
                [sg.Text('Preview:')],
                [sg.Image(key='PREVIEW', size=(200,200))]
            ]),
            sg.VerticalSeparator(),
            sg.Column([
                [sg.Button('Run', key='RUN'), sg.Button('Back')],
                [sg.Multiline(size=(60,10), key='OUT', autoscroll=True, disabled=True)]
            ]),
        ]
    ]

    win = sg.Window('Test Models', layout, modal=True, finalize=True)

    while True:
        event, values = win.read(timeout=100)
        if event in (sg.WIN_CLOSED, 'Back'):
            break

        if event == 'IMGPATH':
            path = values['IMGPATH']
            if os.path.exists(path):
                # load and resize for preview
                img = Image.open(path)
                img.thumbnail((200,200))
                bio = io.BytesIO()
                img.save(bio, format='PNG')
                win['PREVIEW'].update(data=bio.getvalue())
            else:
                win['PREVIEW'].update(data=b'')

        elif event == 'RUN':
            if not values['MODELFILE'] or not values['IMGPATH']:
                win['OUT'].print("Please choose both a model and an image.")
                continue

            win['RUN'].update(disabled=True)
            win['OUT'].update('')
            threading.Thread(
                target=do_test,
                args=(values['MODELFILE'], values['IMGPATH'], win),
                daemon=True
            ).start()

        elif event == '-TEST-DONE-':
            win['RUN'].update(disabled=False)

    win.close()

# show PCA visualization
def pca_window():
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    out_dir = base

    col_scree = [[sg.Text('Scree Plot:')], [sg.Image(key='PCA_SCRE', visible=False, size=(600,450))]]
    col_scatter = [[sg.Text('2D PCA Scatter:')], [sg.Image(key='PCA_SCAT', visible=False, size=(600,450))]]

    layout = [
        [sg.Text('Number of images for PCA:'), sg.InputText('1000', key='NUM')],
        [sg.Button('Run PCA', key='RUN'), sg.Button('Back')],
        [sg.ProgressBar(100, orientation='h', size=(60,20), key='PCA_PROG')],
        [sg.Multiline(size=(60,10), key='PCA_OUT', autoscroll=True, disabled=True)],
        [
            sg.Column(col_scree, size=(620,520), expand_x=True, expand_y=True),
            sg.VerticalSeparator(),
            sg.Column(col_scatter, size=(620,520), expand_x=True, expand_y=True)
        ],
    ]

    win = sg.Window('Visualize PCA', layout, modal=True, size=(1600,900), resizable=True, finalize=True)

    while True:
        event, values = win.read(timeout=100)
        if event in (sg.WIN_CLOSED, 'Back'):
            break

        if event == 'RUN':
            win['RUN'].update(disabled=True)
            win['PCA_PROG'].update_bar(0)
            win['PCA_OUT'].update('')
            win['PCA_SCRE'].update(visible=False)
            win['PCA_SCAT'].update(visible=False)

            try:
                n = int(values['NUM'])
            except ValueError:
                win['PCA_OUT'].print("Enter a valid integer for number of images.")
                win['RUN'].update(disabled=False)
                continue

            def pca_progress_cb(done, total, msg):
                win['PCA_PROG'].update_bar(int(done/total*100))
                win['PCA_OUT'].print(msg)
                if "2D PCA scatter saved" in msg:
                    win.write_event_value('-PCA-DONE-', None)

            threading.Thread(target=visualize_pca, args=(n, pca_progress_cb), daemon=True).start()

        elif event == '-PCA-DONE-':
            win['PCA_SCRE'].update(filename=os.path.join(out_dir, "pca_scree_plot.png"), visible=True)
            win['PCA_SCAT'].update(filename=os.path.join(out_dir, "pca_2d_scatter.png"), visible=True)
            win['RUN'].update(disabled=False)

    win.close()

# show t-SNE visualization
def tsne_window():
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    out_dir = base

    layout = [
        [sg.Text('Number of images for t-SNE:'), sg.InputText('1000', key='NUM')],
        [sg.Button('Run t-SNE', key='RUN'), sg.Button('Back')],
        [sg.ProgressBar(100, orientation='h', size=(40,20), key='TSNE_PROG')],
        [sg.Multiline(size=(60,10), key='TSNE_OUT', autoscroll=True, disabled=True)],
        [sg.Text('t-SNE Scatter:'), sg.Image(key='TSNE_IMG', visible=False)],
    ]

    win = sg.Window('Visualize t-SNE', layout, modal=True, finalize=True)

    while True:
        event, values = win.read(timeout=100)
        if event in (sg.WIN_CLOSED, 'Back'):
            break

        if event == 'RUN':
            win['RUN'].update(disabled=True)
            win['TSNE_PROG'].update_bar(0)
            win['TSNE_OUT'].update('')
            win['TSNE_IMG'].update(visible=False)

            try:
                n = int(values['NUM'])
            except ValueError:
                win['TSNE_OUT'].print("Enter a valid integer for number of images.")
                win['RUN'].update(disabled=False)
                continue

            def tsne_progress_cb(done, total, msg):
                win['TSNE_PROG'].update_bar(int(done/total*100))
                win['TSNE_OUT'].print(msg)
                if "Saved t-SNE scatter" in msg:
                    win.write_event_value('-TSNE-DONE-', None)

            threading.Thread(target=visualize_tsne, args=(n, tsne_progress_cb), daemon=True).start()

        elif event == '-TSNE-DONE-':
            win['TSNE_IMG'].update(filename=os.path.join(out_dir, "tsne_scatter.png"), visible=True)
            win['RUN'].update(disabled=False)

    win.close()

# main menu
if __name__ == '__main__':
    sg.theme('LightBlue2')
    layout = [
        [sg.Text('Welcome to AI Buster', font=('Helvetica', 16))],
        [sg.VerticalSeparator(),
         sg.Button('Train Models'),
         sg.Button('Test Models'),
         sg.Button('Visualize PCA'),
         sg.Button('Visualize t-SNE')],
    ]
    win = sg.Window('AI Buster', layout)

    while True:
        event, _ = win.read()
        if event in (sg.WIN_CLOSED, None):
            break
        if event == 'Train Models':
            train_window()
        elif event == 'Test Models':
            test_window()
        elif event == 'Visualize PCA':
            pca_window()
        elif event == 'Visualize t-SNE':
            tsne_window()

    win.close()

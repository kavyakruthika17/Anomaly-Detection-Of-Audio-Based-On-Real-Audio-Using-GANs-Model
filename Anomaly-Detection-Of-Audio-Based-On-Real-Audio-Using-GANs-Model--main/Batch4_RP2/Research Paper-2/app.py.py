import os
from flask import Flask, request, render_template
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

app = Flask(__name__)

def extract_audio_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=40):
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None, None

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_mean = np.mean(log_mel_spec.T, axis=0)
    mel_std = np.std(log_mel_spec.T, axis=0)

    features = np.concatenate([mfccs_mean, mfccs_std, mel_mean, mel_std])
    spectrogram_filename = save_spectrogram_plot(y, sr, audio_path)
    return features, spectrogram_filename

def save_spectrogram_plot(y, sr, audio_path):
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=40)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()

    if not os.path.exists("static/spectrograms"):
        os.makedirs("static/spectrograms")

    filename = os.path.basename(audio_path).replace(".wav", "_spectrogram.png")
    filepath = os.path.join("static/spectrograms", filename)
    plt.savefig(filepath)
    plt.close()
    return filename  # Only return filename, not full path

def analyze_audio(input_audio_path):
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"

    if not os.path.exists(input_audio_path):
        return "Error: file does not exist.", None
    elif not input_audio_path.lower().endswith(".wav"):
        return "Error: not a .wav file.", None

    features, spectrogram_filename = extract_audio_features(input_audio_path)
    if features is not None:
        scaler = joblib.load(scaler_filename)
        features_scaled = scaler.transform(features.reshape(1, -1))
        svm_classifier = joblib.load(model_filename)
        prediction = svm_classifier.predict(features_scaled)
        result_text = "The input audio is classified as Genuine." if prediction[0] == 0 else "The input audio is classified as Deepfake."
        return result_text, spectrogram_filename
    else:
        return "Could not extract features from the audio.", None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "wav"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audio_file" not in request.files:
            return render_template("index.html", message="No file part.")
        audio_file = request.files["audio_file"]
        if audio_file.filename == "":
            return render_template("index.html", message="No selected file.")
        if audio_file and allowed_file(audio_file.filename):
            if not os.path.exists("uploads"):
                os.makedirs("uploads")
            audio_path = os.path.join("uploads", audio_file.filename)
            audio_file.save(audio_path)
            result, spectrogram_filename = analyze_audio(audio_path)
            os.remove(audio_path)
            return render_template("result.html", result=result, spectrogram_filename=spectrogram_filename)
        return render_template("index.html", message="Invalid file format. Only .wav allowed.")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

    import os
    import glob
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa.display
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix
    import joblib

    def extract_audio_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=40):
        try:
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None

        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)

        # Log-mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = np.mean(log_mel_spec.T, axis=0)
        mel_std = np.std(log_mel_spec.T, axis=0)

        # Concatenate all features
        features = np.concatenate([mfccs_mean, mfccs_std, mel_mean, mel_std])

        # Save spectrogram plot
        save_spectrogram_plot(y, sr, audio_path)

        return features

    def save_spectrogram_plot(y, sr, audio_path):
        plt.figure(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=40)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()

        if not os.path.exists("outputs/spectrograms"):
            os.makedirs("outputs/spectrograms")

        filename = os.path.basename(audio_path).replace(".wav", "_spectrogram.png")
        plt.savefig(os.path.join("outputs/spectrograms", filename))
        plt.close()

    def create_dataset(directory, label):
        X, y = [], []
        audio_files = glob.glob(os.path.join(directory, "*.wav"))
        for audio_path in audio_files:
            features = extract_audio_features(audio_path)
            if features is not None:
                X.append(features)
                y.append(label)
            else:
                print(f"Skipping audio file {audio_path}")

        print("Number of samples in", directory, ":", len(X))
        print("Filenames:", [os.path.basename(path) for path in audio_files])
        return X, y

    def train_model(X, y):
        X = np.array(X)
        y = np.array(y)

        unique_classes = np.unique(y)
        print("Unique classes in y:", unique_classes)

        if len(unique_classes) < 2:
            raise ValueError("At least two classes are needed to train the model.")

        class_counts = np.bincount(y)
        if np.min(class_counts) < 2:
            print("Not enough samples to split. Training on all data.")
            X_train, y_train = X, y
            X_test, y_test = None, None
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            print("Train / test sizes:", X_train.shape, X_test.shape)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        svm_classifier = SVC(kernel='linear', random_state=42)
        svm_classifier.fit(X_train_scaled, y_train)

        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            y_pred = svm_classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy:", accuracy)
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        joblib.dump(svm_classifier, "svm_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        print("Model and scaler saved.")

    def analyze_audio(input_audio_path):
        model_filename = "svm_model.pkl"
        scaler_filename = "scaler.pkl"

        svm_classifier = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)

        if not os.path.exists(input_audio_path):
            print("Error: file does not exist.")
            return
        elif not input_audio_path.lower().endswith(".wav"):
            print("Error: not a .wav file.")
            return

        features = extract_audio_features(input_audio_path)
        if features is not None:
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = svm_classifier.predict(features_scaled)
            print("Predicted:", "Genuine" if prediction[0]==0 else "Deepfake")
        else:
            print("Could not extract features.")

    def main():
        genuine_dir = r"C:\Users\nrban\OneDrive\Desktop\Research Paper-1\real_audio"
        deepfake_dir = r"C:\Users\nrban\OneDrive\Desktop\Research Paper-1\deepfake_audio"

        X_genuine, y_genuine = create_dataset(genuine_dir, label=0)
        X_deepfake, y_deepfake = create_dataset(deepfake_dir, label=1)

        X = np.vstack((X_genuine, X_deepfake))
        y = np.hstack((y_genuine, y_deepfake))

        train_model(X, y)

    if __name__ == "__main__":
        main()

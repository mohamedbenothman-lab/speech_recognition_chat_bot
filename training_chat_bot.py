import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import joblib
import sounddevice as sd
import time
from collections import Counter
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

#Configuration 
DATA_DIR      = r"./speech_commands_v0.02"
AUDIO_DATA, SAMPLE_RATE = sf.read("./speech_commands_v0.02/yes/0a7c2a8d_nohash_0.wav")
ALLOWED_WORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'stop']
RANDOM_STATE  = 42
CACHE_FILE    = "dataset_cache.joblib"
MODEL_FILE    = "best_model.joblib"
ENCODER_FILE  = "label_encoder.joblib"

# Task 1 : Audio Exploration
def explore_audio():
    print(f"Sample rate      : {SAMPLE_RATE} Hz")
    print(f"Number of samples: {len(AUDIO_DATA)}")
    print(f"Duration         : {len(AUDIO_DATA) / SAMPLE_RATE:.2f} seconds")
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y=AUDIO_DATA, sr=SAMPLE_RATE)
    plt.title("Waveform of the audio file")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


# Task 2 : Improved Feature Extraction
def extract_features(audio_path, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extracts a rich, fixed-length feature vector from a WAV file.

    Features included
    -----------------
    • MFCCs  (n_mfcc coefficients)
    • Delta-MFCCs  (first derivative)
    • Delta²-MFCCs (second derivative)
    • Chroma (12 pitch-class energies)
    • Spectral contrast (7 bands)
    • Zero-crossing rate (1 value)

    For every feature *matrix* we compute both the mean AND the standard
    deviation across time, so the model sees both the average shape and
    how much the signal varies — much more discriminative than mean alone.

    Returns
    -------
    numpy.ndarray: 1-D feature vector (length ≈ 2 × (n_mfcc×3 + 12 + 7 + 1))
    """
    audio, _ = librosa.load(audio_path, sr=sr)

    # --- MFCCs and their derivatives ---
    mfccs       = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                       n_fft=n_fft, hop_length=hop_length)
    mfcc_delta  = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    # --- Supplementary features ---
    chroma   = librosa.feature.chroma_stft(y=audio, sr=sr,
                                           n_fft=n_fft, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr,
                                                 n_fft=n_fft, hop_length=hop_length)
    zcr      = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)

    # Stack into one matrix, then summarise with mean + std
    combined = np.vstack([mfccs, mfcc_delta, mfcc_delta2, chroma, contrast, zcr])

    mean_vec = np.mean(combined, axis=1)   # shape: (n_features,)
    std_vec  = np.std(combined,  axis=1)   # shape: (n_features,)

    return np.concatenate([mean_vec, std_vec])   # double the info vs. mean alone


# Task 2b : Data Augmentation
def augment_audio(audio, sr):
    """
    Returns a list of augmented versions of the input waveform.
    Each variant introduces a realistic acoustic distortion so the
    model generalises to real-world microphone / speaker variation.

    Augmentations
    -------------
    • Time-stretch  ×0.9  (speak slower)
    • Time-stretch  ×1.1  (speak faster)
    • Pitch-shift   +1 semitone
    • Pitch-shift   −1 semitone
    • White-noise addition (SNR ≈ 40 dB)
    """
    augmented = []

    # Time stretching
    augmented.append(librosa.effects.time_stretch(audio, rate=0.9))
    augmented.append(librosa.effects.time_stretch(audio, rate=1.1))

    # Pitch shifting
    augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=1))
    augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-1))

    # Additive white noise
    noise = np.random.randn(len(audio)) * 0.005
    augmented.append(audio + noise)

    return augmented


# Task 3 : Dataset Preparation
def prepare_dataset(data_directory, sr=16000, n_mfcc=13,
                    allowed_words=ALLOWED_WORDS, use_augmentation=True):
    """
    Walk *data_directory*, extract feature vectors for every WAV whose
    parent folder is in *allowed_words*, optionally augment each sample,
    balance classes, split into train / test, and scale.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    label_encoder                    : fitted LabelEncoder
    scaler                           : fitted StandardScaler
    """
    X, y = [], []
    count = 0

    for root, _, files in os.walk(data_directory):
        label = os.path.basename(root)
        if label not in allowed_words:
            continue

        for file in files:
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(root, file)
            try:
                feat = extract_features(file_path, sr=sr, n_mfcc=n_mfcc)
                X.append(feat)
                y.append(label)

                # Augmentation
                if use_augmentation:
                    audio, _ = librosa.load(file_path, sr=sr)
                    for aug_audio in augment_audio(audio, sr):
                        # Build a temp feature vector directly from memory
                        mfccs       = librosa.feature.mfcc(y=aug_audio, sr=sr,
                                                            n_mfcc=n_mfcc)
                        mfcc_delta  = librosa.feature.delta(mfccs)
                        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
                        chroma      = librosa.feature.chroma_stft(y=aug_audio, sr=sr)
                        contrast    = librosa.feature.spectral_contrast(y=aug_audio, sr=sr)
                        zcr         = librosa.feature.zero_crossing_rate(aug_audio)
                        combined    = np.vstack([mfccs, mfcc_delta, mfcc_delta2,
                                                 chroma, contrast, zcr])
                        aug_feat = np.concatenate([
                            np.mean(combined, axis=1),
                            np.std(combined,  axis=1)
                        ])
                        X.append(aug_feat)
                        y.append(label)

                count += 1
                if count % 100 == 0:
                    print(f"  Processed {count} original files …")

            except Exception as e:
                print(f"  [WARN] Skipping {file_path}: {e}")

    print(f"\nFinished processing {count} original files "
          f"({'with' if use_augmentation else 'without'} augmentation).")

    X = np.array(X)
    y = np.array(y)

    #Class Balancing (oversample minority classes)
    print("\nClass distribution before balancing:")
    for label, cnt in sorted(Counter(y).items()):
        print(f"  {label:10s}: {cnt}")

    max_count = max(Counter(y).values())
    X_balanced, y_balanced = [], []
    for label in np.unique(y):
        mask    = y == label
        X_class = X[mask]
        y_class = y[mask]
        if len(X_class) < max_count:
            X_class, y_class = resample(X_class, y_class,
                                        n_samples=max_count,
                                        random_state=RANDOM_STATE)
        X_balanced.append(X_class)
        y_balanced.append(y_class)

    X = np.vstack(X_balanced)
    y = np.concatenate(y_balanced)
    print(f"\nAfter balancing: {len(X)} total samples, "
          f"{max_count} per class.")

    #  Encode labels 
    label_encoder = LabelEncoder()
    y_encoded     = label_encoder.fit_transform(y)

    #  Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    # Scale 
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, label_encoder, scaler


# Task 4a : SVM with Cross-Validated Hyperparameter Tuning
def train_svm(X_train, X_test, y_train, y_test, label_encoder):
    """
    Train an RBF-SVM.  A coarse GridSearchCV over C and gamma finds the
    best hyperparameters using 5-fold stratified cross-validation, which
    is far more robust than a single train/test split.

    SVM consistently outperforms Random Forest on short-duration audio
    tasks because the RBF kernel separates the compact MFCC clusters
    that represent different phonemes / words.
    """
    print("\n" + "="*60)
    print(" Training: SVM (RBF kernel) with GridSearchCV")
    print("="*60)

    param_grid = {
        "C":     [1, 10, 50, 100],
        "gamma": ["scale", "auto", 0.001, 0.01],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X_train, y_train)

    print(f"\nBest params : {grid.best_params_}")
    print(f"Best CV acc : {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    y_pred     = best_model.predict(X_test)

    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=label_encoder.classes_))
    print(f"SVM Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    return best_model


# Task 4b : Random Forest with Cross-Validated Hyperparameter Tuning
def train_random_forest(X_train, X_test, y_train, y_test, label_encoder):
    """
    Train a Random Forest with GridSearchCV as a reliable fallback /
    comparison baseline.  Useful when you want feature importances or
    faster inference than SVM on large vocabularies.
    """
    print("\n" + "="*60)
    print(" Training: Random Forest with GridSearchCV")
    print("="*60)

    param_grid = {
        "n_estimators":    [200, 350, 500],
        "max_depth":       [None, 20, 30],
        "min_samples_split": [2, 5],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X_train, y_train)

    print(f"\nBest params : {grid.best_params_}")
    print(f"Best CV acc : {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    y_pred     = best_model.predict(X_test)

    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=label_encoder.classes_))
    print(f"Random Forest Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    return best_model


# Task 4c : Pick and Save the Best Model
def train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder):
    """
    Train both SVM and Random Forest, compare their test accuracy, and
    save whichever performs better.
    """
    svm_model = train_svm(X_train, X_test, y_train, y_test, label_encoder)
    rf_model  = train_random_forest(X_train, X_test, y_train, y_test, label_encoder)

    svm_acc = accuracy_score(y_test, svm_model.predict(X_test))
    rf_acc  = accuracy_score(y_test, rf_model.predict(X_test))

    print("\n" + "="*60)
    if svm_acc >= rf_acc:
        print(f" ✔ SVM wins  ({svm_acc:.4f} vs RF {rf_acc:.4f}) — saving SVM")
        best_model = svm_model
    else:
        print(f" ✔ RF wins  ({rf_acc:.4f} vs SVM {svm_acc:.4f}) — saving RF")
        best_model = rf_model
    print("="*60)

    joblib.dump(best_model,    MODEL_FILE)
    joblib.dump(label_encoder, ENCODER_FILE)
    print(f"\n  Model saved   → {MODEL_FILE}")
    print(f"  Encoder saved → {ENCODER_FILE}")

    return best_model


# Task 5 : Record Words
def record_and_save(filename, folder="recordings", duration=1, sr=16000):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    print(f"\nPrepare to record: {file_path}")
    print("Recording starts in 3 seconds …")
    time.sleep(3)
    print("▶ Recording …")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    sf.write(file_path, recording, sr)
    print(f"  Saved to: {file_path}")


# Task 6 : Classify Own Recordings
def classify_own_recording(audio_path, model, label_encoder, scaler):
    """Extract features, scale, predict, and print the result."""
    feat           = extract_features(audio_path).reshape(1, -1)
    feat_scaled    = scaler.transform(feat)
    predicted_idx  = model.predict(feat_scaled)[0]
    predicted_word = label_encoder.inverse_transform([predicted_idx])[0]
    print(f"  {os.path.basename(audio_path):20s} → {predicted_word}")
    return predicted_word


# Entry Point
if __name__ == "__main__":
    RECORDING_DIRECTORY = "recordings"
    files_to_record     = [f"file_{i}.wav" for i in range(1, 6)]

    # Load or build dataset
    if os.path.exists(CACHE_FILE):
        print("\nLoading cached dataset …")
        (X_train, X_test, y_train, y_test,
         label_encoder, scaler) = joblib.load(CACHE_FILE)
    else:
        print("\n─── Preparing Dataset (first run — this may take a while) ───")
        (X_train, X_test, y_train, y_test,
         label_encoder, scaler) = prepare_dataset(DATA_DIR)

        joblib.dump((X_train, X_test, y_train, y_test,
                     label_encoder, scaler), CACHE_FILE)
        print("  Dataset cached to disk.")

    # Load or train model 
    if os.path.exists(MODEL_FILE):
        print("\nLoading existing model …")
        best_model    = joblib.load(MODEL_FILE)
        label_encoder = joblib.load(ENCODER_FILE)
    else:
        print("\n─── Training Models ───")
        best_model = train_and_evaluate(X_train, X_test,
                                        y_train, y_test, label_encoder)

    # Record files 
    print("\n─── Recording files ───")
    for f in files_to_record:
        record_and_save(f, folder=RECORDING_DIRECTORY)
        input("  Press Enter when ready for the next recording …")

    #  Classify recordings
    print("\n─── Classifying recordings ───")
    for f in files_to_record:
        file_path = os.path.join(RECORDING_DIRECTORY, f)
        classify_own_recording(file_path, best_model, label_encoder, scaler)
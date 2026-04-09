import wfdb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# ============================================================
# HOW TO SWITCH BETWEEN MULTICLASS AND BINARY TESTING
# ============================================================
# Currently have the testing in BINARY mode
#
# Note to self:
# To go back to MULTICLASS (N, L, R, V, A):
# 1) Uncomment the multiclass LABEL_MAP / CLASS_NAMES below
# 2) Comment out the binary LABEL_MAP / CLASS_NAMES below
# 3) Use argmax-only prediction logic
# 4) Remove labels=[0, 1] and use the 5-class report/confusion matrix

# --- MODEL PATH ---
MODEL_PATH = 'models/base_rnn_model_v1.keras' # Use specific model 
TEST_RECORD = '200' # Change to the record you want to test check the README for available records
WINDOW_SIZE = 180    # Match with data_extract_resample.py

# --- LABEL MAPPING ---
# N: Normal, L: Left, R: Right, V: Ventricular, A: Atrial

# For multiclass:
# LABEL_MAP = {'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4}
# CLASS_NAMES = ['Normal', 'L-Bundle', 'R-Bundle', 'Ventricular', 'Atrial']

# For binary:
LABEL_MAP = {'N': 0, 'L': 1, 'R': 1, 'V': 1, 'A': 1}
CLASS_NAMES = ['Normal', 'Abnormal']

def load_test_data(record_id):
    print(f"Loading record {record_id}...")
    record = wfdb.rdrecord(f'data/{record_id}')
    ann = wfdb.rdann(f'data/{record_id}', 'atr')
    
    signal = record.p_signal[:, 0]
    # Standardized exactly how it worked during training
    signal = StandardScaler().fit_transform(signal.reshape(-1, 1)).flatten()
    
    # Extract beats using the same logic as training: 180 samples before and after the annotated R-peak
    beats, labels = [], []
    for idx, symbol in zip(ann.sample, ann.symbol):
        if symbol in LABEL_MAP:
            # Match with training segmentation logic: peak - 90 to peak + 90
            start, end = idx - WINDOW_SIZE // 2, idx + WINDOW_SIZE // 2
            if start > 0 and end < len(signal):
                segment = signal[start:end]
                beats.append(segment)
                labels.append(LABEL_MAP[symbol])
                
    # Final shape: (samples, 180, 1)
    return np.array(beats).reshape(-1, WINDOW_SIZE, 1), np.array(labels)

def run_full_test():
    # Load Model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model {MODEL_PATH} loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Data
    X_test, y_test = load_test_data(TEST_RECORD)
    
    # Predict
    print(f"Inference on {len(X_test)} beats...")
    y_probs = model.predict(X_test)

    # --- MULTICLASS VERSION ---
    # y_pred = np.argmax(y_probs, axis=1)

    # --- BINARY VERSION (ACTIVE) ---
    if y_probs.ndim == 1 or (y_probs.ndim == 2 and y_probs.shape[1] == 1):
        y_pred = (y_probs.ravel() >= 0.5).astype(int)
    else:
        y_pred = np.argmax(y_probs, axis=1)
        if y_probs.shape[1] > 2:
            y_pred = (y_pred != 0).astype(int)

    # Metrics Report
    print("\n" + "="*40)
    print(f" DETAILED RESULTS FOR RECORD {TEST_RECORD} ")
    print("="*40)
    
    # --- MULTICLASS VERSION ---
    # present_indices = np.unique(np.concatenate([y_test, y_pred]))
    # target_names = [CLASS_NAMES[i] for i in present_indices]
    # print(classification_report(y_test, y_pred, target_names=target_names))

    # --- BINARY VERSION ---
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0))

    # --- Confusion Matrix Visualization ---
    # --- MULTICLASS VERSION ---
    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
    #             xticklabels=target_names, yticklabels=target_names)

    # --- BINARY VERSION ---
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix - Record {TEST_RECORD}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Visualization of a random prediction
    idx = np.random.randint(0, len(X_test))
    plt.figure(figsize=(10, 4))
    plt.plot(X_test[idx].flatten(), color='blue', linewidth=1.5)
    plt.title(f"Beat Index: {idx} | Actual: {CLASS_NAMES[y_test[idx]]} | Pred: {CLASS_NAMES[y_pred[idx]]}")
    # Mark the center (the R-peak)
    plt.axvline(x=WINDOW_SIZE//2, color='red', linestyle='--', alpha=0.6, label='Annotated Peak')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_full_test()
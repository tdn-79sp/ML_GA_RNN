import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils import class_weight

# NOTE: I have this saved for the multiclass version of the project
# It shouldn't be all too different, but saved it separate just in case I diverged in logic a lot more for binary
# This should be simple(?) to switch over to later I think...?


# Import preprocessed data and model architecture
from rnn_model import build_base_rnn 

# --- Choose the Experiment Path ---

# OPTION 1: Undersampling (Way 1)
# from data_extract_resample import X_way1 as X, y_way1 as y, X_test_rnn, y_test_final, class_weights_dict

# OPTION 2: SMOTE/ADASYN (Way 2)
from data_extract_resample import X_way2 as X, y_way2 as y, X_test_rnn, y_test_final, class_weights_dict

# OPTION 3: Augmentation + Weights (Way 3)
# from data_extract_resample import X_aug as X, y_aug as y, X_test_rnn, y_test_final, class_weights_dict

# Toggle this to False if you want to ignore the weights for a specific test
USE_WEIGHTS = True

############################################################################

# --- LOAD GA RESULTS ---
params_path = 'best_hyperparameters.json'
if os.path.exists(params_path):
    with open(params_path, 'r') as f:
        best_params = json.load(f)
    print(f"File Found: {params_path}. Loading GA results: {best_params}")
else:
    # Fallback defaults if GA hasn't been run
    print("!!! GA file not found. Using default architecture.")
    best_params = {'window_size': 180, 'units': 64, 'dropout': 0.3, 'lr': 0.001}

# --- PREPARE DATA ---
# Extract values and ensure they are the correct type (JSON shenangigans)
win_size = int(best_params['window_size'])
n_units = int(best_params['units'])
drop_rate = float(best_params['dropout'])
learning_rate = float(best_params['lr'])

# Slice data to match the GA's discovered optimal window
X_sliced = X[:, :win_size, :]

# Split X and y to create a validation set for the training process
# Keep X_test_rnn untouched for the final "Real World" test
X_train, X_val, y_train, y_val = train_test_split(
    X_sliced, y, test_size=0.15, random_state=42, stratify=y
)

# --- INITIALIZE MODEL ---
# Pass the GA values directly into existing function
model = build_base_rnn(
    input_shape=(win_size, 1), 
    num_classes=5,
    units=n_units,
    dropout=drop_rate,
    lr=learning_rate
)

# --- EXECUTE TRAINING ---
# We use ReduceLROnPlateau to adjust learning rate if the model plateaus during training
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
# Early stopping can also be added if desired
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weights_dict if USE_WEIGHTS else None, # This will apply the correct weights for whichever data you imported
    callbacks=[lr_reducer],
    # callbacks=[lr_reducer, early_stopping], # Uncomment if you want to use early stopping
    verbose=1
)

# --- EVALUATE ON TEST SET ---
# We slice the real test set to match the window size as well
X_test_final_sliced = X_test_rnn[:, :win_size, :]

print("\n" + "="*30)
print("FINAL EVALUATION ON REAL TEST DATA")
print("="*30)
test_results = model.evaluate(X_test_final_sliced, y_test_final, verbose=1)
print(f"Test Accuracy: {test_results[1]*100:.2f}%")

# Live Plotting the Results
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss Plot
    ax1.plot(history.history['loss'], label='Train Loss', color='#e74c3c')
    ax1.plot(history.history['val_loss'], label='Val Loss', color='#c0392b', linestyle='--')
    ax1.set_title('Model Loss (Cross-Entropy)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy Plot
    ax2.plot(history.history['accuracy'], label='Train Acc', color='#2ecc71')
    ax2.plot(history.history['val_accuracy'], label='Val Acc', color='#27ae60', linestyle='--')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_history(history)

# Save final model
if not os.path.exists('models'): os.makedirs('models')
model.save('models/base_rnn_model_v1.keras')
print("Model saved to /models folder.")
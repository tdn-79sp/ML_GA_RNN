import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

# ============================================================
# HOW TO SWITCH BACK TO MULTICLASS (N, L, R, V, A)
# ============================================================
# Supports both modes automatically via num_classes:
# - num_classes=2  -> binary mode (sigmoid + binary_crossentropy)
# - num_classes=5  -> multiclass mode (softmax + sparse_categorical_crossentropy)
#
# To run multiclass in training/eval scripts:
# 1) Pass num_classes=5 when calling build_base_rnn
# 2) Use original multiclass labels (do not map to binary)
# 3) Use multiclass-compatible evaluation scripts/targets in test_script.py and batch_test.py

# Model architecture flexlible for both binary and multiclass classification
# Have it set so that output layer and loss function automatically adjust based on num_classes
def build_base_rnn(input_shape, num_classes=5, units=64, dropout=0.3, lr=0.001):    
    # Determine if we're in binary or multiclass mode based on num_classes
    is_binary = num_classes == 2
    output_units = 1 if is_binary else num_classes
    output_activation = 'sigmoid' if is_binary else 'softmax'

    # Model Architecture 180 -> 64 -> 32 -> output with dropout in between
    # Input -> BiLSTM (64 units) -> Dropout -> LSTM (32 units) -> Dropout -> Dense (32) -> Output
    model = Sequential([
        # Input shape is (180, 1)
        Input(shape=input_shape),
        
        # Bidirectional LSTM captures patterns in both directions
        Bidirectional(LSTM(units, return_sequences=True)),
        Dropout(dropout),
        
        # Second LSTM layer (takes the sequence from the first)
        LSTM(units // 2),
        Dropout(dropout),
        
        # Hidden Dense layer for feature interpretation
        Dense(32, activation='relu'),
        
        # Output layer switches automatically for binary vs multiclass
        Dense(output_units, activation=output_activation)
    ])

    if is_binary:
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    else:
        loss = 'sparse_categorical_crossentropy'
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics
    )
    
    return model

# Testing stuff if you just run this file directly (not imported)
# if __name__ == "__main__":
#     # Quick test to see if the model builds and summarizes correctly
#     model = build_base_rnn(input_shape=(180, 1), num_classes=5)
#     model.summary()

# Not sure what to put here since the model is meant to be imported and used in main.py and main_multiclass.py, but you can uncomment the block above to do a quick test of the model architecture if you run this file directly.
# Pretty sure I was just testing stuff down here and forgot to remove it but it was there for a reason I just don't remember what I was using it for
# whoops
# X_way1_rnn.shape[1:] should be (180, 1) 
# model = build_base_rnn(input_shape=(180, 1), num_classes=5)
# model.summary()


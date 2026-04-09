import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

# Copy cause I was testing stuff and was scared I couldn't get it back if I messed with the original too much. 
# Bruh I copied the same changed file instead of the original, so now I have two copies of the same file with the same edits. 
# Oh well, at least I have a backup if I mess up the new original...

def build_base_rnn(input_shape, num_classes=5, units=64, dropout=0.3, lr=0.001):    
    is_binary = num_classes == 2
    output_units = 1 if is_binary else num_classes
    output_activation = 'sigmoid' if is_binary else 'softmax'

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

# What was I even doing with these lines down here?
# X_way1_rnn.shape[1:] should be (180, 1)
# model = build_base_rnn(input_shape=(180, 1), num_classes=5)
# model.summary()


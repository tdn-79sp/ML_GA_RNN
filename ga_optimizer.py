import pygad
import tensorflow as tf
from sklearn.metrics import f1_score

# Was just a rough sketch of how the GA optimization could work, but the actual implementation was expanded a bit more in genetic_rnn_optimizer.py 
# The idea is to use PyGAD to optimize the hyperparameters of the RNN model, using F1-score as the fitness function
# The GA will evolve a population of hyperparameter sets
#   For each set: we build and train a model -> evaluate it on a validation set -> return the F1-score as the fitness
#   GA then uses this feedback to evolve better hyperparameters over generations
# Tested this initially as a proof of concept and reference 

# Example fitness function for PyGAD
def fitness_func(ga_instance, solution, solution_idx):
    # 1. Unpack 'chromosomes' into hyperparameters
    learning_rate = solution[0]
    num_units = int(solution[1])
    dropout_rate = solution[2]
    
    # Build the RNN based on GA's 'genes'
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(num_units, input_shape=(360, 1)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy')
    
    # Train (keep epochs low during GA to save time)
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
    
    # Evaluate using F1-score
    predictions = np.argmax(model.predict(X_val), axis=1)
    score = f1_score(y_val, predictions, average='macro')
    
    return score # Higher is better
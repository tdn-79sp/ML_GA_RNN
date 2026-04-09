import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import json

# ============================================================
# HOW TO SWITCH BACK TO MULTICLASS (N, L, R, V, A)
# ============================================================
# 1) In build_ga_model:
#    - Use the commented MULTICLASS block for output/loss
#    - Disable the active BINARY block
#    - Final layer should be Dense(num_classes, activation='softmax')
#    - Loss should be 'sparse_categorical_crossentropy'
#
# 2) In GeneticOptimizer.fitness:
#    - Use the commented MULTICLASS prediction/F1 block
#    - Disable the active BINARY prediction/F1 block
#
# 3) In __main__ train/validation split:
#    - Use the commented MULTICLASS split (stratify=y)
#    - Disable the active BINARY split (y_binary)
#
# 4) Ensure num_classes is 5 when calling build_ga_model ****
#
# 5) Keep data import as X_way1/y_way1 or X_way2/y_way2 (both are multiclass labels)
#    No label remapping is needed in multiclass mode

# Import preprocessed data from existing data extraction module
# Note: You can switch between the two resampling methods by commenting/uncommenting the appropriate import line below
# from data_extract_resample import X_way1 as X, y_way1 as y
from data_extract_resample import X_way2 as X, y_way2 as y

def to_binary_labels(labels):
    # Map labels to binary: 0 = Normal, 1 = Abnormal.
    return (labels != 0).astype(np.int32)

# --- SEARCH SPACE DEFINITION ---
SEARCH_SPACE = {
    # These are just placeholder values I threw together for initial testing
    # These aren't necessarily the best choices, but they give us a starting point to see if the GA is working and improving over generations
    # Could reference the paper on hyperparam tuning but I forget if they mentioned the specific ranges they used for the RNN architecture
    'window_size': [180, 240, 360], 
    'units': [32, 64, 128],        
    'dropout': [0.2, 0.3, 0.5],    
    'lr': [0.001, 0.0001]          
}

# --- MODEL BUILDER ---
def build_ga_model(chromosome, num_classes=2):
    # Translates a dictionary of genes into a Keras model
    win = int(chromosome['window_size'])
    uni = int(chromosome['units'])
    drop = chromosome['dropout']
    lr = chromosome['lr']

    # --- MULTICLASS VERSION ---
    # output_units = num_classes
    # output_activation = 'softmax'
    # loss_name = 'sparse_categorical_crossentropy'

    # --- BINARY VERSION ---
    output_units = 1
    output_activation = 'sigmoid'
    loss_name = 'binary_crossentropy'

    # Model Architecture 180 -> 64 -> 32 -> output with dropout in between
    # Input -> BiLSTM (64 units) -> Dropout -> LSTM (32 units) -> Dropout -> Dense (32) -> Output
    # The architecture is set to be flexible for both binary and multiclass by adjusting the final layer and loss function based on the num_classes parameter
    # For the GA, I set num_classes=2 for binary evaluation, but the same function can be used with num_classes=5 for multiclass if we switch the active blocks in the code
    model = Sequential([
        # Input shape is (window_size, 1) - the GA will adjust the window size based on the chromosome
        Input(shape=(win, 1)),
        Bidirectional(LSTM(uni, return_sequences=True)),
        Dropout(drop),
        LSTM(uni // 2), # Second LSTM layer with half the units of the first
        Dense(32, activation='relu'), # Hidden Dense layer for feature interpretation
        Dense(output_units, activation=output_activation) # Output layer adjusts based on binary vs multiclass
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_name,
        metrics=['accuracy'] # Should probably add more metrics to track here but again just a rough initial version
    )
    return model

# --- THE GENETIC OPTIMIZER CLASS ---
class GeneticOptimizer:
    def __init__(self, population_size=6):
        # The population size is intentionally small for "quick" testing, but can be increased for a more thorough search - it'll take eons to run 
        self.population_size = population_size
        # Initialize population with random chromosomes
        self.population = [self.generate_random_chromosome() for _ in range(population_size)]

    def generate_random_chromosome(self):
        # This creates a random combination of hyperparameters for the initial population
        return {key: random.choice(values) for key, values in SEARCH_SPACE.items()}

    def fitness(self, chromosome, X_train, y_train, X_val, y_val):
        # Evaluation focused on abnormal-vs-normal detection (binary F1) since that's what we're focusing on 
        # The GA will try to maximize this score, so we weight the abnormal F1 more heavily than the overall macro F1 to encourage better performance
        try:
            # Re-slice data to match the chromosome's window size
            # Basically just to ensure that the model is always trained and evaluated on the same window size that it's designed for based on the chromosome, since the GA is optimizing the window size as well
            win = chromosome['window_size']
            X_t = X_train[:, :win, :]
            X_v = X_val[:, :win, :]

            model = build_ga_model(chromosome)
            
            # Somewhat "fast" training for evaluation (3 epochs is enough to see potential) but it takes a fat minute 
            model.fit(X_t, y_train, epochs=3, batch_size=64, verbose=0)

            # --- MULTICLASS VERSION ---
            # y_pred = np.argmax(model.predict(X_v, verbose=0), axis=1)
            # f1_all = f1_score(y_val, y_pred, average='macro', zero_division=0)
            # f1_v = f1_score(y_val, y_pred, labels=[3], average='macro', zero_division=0)
            # f1_a = f1_score(y_val, y_pred, labels=[4], average='macro', zero_division=0)
            # weighted_score = (f1_all * 0.3) + (f1_v * 0.3) + (f1_a * 0.4)
            # return weighted_score

            # --- BINARY VERSION ---
            # predict probabilities and convert to binary predictions based on a 0.5 threshold
            # .ravel() is used to flatten the output in case it's a (n, 1) shape from a single-unit sigmoid output
            # Baiscally just forces the output into a 1D array of probabilities that we can threshold to get binary predictions
            y_prob = model.predict(X_v, verbose=0).ravel()
            y_pred = (y_prob >= 0.5).astype(int)
            # Calculate both the overall macro F1 and the abnormal-specific F1, then combine them into a weighted score that emphasizes the abnormal detection performance
            f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
            f1_abnormal = f1_score(y_val, y_pred, labels=[1], average='macro', zero_division=0)
            
            # The weights here can be adjusted based on how much we want to prioritize overall balanced performance (macro F1) versus specifically improving the abnormal detection (abnormal F1) 
            # In this case, I set it to 40% macro and 60% abnormal to encourage better performance on the abnormal class while still considering overall performance
            weighted_score = (f1_macro * 0.4) + (f1_abnormal * 0.6)
            return weighted_score
        
        except Exception as e:
            print(f"Error evaluating individual: {e}")
            return 0

    def evolve(self, scores):
        # Selection, Crossover, and Mutation 

        # Selection: Keep top 2
        sorted_indices = np.argsort(scores)[::-1]
        best_performers = [self.population[i] for i in sorted_indices[:2]]
        
        new_population = list(best_performers) # Elitism

        while len(new_population) < self.population_size:
            # Crossover/Mutation
            parent = random.choice(best_performers)
            child = parent.copy()
            # Mutate one gene
            gene = random.choice(list(SEARCH_SPACE.keys()))
            child[gene] = random.choice(SEARCH_SPACE[gene])
            new_population.append(child)
            
        self.population = new_population

# --- MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    # IMPORT DATA (Assuming it exists in the specified format from the data extraction module)
    # from data_extract_resample import X_way2, y_way2
    # For now, we'll assume X_way2 and y_way2 are loaded
    
    # --- MULTICLASS VERSION ---
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=0.2, stratify=y
    # )

    # --- BINARY VERSION ---
    y_binary = to_binary_labels(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_binary, test_size=0.2, stratify=y_binary
    )

    # Initialize GA and evolve over generations
    ga = GeneticOptimizer(population_size=6)
    num_generations = 3

    print(f"Starting Genetic Evolution for {num_generations} generations...")

    for gen in range(num_generations):
        print(f"\n=== GENERATION {gen + 1} ===")
        gen_scores = []
        
        for i, individual in enumerate(ga.population):
            print(f"Testing Indiv {i+1}: {individual}")
            score = ga.fitness(individual, X_train, y_train, X_val, y_val)
            gen_scores.append(score)
            print(f"Resulting Fitness Score: {score:.4f}")

        ga.evolve(gen_scores)

    # FINAL RESULTS
    best_final_score_idx = np.argmax(gen_scores)
    winner = ga.population[best_final_score_idx]
    print("\n" + "="*40)
    print("EVOLUTION COMPLETE")
    print(f"Best Architecture: {winner}")
    print("="*40)

# Save the best architecture to a JSON file for future reference
with open('best_hyperparameters.json', 'w') as f:
    json.dump(winner, f)
print("Best parameters saved to best_hyperparameters.json")
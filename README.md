# ECG Arrhythmia Classification Project

This project trains an RNN on MIT-BIH ECG beats to classify heartbeats from the MIT-BIH Arrhythmia Database. (Dataset subject to change - potentially shifting towards 12-lead dataset)

The codebase currently is set up with **binary classification** as the active workflow:
- **Normal** beats
- **Abnormal** beats

The initial **multiclass** implementation is still preserved in comments inside the main scripts so it can be restored later for detailed expansion.

## Project structure

### Core files

#### [data_extract_resample.py](data_extract_resample.py)
Probably the most important part of the project - responsible for loading ECG records, segmenting beats, normalizing signals, and preparing training/test splits.

Key functions and outputs:
- `load_and_segment(records)`: loads raw ECG records and extracts beat windows
- `load_and_segment_with_shifts(records)`: augmentation version that adds shifted abnormal beats
- `get_stratify_labels(labels)`: safely decides whether stratified splitting can be used
- Produces arrays such as:
  - `X_way1`, `y_way1`
  - `X_way2`, `y_way2`
  - `X_aug`, `y_aug`
  - `X_test_rnn`, `y_test_final`
  - `class_weights_dict`

What it does overall:
- Reads ECG signal and annotation files from the `data/` folder
- Extracts fixed-width beat windows around annotated peaks
- Applies normalization
- Creates training/testing datasets
- Optionally balances the training data using undersampling or SMOTE/ADASYN resampling
- Data Augmentation functionality is implemented for further testing

#### [rnn_model.py](rnn_model.py)
Generic outline RNN - defines the reusable RNN architecture.

Key function:
- `build_base_rnn(input_shape, num_classes=5, units=64, dropout=0.3, lr=0.001)`

What it does:
- Builds a Bidirectional LSTM-based network
- Automatically switches between:
  - **binary mode** when `num_classes=2`
  - **multiclass mode** when `num_classes=5`
- Chooses the correct output layer and loss function based on the mode

#### [genetic_rnn_optimizer.py](genetic_rnn_optimizer.py)
Rough build of the GA - runs a genetic algorithm to search for better RNN hyperparameters. (Needs overall refining and better chromosome candidates)

Key parts:
- `SEARCH_SPACE`: ranges for window size, units, dropout, and learning rate
- `build_ga_model(chromosome, num_classes=2)`: builds a model from a GA chromosome
- `GeneticOptimizer.generate_random_chromosome()`: creates random candidate architectures
- `GeneticOptimizer.fitness(...)`: scores each candidate using validation performance
- `GeneticOptimizer.evolve(scores)`: keeps the best models and mutates the population

What it does overall:
- Trains several candidate RNN settings for a few epochs each
- Scores them using validation performance
- Saves the best hyperparameters to `best_hyperparameters.json`

Notes:
- The binary version is currently active
- The multiclass version is still kept in commented blocks for later reuse

#### [main.py](main.py)
Ties everything from before together - trains the final model using the saved hyperparameters from the GA.

Key parts:
- Loads `best_hyperparameters.json`
- Chooses the active dataset import from `data_extract_resample.py`
- Converts labels to binary with `to_binary_labels(labels)`
- Splits training and validation sets
- Builds the final RNN using `build_base_rnn(...)`
- Trains the model and saves it to `models/base_rnn_model_v1.keras`

What it does overall:
- Uses the GA result to train the final model
- Applies class weights if enabled
- Plots loss and accuracy over epochs
- Saves the trained model for later testing

### Testing files

#### [test_script.py](test_script.py)
General test for the saved model on one ECG record and shows detailed performance.

Key parts:
- `load_test_data(record_id)`: loads and segments one test record using the same preprocessing as training
- `run_full_test()`: loads the model, runs inference, prints classification metrics, and displays a confusion matrix

What it shows:
- Classification report
- Confusion matrix
- A random beat visualization with the annotated peak

Current mode:
- Binary classification is active
- Multiclass logic is preserved in comments

#### [batch_test.py](batch_test.py)
Runs testing across multiple records and summarizes performance.

Key parts:
- Uses `load_test_data()` from `test_script.py`
- Loads the saved model once
- Evaluates each record in a loop
- Computes accuracy, macro F1, and abnormal F1

What it shows:
- Per-record metrics table
- Overall average metrics across all tested records

Current mode:
- Binary classification is active
- Multiclass metric logic is preserved in comments

### Other files

#### [main_multiclass.py](main_multiclass.py)
Backup or older multiclass training version.

#### [ga_optimizer.py](ga_optimizer.py)
Older or experimental GA-related file.

#### [preprocess.py](preprocess.py)
Experimental preprocessing helper.

#### [rnn_model coppy.py](rnn_model%20coppy.py)
Backup copy of the model file.

## Recommended workflow

### 1. Set up the Python environment

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
py --version
```

Ensure you have Python 3.12 installed. Create a virtual environment (I needed to do this on my machine) and check which version it's running on. Python 3.12 is the latest version that works with the necessary dependencies - newer python versions will cause issues.

Install dependencies:

```powershell
pip install tensorflow wfdb scikit-learn imbalanced-learn pandas numpy seaborn matplotlib
```

### 2. Prepare the data

Run:

```powershell
py .\data_extract_resample.py
```

This creates the beat windows, train/test split, and class weights used by the rest of the pipeline.

### 3. Optimize the model architecture

Run:

```powershell
py .\genetic_rnn_optimizer.py
```

This searches for better hyperparameters and writes the best result to `best_hyperparameters.json`.

### 4. Train the final model

Run:

```powershell
py .\main.py
```

This loads the GA result, trains the final model, evaluates it, and saves the model to `models/base_rnn_model_v1.keras`.

### 5. Test the model

Run a single-record test:

```powershell
py .\test_script.py
```

Run batch testing across several records:

```powershell
py .\batch_test.py
```

## Binary vs multiclass

The active codebase is currently configured for **binary classification**.

Binary mode:
- `Normal = 0`
- `Abnormal = 1`

Self note for going back to multiclass later:
- Restore the commented multiclass label maps in `test_script.py` and `batch_test.py`
- Use `num_classes=5` in `main.py` and `genetic_rnn_optimizer.py`
- Remove binary label conversion in `main.py`
- Use multiclass prediction and reporting logic again

## Available records

### 100 series
Primarily chosen to represent a variety of ECG morphologies (wider range of different looking ones)
.

- 100, 101, 102, 103, 104, 105, 106, 107, 108, 109
- 111, 112, 113, 114, 115, 116, 117, 118, 119
- 121, 122, 123, 124

### 200 series
Rarer but pretty significant arrhythmias

- 200, 201, 202, 203, 205, 207, 208, 209, 
- 210, 212, 213, 214, 215, 217, 219, 
- 220, 221, 222, 223, 228, 
- 230, 231, 232, 233

## Notes about running it

- The GA step takes eons to run - ideally we'd want more generations but it takes a dynasty to run with the current setup already. 
    - Definitely DO NOT run this on your laptop, because it will FRY and attempt unwilling levitation via jet propulsion from its interal fans.
- Make sure the model, test script, and batch test script all agree on binary vs multiclass mode - I'm pretty sure I have it set to all binary but still check
- Might need to record more metrics and compare precision, recall, and F1 rather than accuracy alone.


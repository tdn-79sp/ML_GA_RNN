import wfdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.utils import class_weight

# Configuration
DATA_PATH = 'data/'
WINDOW_SIZE = 180    
RECORDS = ['100', '101', '103', '105', '106', '108', '111'] # Expand as needed

# Define which symbols we care about
# N: Normal, L: Left Bundle, R: Right Bundle, V: Ventricular, A: Atrial
LABEL_MAP = {'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4}

# Stratify function to handle rare classes during train/test split (returns None if stratification isn't possible)
# Stratify is basically just "make sure the train and test sets have the same class distribution as the original dataset"
def get_stratify_labels(labels):
    # Return labels for stratification only when every class has at least 2 samples
    counts = Counter(labels)
    rare_classes = [cls for cls, count in counts.items() if count < 2]
    if rare_classes:
        print(
            "Warning: skipping stratified split because some classes have fewer than "
            f"2 samples: {rare_classes}"
        )
        return None
    return labels

# Loads the records, segments the beats, and applies the necessary preprocessing steps (mainly just normalization, reshaping) to prepare the data for training and testing
# Also have implementation for different resampling strategies to handle class imbalance but need to test them 
# (1) Undersampling the majority class to a 3:1 ratio with the minority classes
# (2) A hybrid approach where we cap the majority class at a certain number and then
#     apply SMOTE or ADASYN to the minority classes to reach a target number of samples (e.g. 1500 per minority class)
#     SMOTE/ADASYN synthesizes new minority class samples to balance the dataset, but capped cause synthetic data can be noisy and we don't want to overwhelm the model with too many synthetic samples that might not be representative of real data
# (3) A simple augmentation strategy where we create shifted versions of the minority class samples to increase their representation without creating synthetic data
#     This can be used in conjunction with class weights to further emphasize the importance of the minority
#     classes during training without overwhelming the model with too many synthetic samples
def load_and_segment(records):
    all_beats = []
    all_labels = []
    scaler = StandardScaler()
    
    for res in records:
        try:
            # Load record
            record = wfdb.rdrecord(DATA_PATH + res)
            ann = wfdb.rdann(DATA_PATH + res, 'atr')
            signal = record.p_signal[:, 0] # Lead II
            
            # Z-score Normalize the entire record signal before segmenting
            signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
            
            for peak, label in zip(ann.sample, ann.symbol):
                if label in LABEL_MAP:
                    start, end = peak - WINDOW_SIZE // 2, peak + WINDOW_SIZE // 2
                    if start > 0 and end < len(signal):
                        segment = signal[start:end]
                        all_beats.append(segment)
                        all_labels.append(LABEL_MAP[label])
        except Exception as e:
            print(f"Skipping record {res}: {e}")
                
    return np.array(all_beats), np.array(all_labels)

# --- LOAD AND PREPARE ---
X, y = load_and_segment(RECORDS)

if len(X) == 0:
    print("No data found. Check your folder path and record IDs.")
    exit()

X_flat = X.reshape(X.shape[0], -1)

# --- TRAIN/TEST SPLIT (Split ratio: 80/20) ---
# We split BEFORE resampling so the test set remains untouched
# Stratify to maintain class distribution in train/test split, but only if we have enough samples for each class to allow it
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=get_stratify_labels(y)
)

print(f"Original Training Distribution: {Counter(y_train)}")

# --- WAY 1: DYNAMIC 3:1 UNDERSAMPLING ---
# Calculate total arrhythmias present to set the 'Normal' (0) target
minority_total = sum(v for k, v in Counter(y_train).items() if k != 0)
target_n_way1 = minority_total * 3

# Only undersample if we actually have more than the target
if Counter(y_train)[0] > target_n_way1:
    rus_31 = RandomUnderSampler(sampling_strategy={0: target_n_way1}, random_state=42)
    X_train_way1, y_train_way1 = rus_31.fit_resample(X_train, y_train)
else:
    X_train_way1, y_train_way1 = X_train, y_train

# --- WAY 2: DYNAMIC HYBRID (Undersample Cap + Synthetic Data) ---
# Step A: Cap 'Normal' (Class 0) at 5000
if Counter(y_train)[0] > 5000:
    under = RandomUnderSampler(sampling_strategy={0: 5000}, random_state=42)
    X_temp, y_temp = under.fit_resample(X_train, y_train)
else:
    X_temp, y_temp = X_train, y_train

#### NOTE: DIFFERENT IMPLEMENTATIONS OF SYNTHETIC OVERSAMPLING ####
# SMOTE VERSION: Better for moderate imbalance, creates synthetic samples along the line segments between neighbors.
# ADASYN VERSION: Focuses more on harder-to-learn samples, can be better for severe imbalance but may introduce more noise.

# Step B: Dynamic SMOTE with Safety Logic
temp_counts = Counter(y_temp)
hybrid_over_strat = {}

for class_id, count in temp_counts.items():
    if class_id != 0:
        # KEY LOGIC: Target set to 1500, UNLESS we already have more than that
        # This prevents the "cannot over-sample to a smaller number" error
        # Could tweak the target number based on the specific class distributions and how much synthetic data you want to introduce, but 1500 is a reasonable starting point that gives the model more examples of the minority classes without overwhelming it with synthetic data
        # Not sure if we want to set a cap on the maximum number of synthetic samples as well to prevent the model from being overwhelmed by synthetic data, but this is a good starting point for now I think
        target_count = max(1500, count)
        
        # Only add to stratification dict if we actually need to oversample
        if target_count > count:
            # Also ensure we have enough neighbors to perform SMOTE (at least 6)
            if count > 6:
                hybrid_over_strat[class_id] = target_count

if hybrid_over_strat:
    over = SMOTE(sampling_strategy=hybrid_over_strat, random_state=42)
    X_train_way2, y_train_way2 = over.fit_resample(X_temp, y_temp)
else:
    X_train_way2, y_train_way2 = X_temp, y_temp
#-------SMOTE VERSION COMPLETE ---------------


# # Step B: Dynamic ADASYN with Safety Logic (Alternative to SMOTE)
# temp_counts = Counter(y_temp)
# hybrid_over_strat = {}

# for class_id, count in temp_counts.items():
#     if class_id != 0:
#         target_count = max(1500, count)
#         if target_count > count:
#             if count > 6: # Still need enough neighbors
#                 hybrid_over_strat[class_id] = target_count

# if hybrid_over_strat:
#     # 2. Swap SMOTE for ADASYN here
#     over = ADASYN(sampling_strategy=hybrid_over_strat, random_state=42)
#     X_train_way2, y_train_way2 = over.fit_resample(X_temp, y_temp)
# else:
#     X_train_way2, y_train_way2 = X_temp, y_temp
# #-------ADASYN VERSION COMPLETE ---------------

###### NOTE: You can choose to use either the SMOTE or ADASYN version by commenting/uncommenting the appropriate block above. Both have the same safety logic to prevent errors.

# --- FINAL RESHAPE ---
X_train_way1_rnn = X_train_way1.reshape(-1, WINDOW_SIZE, 1)
X_train_way2_rnn = X_train_way2.reshape(-1, WINDOW_SIZE, 1)
X_test_rnn = X_test.reshape(-1, WINDOW_SIZE, 1)

print("\n--- Processed Successfully ---")
print(f"Way 1 Train Shape: {X_train_way1_rnn.shape} | Counts: {Counter(y_train_way1)}")
print(f"Way 2 Train Shape: {X_train_way2_rnn.shape} | Counts: {Counter(y_train_way2)}")
print(f"Test Set Shape: {X_test_rnn.shape} (Pure/Real data)")


### ------------------------------------------------------ ###

##### Using Class Weights instead of resampling (Alternative Approach) #####
# No data "hallucination" here, but the model will be penalized more for misclassifying rare classes
# Want it to find the rare patterns so it can learn to detect them, but don't want it to be overwhelmed by the huge number of 'Normal' samples
# Tell it that the rare classes are 50x more important than the 'Normal' class or something

# # --- LOAD AND SPLIT ---
# X, y = load_and_segment(RECORDS)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # --- CALCULATE CLASS WEIGHTS ---
# # This creates a dictionary where rare classes have higher values
# weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weights_dict = dict(enumerate(weights))

# # --- FINAL RESHAPE FOR RNN ---
# X_train_rnn = X_train.reshape(-1, WINDOW_SIZE, 1)
# X_test_rnn = X_test.reshape(-1, WINDOW_SIZE, 1)

# print("\n--- Processed with Class Weights ---")
# print(f"Train Shape: {X_train_rnn.shape}")
# print(f"Class Weights: {class_weights_dict}")



###--- Data Augmentation Function ---###
# (Optional, can be used in training loop instead of resampling)
# Can also use it with the class weights above 
# Multiplies the number of rare samples without creating synthetic data, but it can help the model see more variations of the existing rare samples

# Shifting the signal left or right by a small number of samples to create new "augmented" samples for the minority classes
def shift_signal(signal, shift_limit=10, shift_val=None):
    # Shift signal by a fixed value or a random value within +/- shift_limit.
    # If shift_val is provided, it will be used directly. Otherwise, a random shift will be applied
    if shift_val is None:
        shift = np.random.randint(-shift_limit, shift_limit + 1)
    else:
        shift = int(shift_val)
    return np.roll(signal, shift)

## Modified load_and_segment to include augmentation for rare classes
# The logic is: for each beat, we add the original 
#   If it's a rare class, we also add a left-shifted and right-shifted version of that beat 
# This way we increase the representation of the rare classes without creating synthetic data 
# just slightly modified versions of existing data
def load_and_segment_with_shifts(records):
    all_beats, all_labels = [], []
    scaler = StandardScaler()
    
    for res in records:
        try:
            record = wfdb.rdrecord(DATA_PATH + res)
            ann = wfdb.rdann(DATA_PATH + res, 'atr')
            signal = scaler.fit_transform(record.p_signal[:, 0].reshape(-1, 1)).flatten()
            
            for peak, label in zip(ann.sample, ann.symbol):
                if label in LABEL_MAP:
                    start, end = peak - WINDOW_SIZE // 2, peak + WINDOW_SIZE // 2
                    if start > 0 and end < len(signal):
                        segment = signal[start:end]
                        label_idx = LABEL_MAP[label]
                        
                        # Always add the original
                        all_beats.append(segment)
                        all_labels.append(label_idx)
                        
                        # --- THE SHIFTING LOGIC ---
                        # If it's NOT Normal (0), create 2 extra shifted versions
                        if label_idx != 0:
                            all_beats.append(shift_signal(segment, shift_val=-5)) # Shift left
                            all_labels.append(label_idx)
                            
                            all_beats.append(shift_signal(segment, shift_val=5))  # Shift right
                            all_labels.append(label_idx)
                            
        except Exception as e:
            print(f"Skipping record {res}: {e}")
                
    return np.array(all_beats), np.array(all_labels)


## Example usage:
# # --- LOAD AND PREPARE ---
# X, y = load_and_segment_with_shifts(RECORDS)

# # --- TRAIN/TEST SPLIT ---
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # --- FINAL RESHAPE ---
# X_train_rnn = X_train.reshape(-1, WINDOW_SIZE, 1)
# X_test_rnn = X_test.reshape(-1, WINDOW_SIZE, 1)

# print("\n--- Processed with Time Shifting ---")
# print(f"Final Train Shape: {X_train_rnn.shape}")
# print(f"New Distribution: {Counter(y_train)}")

#### ----- Using Weights AND Augmentation Together ----- #####
# --- LOAD AND SPLIT ---
X, y = load_and_segment_with_shifts(RECORDS)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=get_stratify_labels(y)
)

# --- CALCULATE CLASS WEIGHTS ---
# We calculate this on the augmented training set
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(weights))

# --- FINAL RESHAPE ---
X_train_rnn = X_train.reshape(-1, WINDOW_SIZE, 1)
X_test_rnn = X_test.reshape(-1, WINDOW_SIZE, 1)

print("\n--- Processed with Shifting + Weights ---")
print(f"Final Train Shape: {X_train_rnn.shape}")
print(f"Final Class Weights: {class_weights_dict}")
# You'll need to pass `class_weights_dict` to the `fit()` method of your model during training to use these weights
# Otherwise, the model will treat all classes as equally important, which may not be ideal given the imbalance in the data
# Go to the .fit() call in main.py and add `class_weight=class_weights_dict` to the arguments to enable this feature.


# --- FINAL PREPARATION FOR EXPORT ---

# WAY 1: Undersampling 3:1
X_way1 = X_train_way1.reshape(-1, WINDOW_SIZE, 1)
y_way1 = y_train_way1

# WAY 2: SMOTE/ADASYN (Depending on which block you uncommented above)
X_way2 = X_train_way2.reshape(-1, WINDOW_SIZE, 1)
y_way2 = y_train_way2

# WAY 3: Augmentation + Weights 
# (Note: X_train and y_train here come from the load_and_segment_with_shifts function)
X_aug = X_train.reshape(-1, WINDOW_SIZE, 1)
y_aug = y_train

# The Test Set (Universal for all ways)
X_test_rnn = X_test.reshape(-1, WINDOW_SIZE, 1)
y_test_final = y_test
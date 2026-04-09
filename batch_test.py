import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
from test_script import load_test_data 

# ============================================================
# HOW TO SWITCH BETWEEN MULTICLASS AND BINARY BATCH TESTING
# ============================================================
# Currently have the testing in BINARY mode
#
# Note to self:
# To go back to MULTICLASS later
# 1) Use multiclass labels from `load_test_data`
# 2) Restore multiclass prediction with `argmax()` only
# 3) Restore `Atrial_F1` and the 5-class report/summary
# 4) If you previously changed the model output to binary, switch it back to
#    5 outputs with softmax and use the multiclass saved model

def run_batch_test(record_list):
    model = tf.keras.models.load_model('models/base_rnn_model_v1.keras')
    results = []

    # Binary output summary.
    print(f"{'Record':<10} | {'Accuracy':<10} | {'Macro F1':<10} | {'Abnormal F1':<12}")
    print("-" * 50)

    for rid in record_list:
        try:
            X_test, y_test = load_test_data(rid)

            # --- MULTICLASS VERSION ---
            # y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

            # --- BINARY VERSION ---
            # Handles either a 1-unit sigmoid model or a 2-unit output model
            y_probs = model.predict(X_test, verbose=0)
            if y_probs.ndim == 1 or (y_probs.ndim == 2 and y_probs.shape[1] == 1):
                y_pred = (y_probs.ravel() >= 0.5).astype(int)
            else:
                y_pred = np.argmax(y_probs, axis=1)
                # If a multiclass model is ever loaded, collapse non-normal classes to abnormal for binary evaluation
                # Failsafes against accidentally loading a multiclass model while still using binary evaluation logic. If the model has more than 2 outputs, we treat all non-zero classes as abnormal.
                if y_probs.shape[1] > 2:
                    y_pred = (y_pred != 0).astype(int)
            
            # Accuracy for general performance overview but uh don't rely on it too much in imbalanced scenarios
            acc = accuracy_score(y_test, y_pred)

            # Macro F1 gives balanced view across normal/abnormal, but can be low if the model struggles with the minority class
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # --- MULTICLASS VERSION ---
            # f1_atrial = f1_score(y_test, y_pred, labels=[4], average='macro', zero_division=0)

            # Binary-specific abnormal score
            # Focus on the model's ability to detect abnormalities, which is the focus for this version of implementation and probably the most clinically relevant aspect
            f1_abnormal = f1_score(y_test, y_pred, labels=[1], average='macro', zero_division=0)

            results.append({
                'Record': rid,
                'Accuracy': acc,
                'Macro_F1': f1_macro,
                'Abnormal_F1': f1_abnormal
                # 'Atrial_F1': f1_atrial # Uncomment if you want to track the atrial-specific performance in a multiclass scenario
            })

            print(f"{rid:<10} | {acc:<10.2f} | {f1_macro:<10.2f} | {f1_abnormal:<12.2f}")
        
        except Exception as e:
            print(f"Skipping {rid}: {e}")

    # Final Summary
    df = pd.DataFrame(results)
    print("\n" + "="*30)
    print(" OVERALL DATABASE PERFORMANCE ")
    print("="*30)
    print(df.mean(numeric_only=True))
    
    return df

# Records with diverse arrhythmias
test_records = ['100', '101', '103', '105', '111', '200', '203', '215']
batch_results = run_batch_test(test_records)
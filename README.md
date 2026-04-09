# Main Files:
- data_extract_resample.py
- rnn_model.py
- genetic_rnn_optimizer.py
- main.py
    - Supporting Files:
        - test_scrip.py
        - batch_test.py

    - Template Files:
        - ga_optimizer.py
        - preprocess.py
    
    - JustInCase Files:
        - main_multiclass.py
        - rnn_model coppy.py
        

# Steps to run the suite
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
py --version (to check)


pip install tensorflow wfdb scikit-learn imbalanced-learn pandas numpy seaborn


Data Extract
    py .\data_extract_resample.py
    (Undersample or Cap + SMOTE)
Test Making a RNN model
    py .\rnn_model.py
    (Shows model architecture with details)
Experimental GA RNN (idk what I'm doing here)
    py .\genetic_rnn_optimizer.py
    (This takes a fat minute and a half so don't run it on your laptop's hardware)
The main thing to run a full suite
    py .\main.py
    
Testing
    py .\test_script.py
    py .\batch_test.py

# List of records you can load in:
100 Series
> Primarily chosen to represent a variety of ECG morphologies (wider range of different looking ones)
- 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124

200 Series
> Rarer but pretty significant arrhythmias
- 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233
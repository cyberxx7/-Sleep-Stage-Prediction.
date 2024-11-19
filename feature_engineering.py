import numpy as np
import pandas as pd

def extract_features(data):
    features = []
    labels = []
    for subject_id in data['labels'].keys():
        heart_rate_data = data['heart_rate'][subject_id]
        motion_data = data['motion'][subject_id]
        step_data = data['steps'][subject_id]
        label_data = data['labels'][subject_id]

        # Ensure data columns are numeric by converting if needed
        heart_rate_data = heart_rate_data.apply(pd.to_numeric, errors='coerce')
        motion_data = motion_data.apply(pd.to_numeric, errors='coerce')
        step_data = step_data.apply(pd.to_numeric, errors='coerce')
        label_data = label_data.apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values to ensure calculations don't fail
        heart_rate_data.dropna(inplace=True)
        motion_data.dropna(inplace=True)
        step_data.dropna(inplace=True)
        label_data.dropna(inplace=True)

        # Example feature extraction (mean heart rate, total acceleration, step count)
        hr_mean = heart_rate_data.iloc[:, 0].mean() if not heart_rate_data.empty else 0
        acc_magnitude = (
            np.sqrt((motion_data.iloc[:, 0]**2 + motion_data.iloc[:, 1]**2 + motion_data.iloc[:, 2]**2)).mean()
            if not motion_data.empty else 0
        )
        step_sum = step_data.iloc[:, 0].sum() if not step_data.empty else 0

        features.append([hr_mean, acc_magnitude, step_sum])
        labels.append(label_data.iloc[:, 0] if not label_data.empty else 0)
    
    return np.array(features), np.array(labels)

import os
import pandas as pd
from config import DATA_DIR

def read_data():
    data = {}
    folders = ['heart_rate', 'motion', 'labels', 'steps']
    
    for folder in folders:
        folder_path = os.path.join(DATA_DIR, folder)
        data[folder] = {}
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                subject_id = filename.split('_')[0]
                data[folder][subject_id] = pd.read_csv(file_path, sep=',', header=None)
    
    return data

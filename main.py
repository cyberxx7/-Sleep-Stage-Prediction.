from data_preprocessing import read_data
from feature_engineering import extract_features
from model_training import train_model
from utils import remove_invalid_samples

if __name__ == '__main__':
    # Step 1: Read Data
    data = read_data()
    
    # Step 2: Feature Extraction
    features, labels = extract_features(data)
    
    # Step 3: Remove Invalid Samples
    features, labels = remove_invalid_samples(features, labels)
    
    # Step 4: Train Model
    train_model(features, labels)
    
    print("Pipeline Completed Successfully!")

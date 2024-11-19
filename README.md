# Sleep Stage Prediction Project

## Project Overview
This project uses the dataset "Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography". The dataset was collected and described by Olivia Walch, Yitong Huang, Daniel Forger, and Cathy Goldstein in their paper "Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device" (Sleep, zsz180, https://doi.org/10.1093/sleep/zsz180).
This project is focused on predicting sleep stages using data from wrist-worn wearable devices such as heart rate, motion, and step count. The data is used to train machine learning and deep learning models to classify sleep stages accurately. This project includes various steps such as data preprocessing, feature extraction, model training, and evaluation.

## Project Structure
The project is structured into multiple Python scripts to modularize each part of the pipeline:

- `data_preprocessing.py`: Reads and preprocesses raw data.
- `feature_engineering.py`: Extracts relevant features from the preprocessed data.
- `model_training.py`: Trains both a Random Forest model and a neural network for sleep stage classification.
- `utils.py`: Contains helper functions for the project, such as data cleaning.
- `config.py`: Contains configuration variables like data directories and model paths.
- `main.py`: The main entry point for running the entire pipeline.

## How It Works
1. **Data Loading and Preprocessing**:
   - Raw data is read from text files containing heart rate, motion, and step data.
   - The `read_data()` function loads data from different folders (`heart_rate`, `motion`, `labels`, and `steps`).
2. **Feature Extraction**:
   - Features like mean heart rate, total acceleration magnitude, and total step count are extracted using `extract_features()`.
   - Non-numeric values are converted, and invalid rows are dropped to clean the data.
3. **Model Training**:
   - Two models are trained:
     - **Random Forest Classifier**: Evaluated using cross-validation and trained to classify sleep stages.
     - **Neural Network**: Built using PyTorch, incorporating dropout for regularization to prevent overfitting.
4. **Evaluation**:
   - Both models are evaluated using metrics like precision, recall, and accuracy.
   - Early stopping is implemented in the neural network to avoid unnecessary training epochs.

## Requirements
This project requires Python and the following Python packages:

```
pandas==1.5.3
numpy==1.24.0
scikit-learn==1.2.0
joblib==1.3.0
torch==2.0.0
matplotlib==3.7.1
```

To install the required packages, run:

```sh
pip install -r requirements.txt
```

## Setting Up the Project
1. **Clone the Repository**:
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Prepare the Data**:
   - Place the data files in the correct folder structure as indicated in `config.py`. The data directory should include subfolders for `heart_rate`, `motion`, `labels`, and `steps`, each containing `.txt` files for each subject.
3. **Run the Pipeline**:
   - Run the `main.py` script to execute the complete data processing and model training pipeline:
   ```sh
   python main.py
   ```

## Running the Project
The entire process can be executed using the `main.py` script, which will:
1. **Read the Data** from the given files.
2. **Extract Features** necessary for model training.
3. **Remove Invalid Samples** to ensure data integrity.
4. **Train Models** (Random Forest and Neural Network).

Once complete, the trained models are saved in the `models/` directory.

## How to Use the Models
- The trained Random Forest model and the scaler are saved as `.pkl` files in the `models/` folder.
- You can load these models using `joblib` to make predictions on new data:
  ```python
  import joblib
  clf = joblib.load('models/sleep_stage_classifier.pkl')
  scaler = joblib.load('models/scaler.pkl')
  ```

## Project Improvements and Next Steps
- **More Diverse Data**: Incorporate a more diverse dataset to improve generalizability.
- **Data Augmentation**: Experiment with data augmentation to address overfitting issues.
- **Model Optimization**: Further optimize hyperparameters for better performance.

## Contributions
Feel free to contribute to this project by creating a pull request or opening an issue. Suggestions for improvements or additional features are always welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from config import MODEL_PATH
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple neural network model
class SleepStageModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SleepStageModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.6)  # Increase dropout rate for stronger regularization
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def train_model(features, labels):
    # Use StratifiedKFold for cross-validation to ensure better balance of classes
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=10, min_samples_leaf=5, random_state=42)

    # Perform cross-validation to assess model performance
    cross_val_scores = cross_val_score(clf, features, labels, cv=stratified_kfold)
    print(f'Cross-Validation Scores: {cross_val_scores}')
    print(f'Mean Cross-Validation Score: {cross_val_scores.mean()}')
    
    # Split dataset for training and testing
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)
    
    # Data Augmentation: Add Gaussian noise to the training data to increase variability
    noise_factor = 0.05
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)  # Clip values to stay within a valid range

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_noisy)
    X_test = scaler.transform(X_test)
    
    # Train the RandomForestClassifier
    clf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save the RandomForest model
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    joblib.dump(clf, os.path.join(MODEL_PATH, 'sleep_stage_classifier.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_PATH, 'scaler.pkl'))
    
    # Train a simple neural network using PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_train.shape[1]
    hidden_size = 50
    num_classes = len(set(labels))
    model = SleepStageModel(input_size, hidden_size, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.02)  # Increased weight decay for stronger regularization
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # Training loop with early stopping
    num_epochs = 50
    best_loss = float('inf')
    early_stop_counter = 0
    patience = 5  # Early stopping after 5 epochs without improvement

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # Implement early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f'Neural Network Accuracy: {accuracy * 100:.2f}%')


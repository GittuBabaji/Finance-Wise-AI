import os
import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the Neural Network class (must match your training script)
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.network(x))

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    print("Scaler loaded successfully!")

# Load the model
input_dim = scaler.n_features_in_  # Get input dimension from scaler
model = FraudDetectionModel(input_dim)
model.load_state_dict(torch.load("fraud_detection_model.pth"))  # or "best_fraud_model.pth"
model.eval()  # Set to evaluation mode
print("Model loaded successfully!")

# Optional: Load LLM for explanations
try:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    llm_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    use_llm = True
    print("LLM loaded successfully!")
except Exception as e:
    print(f"Could not initialize LLM: {e}")
    use_llm = False

# Function to generate fraud explanation (same as training script)
def generate_fraud_explanation(transaction_data):
    if not use_llm:
        return "LLM not available"
    
    transaction_text = (
        f"Amount: ${transaction_data['amount']:.2f}, "
        f"Credit Limit: ${transaction_data['credit_limit']:.2f}, "
        f"Ratio: {transaction_data['amount_ratio']:.2f}, "
        f"Chip Usage: {transaction_data.get('use_chip', 'N/A')}"
    )
    
    inputs = tokenizer(transaction_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = llm_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return f"Fraud probability from LLM: {probs[0][1]:.2f}"

# Function to preprocess new data (adapt based on your training preprocessing)
def preprocess_new_data(df):
    # Apply the same preprocessing as in training
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].apply(lambda x: pd.factorize(x)[0])
    
    # Drop irrelevant columns (same as training)
    X = df.drop([col for col in df.columns if 'id' in col.lower() or 'date' in col.lower()], 
                axis=1, 
                errors='ignore')
    
    # Handle missing values
    X = X.fillna(X.mean())
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    
    # Scale the data
    X_scaled = scaler.transform(X)
    return X_scaled, X

# Function to predict fraud
def predict_fraud(X_scaled, original_data=None):
    X_tensor = torch.FloatTensor(X_scaled)
    with torch.no_grad():
        predictions = model(X_tensor).numpy().flatten()
    
    # Convert predictions to binary (0 or 1) using a threshold
    binary_predictions = (predictions > 0.5).astype(int)
    
    results = []
    for i, (prob, pred) in enumerate(zip(predictions, binary_predictions)):
        result = {
            "probability": prob,
            "prediction": "Fraud" if pred == 1 else "Not Fraud"
        }
        if original_data is not None and use_llm:
            result["explanation"] = generate_fraud_explanation(original_data.iloc[i])
        results.append(result)
    
    return results

# Example: Load and test new data
# Replace "new_transactions.csv" with your actual test data file
new_data = pd.read_csv("new_transactions.csv")  # Your new test data
X_scaled, original_data = preprocess_new_data(new_data)
predictions = predict_fraud(X_scaled, original_data)

# Display predictions
for i, result in enumerate(predictions):
    print(f"Transaction {i+1}:")
    print(f"  Fraud Probability: {result['probability']:.4f}")
    print(f"  Prediction: {result['prediction']}")
    if "explanation" in result:
        print(f"  Explanation: {result['explanation']}")
    print()

# Optional: Save predictions to a file
results_df = pd.DataFrame(predictions)
results_df.to_csv("fraud_predictions.csv", index=False)
print("Predictions saved to 'fraud_predictions.csv'!")
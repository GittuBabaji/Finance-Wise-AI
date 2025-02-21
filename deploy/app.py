import os
import pickle
import torch
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Neural Network class
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

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    print("Scaler loaded successfully!")

# Load model
input_dim = scaler.n_features_in_
model = FraudDetectionModel(input_dim).to(device)
model.load_state_dict(torch.load("fraud_detection_model.pth", map_location=device))
model.eval()
print("Fraud detection model loaded successfully!")

# Load LLM for explanations
try:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    llm_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-sst-2-english"
    ).to(device)
    use_llm = True
    print("LLM loaded successfully!")
except Exception as e:
    print(f"LLM loading failed: {e}")
    use_llm = False

# Function to preprocess input data
def preprocess_input(data):
    df = pd.DataFrame([data])
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].apply(lambda x: pd.factorize(x)[0])

    # Ensure correct features
    expected_features = scaler.feature_names_in_
    X = df.drop(columns=[col for col in df.columns if 'id' in col.lower() or 'date' in col.lower()], errors='ignore')
    
    missing_cols = set(expected_features) - set(X.columns)
    for col in missing_cols:
        X[col] = 0  # Add missing columns

    X = X[expected_features]  # Ensure column order
    X = scaler.transform(X)  # Scale data
    return torch.FloatTensor(X).to(device)

# Function to generate LLM explanations
def generate_fraud_explanation(transaction):
    if not use_llm:
        return "LLM not available"

    text = f"Amount: ${transaction['amount']:.2f}, Credit Limit: ${transaction['credit_limit']:.2f}, " \
           f"Ratio: {transaction['amount_ratio']:.2f}, Chip Usage: {transaction.get('use_chip', 'N/A')}"

    inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = llm_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return f"Fraud probability from LLM: {probs[0][1]:.2f}"

# API endpoint for fraud detection
@app.route('/predict', methods=['POST'])
def predict():
    try:
        transaction = request.json  # Get JSON data
        X_scaled = preprocess_input(transaction)
        
        with torch.no_grad():
            prediction = model(X_scaled).cpu().numpy().flatten()[0]

        fraud_probability = float(prediction)
        is_fraud = "Fraud" if fraud_probability > 0.5 else "Not Fraud"

        response = {
            "probability": fraud_probability,
            "prediction": is_fraud
        }

        # Generate explanation if LLM is available
        if use_llm:
            response["explanation"] = generate_fraud_explanation(transaction)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

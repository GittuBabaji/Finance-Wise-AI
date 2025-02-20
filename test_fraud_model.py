import os
import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Verify pandas and CUDA
print("Status: Verifying pandas import...")
if 'pd' not in globals():
    raise ImportError("Pandas not imported! Please ensure 'import pandas as pd' is present and pandas is installed.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Status: Using device: {device}")

# Define the Neural Network class
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
print("Status: Loading scaler...")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    print("Status: Scaler loaded successfully!")
    print("Scaler expected features:", scaler.feature_names_in_)

# Load the model
input_dim = scaler.n_features_in_
model = FraudDetectionModel(input_dim).to(device)
model.load_state_dict(torch.load("fraud_detection_model.pth"))
model.eval()
print("Status: Model loaded successfully!")

# Load LLM for explanations
try:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    llm_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    ).to(device)
    use_llm = True
    print("Status: LLM loaded successfully!")
except Exception as e:
    print(f"Status: Could not initialize LLM: {e}")
    use_llm = False

# Function to generate fraud explanations in batches
def generate_fraud_explanation_batch(transactions):
    if not use_llm:
        return ["LLM not available"] * len(transactions)
    
    texts = [f"Amount: ${t['amount']:.2f}, Credit Limit: ${t['credit_limit']:.2f}, "
             f"Ratio: {t['amount_ratio']:.2f}, Chip Usage: {t.get('use_chip', 'N/A')}"
             for t in transactions]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = llm_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return [f"Fraud probability from LLM: {p[1]:.2f}" for p in probs]

# Function to preprocess new data
def preprocess_new_data(df):
    print("Status: Starting data preprocessing...")
    expected_features = scaler.feature_names_in_
    
    print(f"Status: Initial row count: {len(df)}")
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].apply(lambda x: pd.factorize(x)[0])
    
    X = df.drop([col for col in df.columns if 'id' in col.lower() or 'date' in col.lower()], 
                axis=1, 
                errors='ignore')
    
    print("Status: Columns in new data before alignment:", X.columns.tolist())
    
    missing_cols = set(expected_features) - set(X.columns)
    extra_cols = set(X.columns) - set(expected_features)
    
    X = X.drop(columns=extra_cols, errors='ignore')
    for col in missing_cols:
        X[col] = 0
    
    X = X[expected_features]
    print("Status: Columns in new data after alignment:", X.columns.tolist())
    
    X = X.fillna(X.mean())
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    print(f"Status: Rows after handling missing values: {len(X)} (dropped {len(df) - len(X)} rows)")
    
    X_scaled = scaler.transform(X)
    print("Status: Data scaling completed")
    return X_scaled, X

# Function to predict fraud with batched LLM
def predict_fraud(X_scaled, original_data=None):
    print(f"Status: Starting predictions for {len(X_scaled)} transactions")
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    batch_size = 1000
    total_samples = len(X_scaled)
    predictions = []
    
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch = X_tensor[i:batch_end]
            batch_preds = model(batch).cpu().numpy().flatten()
            predictions.extend(batch_preds)
            progress = (batch_end / total_samples) * 100
            print(f"Status: Prediction progress: {progress:.1f}% ({batch_end}/{total_samples} transactions processed)")
    
    predictions = np.array(predictions)
    binary_predictions = (predictions > 0.5).astype(int)
    
    results = []
    if original_data is not None and use_llm:
        llm_batch_size = 32
        explanations = []
        total_batches = (len(original_data) + llm_batch_size - 1) // llm_batch_size
        for i in range(0, len(original_data), llm_batch_size):
            batch_data = original_data.iloc[i:i + llm_batch_size].to_dict('records')
            batch_explanations = generate_fraud_explanation_batch(batch_data)
            explanations.extend(batch_explanations)
            batch_num = i // llm_batch_size
            if batch_num % 100 == 0:  # Update every 100 batches (~3200 transactions)
                print(f"Status: LLM processing: {i}/{len(original_data)} transactions ({batch_num}/{total_batches} batches)")
    
    for i, (prob, pred) in enumerate(zip(predictions, binary_predictions)):
        if i % 10000 == 0:
            print(f"Status: Building results: {i}/{len(predictions)} processed")
        if not (0 <= prob <= 1):
            print(f"Status: Warning - Invalid probability at index {i}: {prob}")
        
        result = {
            "probability": prob,
            "prediction": "Fraud" if pred == 1 else "Not Fraud"
        }
        if original_data is not None and use_llm:
            result["explanation"] = explanations[i]
        results.append(result)
    
    print("Status: Prediction generation completed")
    return results

# Main execution
print("Status: Loading new data...")
file_path = "/home/wizard/code/projects/Ai/new_transactions.csv"
if not os.path.exists(file_path):
    print(f"Status: Error - '{file_path}' not found!")
    print("Please place 'new_transactions.csv' in the correct directory or update the path.")
    exit(1)

try:
    new_data = pd.read_csv(file_path)
    print("Status: New data loaded successfully with columns:", new_data.columns.tolist())
except Exception as e:
    print(f"Status: Error loading data - {str(e)}")
    exit(1)

new_data = new_data.rename(columns={"amt": "amount", "cc_num": "credit_card_number"})
print("Status: Starting preprocessing...")
X_scaled, original_data = preprocess_new_data(new_data)

print("Status: Starting fraud prediction...")
predictions = predict_fraud(X_scaled, original_data)

# Display predictions with status
print("Status: Displaying results...")
fraud_count = 0
for i, result in enumerate(predictions):
    print(f"Transaction {i+1}:")
    print(f"  Fraud Probability: {result['probability']:.4f}")
    print(f"  Prediction: {result['prediction']}")
    if "explanation" in result:
        print(f"  Explanation: {result['explanation']}")
    print()
    
    if result['prediction'] == "Fraud":
        fraud_count += 1
    
    if (i + 1) % 100 == 0:
        print(f"Status: Processed {i + 1} transactions ({(i + 1)/len(predictions)*100:.1f}%)")

print(f"Status: Summary - Total transactions: {len(predictions)}, Fraud cases: {fraud_count}")
print(f"Status: Fraud rate: {(fraud_count/len(predictions)*100):.2f}%")

# Save predictions
results_df = pd.DataFrame(predictions)
results_df.to_csv("fraud_predictions.csv", index=False)
print("Status: Predictions saved to 'fraud_predictions.csv'!")

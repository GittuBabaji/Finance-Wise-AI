import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import kagglehub
import glob
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Download dataset
print("Downloading dataset...")
path = kagglehub.dataset_download("computingvictor/transactions-fraud-datasets")
print(f"Path to dataset files: {path}")

# Load all CSV files
csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
print(f"Found {len(csv_files)} CSV files:")
for file in csv_files:
    print(f" - {file}")

# Load all data files
data_dict = {}
for file in csv_files:
    filename = os.path.basename(file).split('.')[0]
    data_dict[filename] = pd.read_csv(file)
    print(f"Loaded {filename} with shape: {data_dict[filename].shape}")

# Extract the dataframes
transactions_df = data_dict.get('transactions_data')
cards_df = data_dict.get('cards_data')
users_df = data_dict.get('users_data')

if transactions_df is None or cards_df is None or users_df is None:
    raise ValueError("Missing one or more required datasets")

# Clean up credit_limit column in cards_df
if 'credit_limit' in cards_df.columns:
    cards_df['credit_limit'] = cards_df['credit_limit'].replace(r'[\$,]', '', regex=True).astype(float)


# Generate fraud indicators based on transaction patterns
def generate_fraud_indicators(transactions_df, cards_df, users_df):
    print("Generating fraud indicators...")

    # Ensure amount is numeric
    if 'amount' in transactions_df.columns and transactions_df['amount'].dtype == object:
        transactions_df['amount'] = transactions_df['amount'].replace(r'[\$,]', '', regex=True).astype(float)
    
    # Merge transactions with cards and users
    df = transactions_df.merge(cards_df, left_on='card_id', right_on='id', suffixes=('', '_card'))
    df = df.merge(users_df, left_on='client_id', right_on='id', suffixes=('', '_user'))
    df.fillna(0, inplace=True)
    # Feature engineering to detect potential fraud
    # 1. Suspicious large amounts relative to credit_limit
    df['amount_ratio'] = df['amount'] / df['credit_limit']
    
    # 2. Transactions without chip when card has chip
    df['suspicious_no_chip'] = ((df['has_chip'] == 'YES') & (df['use_chip'] == 'NO')).astype(int)
    
    # 3. Transactions in different states than user's home state (using zip code as proxy)
    df['zip_first3'] = df['zip'].astype(str).str[:3]
    
    # 4. Create a synthetic fraud label (this is for demonstration - in reality would need more sophisticated rules)
    # Mark as potential fraud if: large amount ratio OR suspicious chip use
    df['is_fraud'] = ((df['amount_ratio'] > 0.5) | (df['suspicious_no_chip'] == 1)).astype(int)
    
    # Keep necessary columns
    df = df.drop(['id_card', 'id_user'], axis=1, errors='ignore')
    return df

# Process the data
df = generate_fraud_indicators(transactions_df, cards_df, users_df)
print(f"Generated dataset with shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean():.2%}")

# Initialize LLM for fraud explanation
print("Initializing LLM for fraud explanation...")
try:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    llm_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    use_llm = True
    print("LLM initialized successfully")
except Exception as e:
    print(f"Could not initialize LLM: {e}")
    print("Proceeding without LLM integration")
    use_llm = False

# Create a function to generate text explanation using LLM
def generate_fraud_explanation(transaction_data):
    if not use_llm:
        return "LLM not available for explanation"
    
    # Convert transaction data to text format for LLM
    transaction_text = f"""m 
    Transaction amount: ${transaction_data['amount']:.2f}
    Credit limit: ${transaction_data['credit_limit']:.2f}
    Amount ratio: {transaction_data['amount_ratio']:.2f}
    Has chip: {transaction_data['has_chip']}
    Used chip: {transaction_data['use_chip']}
    """
    
    inputs = tokenizer(transaction_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = llm_model(**inputs)
    
    # Using softmax to get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    fraud_prob = probs[0][1].item()
    
    if fraud_prob > 0.5:
        explanation = "This transaction appears fraudulent due to "
        if transaction_data['amount_ratio'] > 0.5:
            explanation += f"the unusually high amount ({transaction_data['amount_ratio']:.1f}x) relative to credit limit. "
        if transaction_data['suspicious_no_chip'] == 1:
            explanation += "the card has a chip but wasn't used for this transaction. "
    else:
        explanation = "This transaction appears legitimate. "
    
    return explanation

# Handle missing values
df.fillna(0, inplace=True)

# Convert categorical variables to numerical
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.factorize(df[col])[0]

# Remove ID columns and select features
id_cols = [col for col in df.columns if 'id' in col.lower()]
date_cols = [col for col in df.columns if 'date' in col.lower()]
X = df.drop(['is_fraud'] + id_cols + date_cols, axis=1, errors='ignore')
y = df['is_fraud']
# Replace infinite values with NaN
X = np.where(np.isinf(X), np.nan, X)

# Remove rows with NaN values
X = X[~np.isnan(X).any(axis=1)]
y = y[~np.isnan(X).any(axis=1)]
# First, handle NaN values in X
X = X.dropna()

# Then, update y to match X's new index
y = y.loc[X.index]

# Now you can safely apply the boolean indexing
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
y = y[mask]
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Use a smaller sample for training if dataset is very large
if len(df) > 1000000:
    print("Dataset is large, using a random sample of 100,0000 rows for training")
    sample_indices = np.random.choice(len(df), 1000000, replace=False)
    X = X.iloc[sample_indices]
    y = y.iloc[sample_indices]
    df = df.iloc[sample_indices]

# Split the data
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Custom Dataset
class FraudDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets.values).reshape(-1, 1)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Create dataset and dataloader
train_dataset = FraudDataset(X_train, y_train)
test_dataset = FraudDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Neural Network Model
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 16)
        self.layer4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(16)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        
        x = self.layer4(x)
        return torch.sigmoid(x)

# Initialize model
input_dim = X_train.shape[1]
model = FraudDetectionModel(input_dim)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# Train the model
print("Training model...")
train_model(model, train_loader, criterion, optimizer)

# Evaluation
def evaluate_model(model, test_loader, df_test):
    model.eval()
    correct = 0
    total = 0
    true_positives = 0
    predicted_positives = 0
    actual_positives = 0
    
    explanations = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            predicted_positives += (predicted == 1).sum().item()
            actual_positives += (labels == 1).sum().item()
            
            # Generate explanations for some predicted frauds
            if i < 5:  # Only do this for first few batches for efficiency
                batch_size = inputs.size(0)
                for j in range(batch_size):
                    if predicted[j] == 1:
                        idx = i * batch_size + j
                        if idx < len(df_test):
                            transaction_data = df_test.iloc[idx].to_dict()
                            explanation = generate_fraud_explanation(transaction_data)
                            explanations.append({
                                'transaction_id': idx,
                                'predicted': 'Fraud',
                                'actual': 'Fraud' if labels[j] == 1 else 'Legitimate',
                                'explanation': explanation
                            })
                            if len(explanations) >= 10:
                                break
    
    accuracy = correct / total
    precision = true_positives / max(predicted_positives, 1)
    recall = true_positives / max(actual_positives, 1)
    f1 = 2 * (precision * recall) / max((precision + recall), 1e-8)
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # Save explanations to file
    if explanations:
        with open('fraud_explanations.json', 'w') as f:
            json.dump(explanations, f, indent=2)
        print(f"Saved {len(explanations)} fraud explanations to fraud_explanations.json")

print("Evaluating model...")
evaluate_model(model, test_loader, df_test)

# Save the model
torch.save(model.state_dict(), 'fraud_detection_model.pth')
print("Model saved as fraud_detection_model.pth")

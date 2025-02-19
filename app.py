from flask import Flask, request, jsonify
import torch
import pandas as pd
import numpy as np
from fraud_detection import FraudDetectionModel, generate_fraud_explanation, StandardScaler

# Load the trained model
model_path = r"C:\Users\Asus\OneDrive\Desktop\AI\fraud_detection_model.pth"
scaler_path = r"C:\Users\Asus\OneDrive\Desktop\AI\scaler.pkl"

app = Flask(__name__)

# Load model
input_dim = 10  # Adjust based on features used
model = FraudDetectionModel(input_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load the scaler
import pickle
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])
        # Preprocess input (convert to numeric, scale)
        df_scaled = scaler.transform(df)
        tensor_input = torch.FloatTensor(df_scaled)

        # Preprocess input (convert to numeric, scale)
        df_scaled = scaler.transform(df)
        tensor_input = torch.FloatTensor(df_scaled)

        # Get prediction
        with torch.no_grad():
            output = model(tensor_input)
        
        prediction = int(output.item() > 0.5)
        explanation = generate_fraud_explanation(data) if prediction == 1 else "Transaction is normal."

        return jsonify({"fraud_prediction": prediction, "explanation": explanation})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

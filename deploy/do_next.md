To integrate `app.py` (Flask backend) with your **website frontend**, follow these steps:  

---

## **1️⃣ Expose Your AI Model as an API using Flask**  
The Flask backend (`app.py`) will:  
✅ Load the trained fraud detection model (`fraud_detection_model.pth`).  
✅ Process incoming requests (from the website).  
✅ Return fraud prediction results in **JSON format**.  

Here's an example `app.py` to deploy your AI model as a REST API:  

### **📌 Create `app.py` (Flask Backend)**
```python
from flask import Flask, request, jsonify
import torch
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Load trained model & scaler
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define model class
class FraudDetectionModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.network(x))

# Load model
input_dim = scaler.n_features_in_
model = FraudDetectionModel(input_dim).to(device)
model.load_state_dict(torch.load("fraud_detection_model.pth", map_location=device))
model.eval()

# Function to preprocess new transaction data
def preprocess_data(data):
    df = pd.DataFrame([data])  # Convert JSON input to DataFrame
    df = df.rename(columns={"amt": "amount", "cc_num": "credit_card_number"})  # Standardize column names
    
    # Ensure feature alignment
    missing_cols = set(scaler.feature_names_in_) - set(df.columns)
    extra_cols = set(df.columns) - set(scaler.feature_names_in_)
    df = df.drop(columns=extra_cols, errors="ignore")
    for col in missing_cols:
        df[col] = 0

    df = df[scaler.feature_names_in_]  # Ensure column order
    df = df.fillna(df.mean())  # Handle missing values
    X_scaled = scaler.transform(df)  # Apply scaler
    return torch.FloatTensor(X_scaled).to(device)

# API Route for Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Receive JSON input
        X_scaled = preprocess_data(data)  # Preprocess data
        
        with torch.no_grad():
            probability = model(X_scaled).cpu().numpy().flatten()[0]  # Model inference
        
        response = {
            "probability": round(float(probability), 4),
            "prediction": "Fraud" if probability > 0.5 else "Not Fraud"
        }
        return jsonify(response)  # Return JSON response
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
```
---

## **2️⃣ Run the Flask Server**
After saving the script as `app.py`, run:
```sh
python app.py
```
This starts the API at `http://127.0.0.1:5000/predict`.

---

## **3️⃣ Connect the Flask API to Your Website**  
### **Frontend (JavaScript Example)**
If your website is built with **HTML + JavaScript**, you can use **AJAX (Fetch API)** to send transaction data to the Flask API and display the results.

📌 **Modify `script.js` in your website**
```javascript
async function checkFraud() {
    let transaction = {
        amount: parseFloat(document.getElementById("amount").value),
        credit_limit: parseFloat(document.getElementById("credit_limit").value),
        amount_ratio: parseFloat(document.getElementById("amount_ratio").value),
        use_chip: document.getElementById("use_chip").value
    };

    let response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(transaction)
    });

    let result = await response.json();
    document.getElementById("output").innerHTML = `
        <strong>Prediction:</strong> ${result.prediction} <br>
        <strong>Fraud Probability:</strong> ${result.probability * 100}%
    `;
}
```

---

### **4️⃣ Create an HTML Form to Send Data**
📌 **Modify `index.html`**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Detection</title>
    <script defer src="script.js"></script>
</head>
<body>
    <h2>Fraud Detection System</h2>
    
    <label>Transaction Amount ($):</label>
    <input type="number" id="amount"><br>

    <label>Credit Limit ($):</label>
    <input type="number" id="credit_limit"><br>

    <label>Amount Ratio:</label>
    <input type="number" step="0.01" id="amount_ratio"><br>

    <label>Chip Used (yes/no):</label>
    <input type="text" id="use_chip"><br>

    <button onclick="checkFraud()">Check Fraud</button>

    <h3>Result:</h3>
    <p id="output"></p>
</body>
</html>
```
---

## **5️⃣ Deploy Flask App Online**
If you want to host your backend **so your website can access it globally**, you can use **Render, Railway, Hugging Face Spaces, or Fly.io**.

For **Render**:
1. Upload your `app.py`, model files, and `requirements.txt` (list dependencies like `flask`, `torch`, `pandas`).
2. Deploy as a **Flask Web Service**.
3. Get the **public API URL** (e.g., `https://fraud-detect-api.onrender.com`).
4. Update your frontend JavaScript:
   ```javascript
   let response = await fetch("https://fraud-detect-api.onrender.com/predict", { ... });
   ```

---

## **🔥 Final Result**
✅ **User inputs transaction details** on the website  
✅ **Website sends request to Flask API**  
✅ **AI Model predicts fraud & sends response**  
✅ **Website displays fraud probability & prediction**  

---

### 🚀 **Next Steps**
- Do you want a **React.js frontend** instead of vanilla HTML+JS?  
- Would you like **FastAPI instead of Flask** for better performance?  
- Need help with **deploying Flask on the cloud**?  

Let me know! 🚀

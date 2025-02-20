# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import openai
import os

app = Flask(__name__)
CORS(app)
x=input("your-api-key-here")
# Initialize OpenAI client with API key
openai_client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY', x)
)

# Investment options
investment_options = [
    "Fixed Deposits (FDs)", "Public Provident Fund (PPF)", "Employee Provident Fund (EPF)",
    "Sovereign Gold Bonds (SGBs)", "National Savings Certificate (NSC)", "Debt Mutual Funds",
    "Hybrid Mutual Funds", "Index Funds", "Exchange-Traded Funds (ETFs)", "Corporate Bonds",
    "Recurring Deposits (RDs)", "Post Office Monthly Income Scheme (POMIS)",
    "Equity Mutual Funds", "Direct Stock Market Investment", "Equity-Linked Savings Scheme (ELSS)",
    "Real Estate Investment Trusts (REITs)", "Infrastructure Investment Trusts (InvITs)",
    "Commodity Market (MCX - Gold, Silver, Oil)", "Cryptocurrency (Bitcoin, Ethereum, etc.)",
    "Stock Derivatives (Futures & Options)", "National Pension System (NPS)",
    "Unit Linked Insurance Plans (ULIPs)", "Gold ETFs & Digital Gold",
    "Life Insurance & Health Insurance"
]

# System prompt
system_prompt = """
You are a professional financial advisor AI with deep expertise in investment options. Your goal is to understand the user's request thoroughly and provide detailed, well-structured, and accurate advice. When asked about investments, consider factors like risk tolerance, investment horizon, liquidity needs, and financial goals. Use the following list of investment options to tailor your responses:  
""" + "\n- " + "\n- ".join(investment_options) + """
Provide clear explanations, pros and cons, and examples where applicable. Be empathetic, professional, and concise unless the user requests in-depth details and be zesty.
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('message', '')
        
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        # Updated OpenAI API call
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return jsonify({
            'response': response.choices[0].message.content.strip()
        })
    
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

import openai

# Leave space for your OpenAI API key (replace 'your-api-key-here' with your actual key when ready)
openai.api_key = 'your-api-key-here'

# List of investment options as provided
investment_options = [
    "Fixed Deposits (FDs)",
    "Public Provident Fund (PPF)",
    "Employee Provident Fund (EPF)",
    "Sovereign Gold Bonds (SGBs)",
    "National Savings Certificate (NSC)",
    "Debt Mutual Funds",
    "Hybrid Mutual Funds",
    "Index Funds",
    "Exchange-Traded Funds (ETFs)",
    "Corporate Bonds",
    "Recurring Deposits (RDs)",
    "Post Office Monthly Income Scheme (POMIS)",
    "Equity Mutual Funds",
    "Direct Stock Market Investment",
    "Equity-Linked Savings Scheme (ELSS)",
    "Real Estate Investment Trusts (REITs)",
    "Infrastructure Investment Trusts (InvITs)",
    "Commodity Market (MCX - Gold, Silver, Oil)",
    "Cryptocurrency (Bitcoin, Ethereum, etc.)",
    "Stock Derivatives (Futures & Options)",
    "National Pension System (NPS)",
    "Unit Linked Insurance Plans (ULIPs)",
    "Gold ETFs & Digital Gold",
    "Life Insurance & Health Insurance"
]

# System prompt to define the AI's behavior
system_prompt = """
You are a professional financial advisor AI with deep expertise in investment options. Your goal is to understand the user's request thoroughly and provide detailed, well-structured, and accurate advice. When asked about investments, consider factors like risk tolerance, investment horizon, liquidity needs, and financial goals. Use the following list of investment options to tailor your responses:  
""" + "\n- " + "\n- ".join(investment_options) + """
Provide clear explanations, pros and cons, and examples where applicable. Be empathetic, professional, and concise unless the user requests in-depth details and be zesty.
"""

# Function to interact with OpenAI API
def get_ai_response(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Replace with a free-tier model or your preferred model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=500,  # Adjust based on how detailed you want responses
            temperature=0.7   # Balanced creativity and professionalism
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}. Please ensure your API key is set and valid."

# Example interaction loop
def main():
    print("Welcome to your Professional Investment Advisor AI!")
    print("Ask me anything about investments or financial planning.")
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break
        
        response = get_ai_response(user_input)
        print("\nAI Response:", response)

# Run the script
if __name__ == "__main__":
    main()

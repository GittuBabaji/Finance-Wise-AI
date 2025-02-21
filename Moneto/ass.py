import os
from groq import Groq

# # Option 1: Use Hardcoded API Key (Not Recommended)
# client = Groq(api_key=api_key)  # Directly pass the key

# Option 2: Use Environment Variable (Recommended)
# Ensure you have exported 'GROQ_API_KEY' before running the script
client = Groq(api_key=os.environ.get("API_KEY"))


# Define system role more concisely

user_input = input("You: ")  # Get user input

if user_input.lower() in ["exit", "quit"]:  # Exit condition
    print("Exiting chat...")
    exit()



chat_completion = client.chat.completions.create(
    messages=[
        {
        "role": "system",
        "content": (
            "You are FinanceWise AI,named Moneto, a financial expert who evaluates user decisions. "
            "If the question is finance-related, provide a direct, concise answer, "
            "stating whether the user's thinking is correct and guiding them if necessary. "
            "For non-financial questions, creatively relate them to finance and answer briefly. "
            "Always end non-financial answers(dont write this in very general topics ) with: 'I am not an expert in that field of interest.Lets talk about finance :) '"
            "Always give short and crisp answers which are to the point."
            "Be responsible for other's hard earned money"
        )
        },
        {
            "role": "user",
            "content": user_input,
        },

    ],
    model="llama-3.3-70b-versatile", 

    temperature=0.3,

    top_p=0.7,

    stop=None,

    stream=False,
)



print(chat_completion.choices[0].message.content)


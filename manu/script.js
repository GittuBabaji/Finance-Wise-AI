const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');

// Function to call the Flask API for fraud detection
async function checkFraud(data) {
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (result.error) {
            return `Error: ${result.error}`;
        }
        return `Fraud Prediction: ${result.fraud_prediction === 1 ? 'Fraudulent' : 'Normal'}\nExplanation: ${result.explanation}`;
    } catch (error) {
        return `Error connecting to fraud detection service: ${error.message}`;
    }
}

// Function to call the Flask API for investment advice
async function getAdvice(message) {
    try {
        const response = await fetch('http://localhost:5000/advice', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        const result = await response.json();
        if (result.error) {
            return `Error: ${result.error}`;
        }
        return result.response;
    } catch (error) {
        return `Error connecting to advice service: ${error.message}`;
    }
}

// Determine response based on user input
async function getAIResponse(message) {
    const lowerMessage = message.toLowerCase();

    if (lowerMessage.includes('fraud') || lowerMessage.includes('check transaction')) {
        // Simulate a transaction data object (replace with actual parsing if needed)
        const sampleTransaction = {
            feature1: 100, feature2: 50, feature3: 25, feature4: 10, feature5: 5,
            feature6: 0, feature7: 15, feature8: 20, feature9: 30, feature10: 0
        };
        return await checkFraud(sampleTransaction);
    } else {
        // Default to investment advice via OpenAI
        return await getAdvice(message);
    }
}

// Add message to chat
function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(isUser ? 'user-message' : 'ai-message');
    messageDiv.textContent = content;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Send message and get response
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    addMessage(message, true);
    const aiResponse = await getAIResponse(message);
    setTimeout(() => addMessage(aiResponse), 500); // Simulate delay
    userInput.value = '';
}

// Event listeners
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Initial welcome message
addMessage('Welcome to your Investment Advisor AI! Ask about investments or fraud detection with zest!');

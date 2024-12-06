// Enable/Disable send button based on input field
function toggleButton() {
    const inputField = document.getElementById("user-input");
    const sendButton = document.getElementById("send-btn");
    sendButton.disabled = inputField.value.trim() === "";
}

// Function to format time
function getFormattedTime() {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, "0");
    const minutes = String(now.getMinutes()).padStart(2, "0");
    return `${hours}:${minutes}`;
}

// Function to send message
async function sendMessage() {
    const inputField = document.getElementById("user-input");
    const chatWindow = document.getElementById("chat-window");
    const userMessage = inputField.value.trim();

    if (userMessage === "") return;

    // Add user message to chat window
    const userBubble = document.createElement("div");
    userBubble.className = "chat-bubble user-message";
    userBubble.textContent = userMessage;
    const timestamp = document.createElement("div");
    timestamp.className = "timestamp";
    timestamp.textContent = getFormattedTime();
    userBubble.appendChild(timestamp);
    chatWindow.appendChild(userBubble);
    inputField.value = "";
    toggleButton();

    // Scroll to the bottom
    chatWindow.scrollTop = chatWindow.scrollHeight;

    // Send request to backend
    const response = await fetch(`/interact/${docId}`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: userMessage }),
    });

    const result = await response.json();

    // Add bot response to chat window
    const botBubble = document.createElement("div");
    botBubble.className = "chat-bubble bot-message";
    botBubble.textContent = result.response;
    const botTimestamp = document.createElement("div");
    botTimestamp.className = "timestamp";
    botTimestamp.textContent = getFormattedTime();
    botBubble.appendChild(botTimestamp);
    chatWindow.appendChild(botBubble);

    // Scroll to the bottom
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

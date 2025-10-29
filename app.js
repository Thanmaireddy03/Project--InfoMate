// Configuration
const CONFIG = {
    apiBaseUrl: 'http://localhost:8080',
    defaultModel: 'gemma3:4b',
    defaultSession: 'default',
    temperature: 0.6,
    systemPrompt: 'You are a concise assistant. Respond ONLY in English. Do not greet, do not introduce yourself, and do not mention gemma3/OpenAI or any affiliations. Answer directly and briefly unless the user asks for details.',
    maxContext: 8192,
    maxPredict: 256
};

// State
let currentModel = CONFIG.defaultModel;
let currentSession = CONFIG.defaultSession;
let isProcessing = false;
let chatHistory = [];

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const modelSelect = document.getElementById('modelSelect');
const sessionSelect = document.getElementById('sessionSelect');
const resetBtn = document.getElementById('resetBtn');
const newSessionBtn = document.getElementById('newSessionBtn');
const settingsBtn = document.getElementById('settingsBtn');
const settingsModal = document.getElementById('settingsModal');
const closeModal = document.getElementById('closeModal');
const saveSettings = document.getElementById('saveSettings');
const cancelSettings = document.getElementById('cancelSettings');
const statusText = document.getElementById('statusText');
const temperature = document.getElementById('temperature');
const tempValue = document.getElementById('tempValue');
const systemPrompt = document.getElementById('systemPrompt');
const maxContext = document.getElementById('maxContext');
const maxPredict = document.getElementById('maxPredict');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    loadSession();
    updateTempValue();
}

function setupEventListeners() {
    // Send button
    sendBtn.addEventListener('click', sendMessage);
    
    // Enter to send, Shift+Enter for new line
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Model and session changes
    modelSelect.addEventListener('change', (e) => {
        currentModel = e.target.value;
        statusText.textContent = `Model changed to ${currentModel}`;
    });

    sessionSelect.addEventListener('change', async (e) => {
        currentSession = e.target.value;
        await loadSession();
        statusText.textContent = `Switched to session: ${currentSession}`;
    });

    // Reset conversation
    resetBtn.addEventListener('click', async () => {
        if (confirm('Are you sure you want to reset this conversation?')) {
            await resetConversation();
        }
    });

    // New session
    newSessionBtn.addEventListener('click', () => {
        const sessionName = prompt('Enter a name for the new session:');
        if (sessionName && sessionName.trim()) {
            currentSession = sessionName.trim();
            sessionSelect.value = currentSession;
            resetConversation();
        }
    });

    // Settings modal
    settingsBtn.addEventListener('click', openSettingsModal);
    closeModal.addEventListener('click', closeSettingsModal);
    cancelSettings.addEventListener('click', closeSettingsModal);
    saveSettings.addEventListener('click', saveSettingsData);
    
    // Temperature slider
    temperature.addEventListener('input', updateTempValue);
    
    // Close modal on outside click
    window.addEventListener('click', (e) => {
        if (e.target === settingsModal) {
            closeSettingsModal();
        }
    });
}

function updateTempValue() {
    tempValue.textContent = temperature.value;
}

function openSettingsModal() {
    settingsModal.style.display = 'block';
    temperature.value = CONFIG.temperature;
    updateTempValue();
    systemPrompt.value = CONFIG.systemPrompt;
    maxContext.value = CONFIG.maxContext;
    maxPredict.value = CONFIG.maxPredict;
}

function closeSettingsModal() {
    settingsModal.style.display = 'none';
}

function saveSettingsData() {
    CONFIG.temperature = parseFloat(temperature.value);
    CONFIG.systemPrompt = systemPrompt.value;
    CONFIG.maxContext = parseInt(maxContext.value);
    CONFIG.maxPredict = parseInt(maxPredict.value);
    statusText.textContent = 'Settings saved';
    closeSettingsModal();
}

async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message || isProcessing) {
        return;
    }

    // Handle commands
    if (message.startsWith('/')) {
        await handleCommand(message);
        messageInput.value = '';
        return;
    }

    // Clear input and add user message
    messageInput.value = '';
    addMessage('user', message);
    isProcessing = true;
    updateSendButton();

    // Show typing indicator
    const typingId = showTypingIndicator();

    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: currentModel,
                session: currentSession,
                message: message,
                temperature: CONFIG.temperature,
                maxContext: CONFIG.maxContext,
                maxPredict: CONFIG.maxPredict,
                systemPrompt: CONFIG.systemPrompt
            })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        removeTypingIndicator(typingId);
        
        if (data.error) {
            showError(data.error);
        } else {
            addMessage('assistant', data.response);
        }
    } catch (error) {
        removeTypingIndicator(typingId);
        showError(`Failed to send message: ${error.message}. Make sure the Go backend is running on ${CONFIG.apiBaseUrl}`);
        console.error('Error:', error);
    } finally {
        isProcessing = false;
        updateSendButton();
    }
}

async function handleCommand(command) {
    statusText.textContent = 'Processing command...';
    
    const parts = command.trim().split(' ');
    const cmd = parts[0];
    const args = parts.slice(1).join(' ');

    try {
        switch (cmd) {
            case '/help':
                showHelp();
                break;
            case '/reset':
                if (confirm('Are you sure you want to reset this conversation?')) {
                    await resetConversation();
                }
                break;
            case '/history':
                showHistory();
                break;
            case '/model':
                if (args) {
                    currentModel = args;
                    modelSelect.value = currentModel;
                    statusText.textContent = `Model set to ${currentModel}`;
                }
                break;
            case '/save':
                await saveSession();
                break;
            case '/exit':
                // Just show info in web context
                addMessage('system', 'To exit, just close the browser tab.');
                break;
            default:
                showError(`Unknown command: ${cmd}. Use /help to see available commands.`);
        }
    } catch (error) {
        showError(`Command failed: ${error.message}`);
    }
}

function showHelp() {
    const helpText = `/help - Show available commands
/reset - Clear conversation context
/history - Show conversation history
/model [TAG] - Switch model (e.g. /model llama3)
/save - Save current conversation
/exit - Exit (close browser tab)`;

    addMessage('assistant', helpText);
}

function showHistory() {
    if (chatHistory.length === 0) {
        addMessage('system', 'No conversation history yet.');
        return;
    }

    let historyText = 'Recent conversation:\n\n';
    const recentMessages = chatHistory.slice(-10);
    
    recentMessages.forEach(msg => {
        historyText += `${msg.role.toUpperCase()}: ${msg.content}\n\n`;
    });

    addMessage('assistant', historyText);
}

async function loadSession() {
    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/session/${currentSession}`);
        if (response.ok) {
            const data = await response.json();
            chatHistory = data.messages || [];
            
            // Display history
            chatMessages.innerHTML = '';
            chatHistory.slice(1).forEach(msg => { // Skip system message
                if (msg.role !== 'system') {
                    addMessage(msg.role, msg.content, false);
                }
            });
            
            if (chatHistory.length <= 1) {
                // Show welcome message if empty
                chatMessages.innerHTML = `
                    <div class="welcome-message">
                        <h2>Welcome to Info-mate</h2>
                        <p>Start a conversation by typing a message below.</p>
                        <p class="info-text">💡 Tip: Use commands like /help, /reset, or /history</p>
                    </div>
                `;
            }
            
            statusText.textContent = `Loaded session: ${currentSession}`;
        }
    } catch (error) {
        console.error('Error loading session:', error);
        statusText.textContent = 'Could not load session';
    }
}

async function resetConversation() {
    try {
        await fetch(`${CONFIG.apiBaseUrl}/reset/${currentSession}`, {
            method: 'POST'
        });
        chatHistory = [];
        chatMessages.innerHTML = '';
        chatMessages.innerHTML = `
            <div class="welcome-message">
                <h2>Welcome to Info-mate</h2>
                <p>Start a conversation by typing a message below.</p>
                <p class="info-text">💡 Tip: Use commands like /help, /reset, or /history</p>
            </div>
        `;
        statusText.textContent = 'Conversation reset';
    } catch (error) {
        showError(`Failed to reset: ${error.message}`);
    }
}

async function saveSession() {
    statusText.textContent = 'Session saved';
}

function addMessage(role, content, addToHistory = true) {
    if (addToHistory) {
        chatHistory.push({ role, content });
    }

    // Remove welcome message if present
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString();
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    contentDiv.appendChild(timeDiv);
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.parentElement.scrollTop = chatMessages.parentElement.scrollHeight;
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = 'typing-indicator-' + Date.now();
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const typing = document.createElement('div');
    typing.className = 'message-content typing-indicator';
    typing.innerHTML = '<span></span><span></span><span></span>';
    
    typingDiv.appendChild(avatar);
    typingDiv.appendChild(typing);
    chatMessages.appendChild(typingDiv);
    
    chatMessages.parentElement.scrollTop = chatMessages.parentElement.scrollHeight;
    
    return typingDiv.id;
}

function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = `❌ Error: ${message}`;
    chatMessages.appendChild(errorDiv);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
    
    chatMessages.parentElement.scrollTop = chatMessages.parentElement.scrollHeight;
}

function updateSendButton() {
    sendBtn.disabled = isProcessing;
    if (isProcessing) {
        sendBtn.innerHTML = '<span>Processing...</span>';
        statusText.textContent = 'Processing request...';
    } else {
        sendBtn.innerHTML = '<span>Send</span><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
        statusText.textContent = 'Ready';
    }
}


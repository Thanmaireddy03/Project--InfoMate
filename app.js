import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Send, Upload, Settings, ToggleLeft, ToggleRight, Trash2, X, Download } from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8080/api';

// Simple markdown parser function
function parseMarkdown(text) {
  if (!text) return '';
  
  return text
    // Bold text **text** -> <strong>text</strong>
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    // Italic text *text* -> <em>text</em>
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    // Bullet points * item -> <li>item</li>
    .replace(/^\* (.+)$/gm, '<li>$1</li>')
    // Numbered lists 1. item -> <li>item</li>
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    // Line breaks
    .replace(/\n/g, '<br/>')
    // Wrap consecutive <li> elements in <ul>
    .replace(/(<li>.*<\/li>)/gs, (match) => {
      const items = match.split('</li>').filter(item => item.trim()).map(item => item + '</li>');
      return `<ul>${items.join('')}</ul>`;
    });
}

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [session, setSession] = useState('default');
  const [sessions, setSessions] = useState(['default']);
  const [ragEnabled, setRagEnabled] = useState(true);
  const [model, setModel] = useState('gemma3:4b');
  const [availableModels] = useState(['gemma3:4b', 'llama3.1:8b', 'deepseek-r1:1.5b']);
  const [pdfFile, setPdfFile] = useState(null);
  const [pdfStatus, setPdfStatus] = useState('');
  const [context, setContext] = useState({});
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const loadSessions = async () => {
    try {
      const response = await axios.get(`${API_BASE}/sessions`);
      setSessions(response.data.sessions || ['default']);
    } catch (error) {
      console.error('Failed to load sessions:', error);
    }
  };

  const loadContext = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE}/context?session=${session}`);
      setContext(response.data);
      setRagEnabled(response.data.ragEnabled || false);
      setModel(response.data.model || 'gemma3:4b');
    } catch (error) {
      console.error('Failed to load context:', error);
    }
  }, [session]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    loadSessions();
    loadContext();
    loadHistory();
  }, [loadContext]);

  const createSession = async () => {
    const newSession = prompt('Enter session name:');
    if (newSession && !sessions.includes(newSession)) {
      try {
        await axios.post(`${API_BASE}/sessions`, { name: newSession });
        setSessions([...sessions, newSession]);
        setSession(newSession);
        setMessages([]);
        await loadContext();
      } catch (error) {
        console.error('Failed to create session:', error);
        alert('Failed to create session');
      }
    }
  };

  const switchSession = async (newSession) => {
    setSession(newSession);
    await loadHistory(newSession);
    await loadContext();
  };

  const loadHistory = async (sessionName) => {
    try {
      const response = await axios.get(`${API_BASE}/history?session=${sessionName || session}`);
      const msgs = response.data.messages || [];
      // Filter out system messages for display
      const displayMsgs = msgs.filter(m => m.role !== 'system');
      setMessages(displayMsgs);
    } catch (error) {
      console.error('Failed to load history:', error);
      setMessages([]);
    }
  };

  const deleteSession = async (sessionName) => {
    if (sessionName === 'default') {
      alert('Cannot delete default session');
      return;
    }
    if (!window.confirm(`Are you sure you want to delete session "${sessionName}"?`)) {
      return;
    }
    try {
      await axios.delete(`${API_BASE}/sessions?name=${sessionName}`);
      setSessions(sessions.filter(s => s !== sessionName));
      if (session === sessionName) {
        setSession('default');
        await loadHistory('default');
        await loadContext();
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
      alert('Failed to delete session');
    }
  };

  const clearChat = async () => {
    if (!window.confirm('Are you sure you want to clear all chat history?')) {
      return;
    }
    try {
      await axios.post(`${API_BASE}/clear`, { session });
      setMessages([]);
    } catch (error) {
      console.error('Failed to clear chat:', error);
      alert('Failed to clear chat');
    }
  };

  const exportChat = () => {
    const chatData = {
      session,
      date: new Date().toISOString(),
      messages: messages
    };
    const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-${session}-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const toggleRag = async () => {
    try {
      const newRagState = !ragEnabled;
      await axios.post(`${API_BASE}/rag`, { 
        session, 
        enabled: newRagState 
      });
      setRagEnabled(newRagState);
    } catch (error) {
      console.error('Failed to toggle RAG:', error);
    }
  };

  const changeModel = async (newModel) => {
    try {
      await axios.post(`${API_BASE}/model`, { 
        session, 
        model: newModel 
      });
      setModel(newModel);
    } catch (error) {
      console.error('Failed to change model:', error);
    }
  };

  const uploadPdf = async () => {
    if (!pdfFile) return;
    
    const formData = new FormData();
    formData.append('file', pdfFile);
    formData.append('session', session);

    try {
      setIsLoading(true);
      setPdfStatus('Uploading PDF...');
      const response = await axios.post(`${API_BASE}/pdf`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPdfStatus(response.data.message);
      setRagEnabled(true);
      await loadContext();
    } catch (error) {
      console.error('Failed to upload PDF:', error);
      const serverMessage = error.response?.data;
      if (serverMessage) {
        if (typeof serverMessage === 'string') {
          setPdfStatus(serverMessage);
        } else if (serverMessage.error) {
          setPdfStatus(serverMessage.error);
        } else {
          setPdfStatus('Failed to upload PDF');
        }
      } else {
        setPdfStatus('Failed to upload PDF');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const messageToSend = inputMessage;
    const userMessage = { role: 'user', content: messageToSend };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/chat`, {
        session,
        message: messageToSend,
        model,
        ragEnabled
      });

      const assistantMessage = { role: 'assistant', content: response.data.response };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Failed to send message:', error);
      const errorMessage = { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app">
      <div className="sidebar">
        <div className="sidebar-header">
          <h2>Info Mate</h2>
          <button className="new-session-btn" onClick={createSession}>
            <Settings size={16} />
            New Session
          </button>
        </div>

        <div className="sessions">
          <h3>Sessions</h3>
          {sessions.map(s => (
            <div key={s} className="session-item-wrapper">
              <button
                className={`session-item ${s === session ? 'active' : ''}`}
                onClick={() => switchSession(s)}
              >
                {s}
              </button>
              {s !== 'default' && (
                <button
                  className="delete-session-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteSession(s);
                  }}
                  title="Delete session"
                >
                  <X size={14} />
                </button>
              )}
            </div>
          ))}
        </div>

        <div className="settings">
          <h3>Settings</h3>
          
          <div className="setting-item">
            <label>Model:</label>
            <select value={model} onChange={(e) => changeModel(e.target.value)}>
              {availableModels.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          <div className="setting-item">
            <label>RAG:</label>
            <button className="toggle-btn" onClick={toggleRag}>
              {ragEnabled ? <ToggleRight size={20} /> : <ToggleLeft size={20} />}
              {ragEnabled ? 'ON' : 'OFF'}
            </button>
          </div>

          <div className="pdf-upload">
            <h4>PDF Upload</h4>
            <input
              type="file"
              accept=".pdf"
              onChange={(e) => setPdfFile(e.target.files[0])}
              id="pdf-input"
              style={{ display: 'none' }}
            />
            <label htmlFor="pdf-input" className="upload-btn">
              <Upload size={16} />
              Choose PDF
            </label>
            {pdfFile && (
              <div className="pdf-info">
                <span>{pdfFile.name}</span>
                <button onClick={uploadPdf} disabled={isLoading}>
                  Upload
                </button>
              </div>
            )}
            {pdfStatus && (
              <div className="pdf-status">{pdfStatus}</div>
            )}
          </div>

          <div className="context-info">
            <h4>Context</h4>
            <div className="context-details">
              <div>RAG: {ragEnabled ? 'ON' : 'OFF'}</div>
              <div>PDF: {context.pdfName || 'None'}</div>
              <div>Chunks: {context.chunks || 0}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="main-content">
        <div className="chat-header">
          <h1>AI Chat</h1>
          <div className="header-actions">
            <div className="session-info">Session: {session}</div>
            <div className="header-buttons">
              {messages.length > 0 && (
                <>
                  <button className="header-btn" onClick={exportChat} title="Export chat">
                    <Download size={18} />
                  </button>
                  <button className="header-btn" onClick={clearChat} title="Clear chat">
                    <Trash2 size={18} />
                  </button>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome">
              <h2>Welcome to Info Mate!</h2>
              <p>Start a conversation or upload a PDF to get started with RAG.</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div key={index} className={`message ${message.role}`}>
                <div className="message-content">
                  {message.role === 'assistant' ? (
                    <div dangerouslySetInnerHTML={{ __html: parseMarkdown(message.content) }} />
                  ) : (
                    message.content
                  )}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="message assistant">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <div className="input-wrapper">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here..."
              disabled={isLoading}
              rows="1"
            />
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="send-btn"
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

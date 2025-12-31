import React, { useState, useEffect, useRef } from 'react';
import './FloatingChatbot.css';

const FloatingChatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const toggleChatbot = () => {
    setIsOpen(!isOpen);
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8001/api/chat/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: inputValue,
          session_id: sessionId
        })
      });

      const data = await response.json();

      if (data.session_id && !sessionId) {
        setSessionId(data.session_id);
      }

      const botMessage = {
        id: Date.now() + 1,
        text: data.response,
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        contextUsed: data.context_used,
        isInScope: data.is_in_scope
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error. Please make sure the backend server is running.',
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        isInScope: false
      };
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

  const clearChat = () => {
    setMessages([]);
    setSessionId(null);
  };

  return (
    <>
      {/* Floating Chat Button */}
      <div className={`floating-chat-button ${isOpen ? 'hidden' : ''}`} onClick={toggleChatbot}>
        <svg
          width="28"
          height="28"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
        <span className="chat-badge">AI</span>
      </div>

      {/* Floating Chat Window */}
      {isOpen && (
        <div className="floating-chat-window">
          <div className="floating-chat-header">
            <div className="chat-header-content">
              <div className="chat-avatar">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="3"/>
                  <path d="M12 1v6m0 6v6m6-9h-6m-6 0h6"/>
                </svg>
              </div>
              <div>
                <h3>Physical AI Assistant</h3>
                <p className="status-text">● Online</p>
              </div>
            </div>
            <button className="close-button" onClick={toggleChatbot}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>

          <div className="floating-chat-messages">
            {messages.length === 0 ? (
              <div className="chat-welcome">
                <div className="welcome-icon">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                  </svg>
                </div>
                <h4>👋 Hello! I'm your Physical AI Assistant</h4>
                <p>Ask me anything about:</p>
                <ul>
                  <li>🤖 Physical AI & Robotics</li>
                  <li>🔧 Sensor Systems & Fusion</li>
                  <li>👥 Human-Robot Interaction</li>
                  <li>🧠 Embodied Intelligence</li>
                </ul>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`chat-message ${message.sender}-msg`}
                >
                  <div className="message-bubble">
                    <div className="message-text">{message.text}</div>
                    <div className="message-time">{message.timestamp}</div>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="chat-message bot-msg">
                <div className="message-bubble">
                  <div className="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="floating-chat-input">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about Physical AI & Robotics..."
              rows="1"
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="send-btn"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
          </div>

          {messages.length > 0 && (
            <button className="clear-all-btn" onClick={clearChat}>
              Clear Chat
            </button>
          )}
        </div>
      )}
    </>
  );
};

export default FloatingChatbot;

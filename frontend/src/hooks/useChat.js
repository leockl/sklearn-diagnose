/**
 * Custom hook for managing chat state and interactions.
 *
 * Following rerender-* best practices for optimal re-render performance.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { sendChatMessage, clearChatHistory, fetchWelcomeMessage } from '../services/api';

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [welcomeLoaded, setWelcomeLoaded] = useState(false);

  // Following rerender-use-ref-transient-values: use ref for scroll tracking
  const messagesEndRef = useRef(null);

  // Scroll to bottom when messages change
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Load welcome message on mount
  useEffect(() => {
    if (!welcomeLoaded) {
      fetchWelcomeMessage()
        .then((welcomeMsg) => {
          setMessages([
            {
              role: 'assistant',
              content: welcomeMsg,
            },
          ]);
          setWelcomeLoaded(true);
        })
        .catch((err) => {
          console.error('Failed to fetch welcome message:', err);
          setError('Failed to load welcome message');
        });
    }
  }, [welcomeLoaded]);

  // Following rerender-functional-setstate: use functional updates for stable callbacks
  const sendMessage = useCallback(async (messageText) => {
    if (!messageText.trim()) return;

    setError(null);
    setIsLoading(true);

    // Optimistically add user message
    const userMessage = { role: 'user', content: messageText };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const data = await sendChatMessage(messageText);

      // Add assistant response
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.response },
      ]);
    } catch (err) {
      console.error('Chat error:', err);
      setError('Failed to send message. Please try again.');

      // Remove the optimistically added message on error
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearChat = useCallback(async () => {
    try {
      await clearChatHistory();
      setMessages([]);
      setWelcomeLoaded(false);
      setError(null);
    } catch (err) {
      console.error('Clear chat error:', err);
      setError('Failed to clear chat history.');
    }
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearChat,
    messagesEndRef,
  };
}

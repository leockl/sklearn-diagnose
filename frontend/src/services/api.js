/**
 * API client for communicating with the FastAPI backend.
 *
 * Following bundle-defer-third-party: API calls are only made when needed.
 */

const API_BASE = '/api';

/**
 * Fetch the diagnosis report.
 */
export async function fetchReport() {
  const response = await fetch(`${API_BASE}/report`);
  if (!response.ok) {
    throw new Error('Failed to fetch report');
  }
  return response.json();
}

/**
 * Fetch the welcome message.
 */
export async function fetchWelcomeMessage() {
  const response = await fetch(`${API_BASE}/welcome`);
  if (!response.ok) {
    throw new Error('Failed to fetch welcome message');
  }
  const data = await response.json();
  return data.message;
}

/**
 * Send a chat message and get a response.
 *
 * @param {string} message - The user's message
 * @returns {Promise<{response: string, history: Array}>}
 */
export async function sendChatMessage(message) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    throw new Error('Failed to send message');
  }

  return response.json();
}

/**
 * Clear the chat history.
 */
export async function clearChatHistory() {
  const response = await fetch(`${API_BASE}/clear`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error('Failed to clear chat history');
  }

  return response.json();
}

/**
 * Check server health.
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) {
    throw new Error('Server health check failed');
  }
  return response.json();
}

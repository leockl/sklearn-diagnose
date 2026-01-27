/**
 * Main chat interface component with input and controls.
 *
 * Following rerender-move-effect-to-event: interaction logic in event handlers.
 */

import { useState, useCallback } from 'react';
import { Send, Trash2, AlertCircle } from 'lucide-react';

export function ChatInterface({ onSendMessage, onClearChat, isLoading, error }) {
  const [input, setInput] = useState('');

  // Following rerender-functional-setstate: stable callback with functional update
  const handleSubmit = useCallback(
    (e) => {
      e.preventDefault();
      if (input.trim() && !isLoading) {
        onSendMessage(input.trim());
        setInput('');
      }
    },
    [input, isLoading, onSendMessage]
  );

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e);
      }
    },
    [handleSubmit]
  );

  return (
    <div className="border-t border-gray-200 bg-white p-4">
      {/* Error display */}
      {error && (
        <div className="mb-3 flex items-center gap-2 px-4 py-3 bg-red-50 border border-red-200 rounded-lg">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
          <p className="text-sm text-red-800">{error}</p>
        </div>
      )}

      {/* Input form */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <div className="flex-1 relative">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask me anything about your model's diagnosis..."
            disabled={isLoading}
            rows={1}
            className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
            style={{ minHeight: '52px', maxHeight: '120px' }}
          />
        </div>

        {/* Send button */}
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Send className="w-5 h-5" />
          <span className="font-medium">Send</span>
        </button>

        {/* Clear button */}
        <button
          type="button"
          onClick={onClearChat}
          disabled={isLoading}
          className="px-4 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors disabled:bg-gray-100 disabled:cursor-not-allowed"
          title="Clear chat history"
        >
          <Trash2 className="w-5 h-5" />
        </button>
      </form>

      {/* Hint text */}
      <p className="mt-2 text-xs text-gray-500 text-center">
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  );
}

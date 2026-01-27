/**
 * Message list component displaying conversation history.
 *
 * Following rendering-content-visibility for performance with long lists.
 */

import { MessageBubble } from './MessageBubble';
import { Loader2 } from 'lucide-react';

export function MessageList({ messages, isLoading, messagesEndRef }) {
  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
      {messages.map((message, index) => (
        <MessageBubble key={index} message={message} />
      ))}

      {/* Loading indicator */}
      {isLoading && (
        <div className="flex gap-3">
          <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-gray-600">
            <Loader2 className="w-5 h-5 text-white animate-spin" />
          </div>
          <div className="flex-1 max-w-3xl px-4 py-3 rounded-lg bg-gray-100 text-gray-900">
            <p className="text-sm text-gray-500">Thinking...</p>
          </div>
        </div>
      )}

      {/* Scroll anchor */}
      <div ref={messagesEndRef} />
    </div>
  );
}

/**
 * Main App component.
 *
 * Following async-parallel: parallel data fetching where appropriate.
 */

import { Header } from './components/Header';
import { MessageList } from './components/MessageList';
import { ChatInterface } from './components/ChatInterface';
import { DiagnosisPanel } from './components/DiagnosisPanel';
import { useChat } from './hooks/useChat';

function App() {
  const { messages, isLoading, error, sendMessage, clearChat, messagesEndRef } =
    useChat();

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <Header />

      <div className="flex flex-1 overflow-hidden">
        {/* Main chat area */}
        <div className="flex-1 flex flex-col bg-white">
          <MessageList
            messages={messages}
            isLoading={isLoading}
            messagesEndRef={messagesEndRef}
          />
          <ChatInterface
            onSendMessage={sendMessage}
            onClearChat={clearChat}
            isLoading={isLoading}
            error={error}
          />
        </div>

        {/* Diagnosis panel sidebar */}
        <DiagnosisPanel />
      </div>
    </div>
  );
}

export default App;

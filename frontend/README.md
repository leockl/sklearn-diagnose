# sklearn-diagnose Chatbot Frontend

React + Vite frontend for the sklearn-diagnose interactive chatbot.

## Development

Install dependencies:
```bash
npm install
```

Start the development server:
```bash
npm run dev
```

The frontend will be available at http://localhost:8001 and will proxy API requests to the FastAPI backend at http://localhost:8000.

## Build for Production

```bash
npm run build
```

The production build will be output to the `dist/` directory.

## Architecture

- **React 18** with hooks
- **Vite** for fast development and optimized builds
- **Tailwind CSS** for styling
- **React Markdown** for rendering LLM responses
- **Lucide React** for icons

## Components

- `App.jsx` - Main application layout
- `Header.jsx` - Application header
- `ChatInterface.jsx` - Message input and controls
- `MessageList.jsx` - Conversation history display
- `MessageBubble.jsx` - Individual message component
- `DiagnosisPanel.jsx` - Sidebar showing diagnosis summary

## Custom Hooks

- `useChat.js` - Manages chat state and API interactions

## API Client

- `services/api.js` - API client for FastAPI backend communication

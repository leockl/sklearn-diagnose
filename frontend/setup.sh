#!/bin/bash
# Setup script for the frontend

echo "Setting up sklearn-diagnose chatbot frontend..."

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed. Please install Node.js and npm first."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
npm install

echo ""
echo "Setup complete! You can now run:"
echo "  npm run dev    - Start development server"
echo "  npm run build  - Build for production"
echo ""

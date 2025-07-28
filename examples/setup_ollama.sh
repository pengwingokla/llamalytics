#!/bin/bash

echo "=== Ollama Setup Script ==="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Please install from: https://ollama.ai/download"
    exit 1
fi

echo "✓ Ollama is installed"

# Start Ollama service (if not running)
echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!
sleep 3

# Pull recommended models
echo "Pulling recommended models..."
echo "This may take several minutes depending on your internet connection"

models=("llama3.2" "phi3")

for model in "${models[@]}"; do
    echo "Pulling $model..."
    if ollama pull "$model"; then
        echo "✓ Successfully pulled $model"
    else
        echo "✗ Failed to pull $model"
    fi
done

echo ""
echo "=== Setup Complete ==="
echo "Available commands:"
echo "  python ollama_setup.py          # Test basic integration"
echo "  python ollama_mcp_server.py     # Start MCP server"
echo ""
echo "To stop Ollama: kill $OLLAMA_PID"
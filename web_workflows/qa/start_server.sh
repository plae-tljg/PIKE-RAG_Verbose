#!/bin/bash
# Start the web-based QA Workflow server

echo "🚀 Starting PikeRAG QA Web Workflow Server..."
echo ""

# Check if virtual environment exists
if [ ! -d "/home/lkm/Pictures/PIKE-RAG/test_env" ]; then
    echo "❌ Virtual environment not found at /home/lkm/Pictures/PIKE-RAG/test_env"
    exit 1
fi

# Activate virtual environment
source /home/lkm/Pictures/PIKE-RAG/test_env/bin/activate

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask..."
    pip install flask
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the web_workflows/qa directory
cd "$SCRIPT_DIR"

echo "✅ Virtual environment activated"
echo "📂 Working directory: $SCRIPT_DIR"
echo ""
echo "🌐 Starting Flask server on http://localhost:5000"
echo ""
echo "To use the interface:"
echo "  1. Open http://localhost:5000 in your browser"
echo "  2. Load your config (e.g., examples/earthquakes/configs/qa.yml)"
echo "  3. Click on questions to see the workflow visualization"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Flask app
python app.py


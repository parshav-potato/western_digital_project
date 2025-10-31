#!/bin/bash

# Quick Start Script for PDF RAPTOR Demo

echo "🚀 PDF RAPTOR Demo - Quick Start"
echo "=================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from template..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "📝 Please edit .env and add your API keys:"
    echo "   - OPENAI_API_KEY (for OpenAI)"
    echo "   - GOOGLE_API_KEY (for Gemini)"
    echo "   - Set LLM_PROVIDER to either 'openai' or 'gemini'"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check if dependencies are installed
if [ ! -d .venv ]; then
    echo "📦 Installing dependencies with uv..."
    uv sync
    echo "✅ Dependencies installed"
    echo ""
fi

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Place your PDF file in the data/ directory"
echo "   2. Open the notebook:"
echo "      jupyter notebook notebooks/raptor_demo.ipynb"
echo "   3. Follow the instructions in the notebook"
echo ""
echo "📚 Documentation: See README.md for more details"

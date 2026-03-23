#!/bin/bash

# GovGig AI Python Backend - Quick Start Script

set -e

echo "======================================"
echo "  GovGig AI - Python Backend Setup"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Install dependencies
echo ""
echo "${YELLOW}Installing dependencies...${NC}"
if [ -f "src/requirements.txt" ]; then
    python3 -m pip install -q -r src/requirements.txt
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    echo -e "${RED}✗${NC} src/requirements.txt not found!"
    exit 1
fi

# Check for .env file
echo ""
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found!${NC}"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env with your actual values before running!${NC}"
    echo ""
    echo "Required variables:"
    echo "  - OPENAI_API_KEY"
    echo "  - PG_PASSWORD"
    echo "  - JWT_SECRET_KEY"
    echo ""
    read -p "Press Enter to edit .env now or Ctrl+C to exit..."
    ${EDITOR:-nano} .env
else
    echo -e "${GREEN}✓${NC} .env file found"
fi

# Test database connection
echo ""
echo "Testing database connection..."
python3 -c "from src.db.connection import test_connection; import sys; sys.exit(0 if test_connection() else 1)" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Database connection successful"
else
    echo -e "${RED}✗${NC} Database connection failed!"
    echo "Please check your database configuration in .env"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run the application
echo ""
echo "======================================"
echo "  Starting GovGig AI Backend"
echo "======================================"
echo ""
echo "Server will start on: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/api/v1/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start server with auto-reload
python3 -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

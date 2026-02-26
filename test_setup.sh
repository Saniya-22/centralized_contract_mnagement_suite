#!/bin/bash

# Test script to verify setup

set -e

echo "Testing GovGig AI Backend Setup"
echo "================================"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
fi

# Test 1: Import core modules
echo "[1/5] Testing core imports..."
./venv/bin/python3 -c "from src.config import settings; from src.state.graph_state import GovGigState" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Core imports successful"
else
    echo "✗ Core imports failed"
    exit 1
fi

# Test 2: Database connection
echo ""
echo "[2/5] Testing database connection..."
./venv/bin/python3 -c "from src.db.connection import test_connection; import sys; sys.exit(0 if test_connection() else 1)" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Database connection successful"
else
    echo "✗ Database connection failed"
    echo "Check your .env file and database setup"
fi

# Test 3: OpenAI API key
echo ""
echo "[3/5] Testing OpenAI API key..."
./venv/bin/python3 -c "from src.config import settings; from openai import OpenAI; client = OpenAI(api_key=settings.OPENAI_API_KEY); client.models.list()" 2>&1 > /dev/null
if [ $? -eq 0 ]; then
    echo "✓ OpenAI API key valid"
else
    echo "✗ OpenAI API key invalid or not set"
fi

# Test 4: Agent initialization
echo ""
echo "[4/5] Testing agent initialization..."
./venv/bin/python3 -c "from src.agents.data_retrieval import DataRetrievalAgent; agent = DataRetrievalAgent()" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Agents initialized successfully"
else
    echo "✗ Agent initialization failed"
fi

# Test 5: Orchestrator
echo ""
echo "[5/5] Testing orchestrator..."
./venv/bin/python3 -c "from src.agents.orchestrator import GovGigOrchestrator; orch = GovGigOrchestrator()" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Orchestrator initialized successfully"
else
    echo "✗ Orchestrator initialization failed"
fi

echo ""
echo "================================"
echo "Setup verification complete!"
echo "================================"

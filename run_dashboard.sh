#!/bin/bash
echo "🚀 Starting GovGig Premium Dashboard..."
source venv/bin/activate
PYTHONPATH=. ./venv/bin/streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

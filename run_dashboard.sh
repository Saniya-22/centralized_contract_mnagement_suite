#!/bin/bash
# If your backend runs on a different port (e.g. 8001), run:
#   API_BASE_URL=http://localhost:8001 bash run_dashboard.sh
export API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
echo "🚀 Starting GovGig Premium Dashboard... (API: $API_BASE_URL)"
source venv/bin/activate
pip install -q -r dashboard/requirements.txt
PYTHONPATH=. ./venv/bin/streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

import streamlit as st
import httpx
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime
import json
import os
import uuid
import asyncio
import websockets
from jose import jwt

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/api/v1/query"
WS_URL = "ws://localhost:8000/ws/chat"
REQUEST_TIMEOUT_SECONDS = float(os.getenv("DASHBOARD_REQUEST_TIMEOUT", "120"))
ST_TITLE = "GovGig AI - Regulatory Command Center"
PRIMARY_COLOR = "#00D1FF"
SECONDARY_COLOR = "#7000FF"
BG_COLOR = "#F0F2F6"  # Lighter background for better contrast
ACCENT_COLOR = "#0E1117"

st.set_page_config(
    page_title=ST_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Glassmorphism & Premium Look) ---
st.markdown(f"""
    <style>
    /* Main Background with a more premium gradient */
    .stApp {{
        background: linear-gradient(135deg, #1A1C2C 0%, #4A192C 50%, #1A1C2C 100%);
        background-attachment: fixed;
    }}
    
    /* Glassmorphism Cards with better contrast */
    [data-testid="stMetric"] {{
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    }}
    
    .stMarkdown div[data-testid="stMarkdownContainer"] blockquote {{
        background: rgba(255, 255, 255, 0.05);
        border-left: 5px solid {PRIMARY_COLOR};
        border-radius: 8px;
        padding: 12px 24px;
        color: #E0E0E0;
    }}
    
    /* Chat bubbles styling - Premium Glassmorphism */
    .user-msg {{
        background: rgba(0, 209, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 18px;
        border-radius: 20px 20px 0 20px;
        border: 1px solid rgba(0, 209, 255, 0.3);
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        color: white;
    }}
    
    .ai-msg {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(8px);
        padding: 18px;
        border-radius: 20px 20px 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        color: #F8F9FA;
    }}
    
    /* Performance Section */
    .perf-card {{
        background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 20px;
        margin: 10px 0;
    }}
    
    /* Document Card */
    .doc-card {{
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 18px;
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }}
    .doc-card:hover {{
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.1);
        border-color: {PRIMARY_COLOR};
    }}
    
    /* Hide default Streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{background: transparent !important;}}
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def generate_internal_token():
    """Generate a valid JWT token for internal dashboard use."""
    try:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode = {
            "sub": "internal-dashboard-user",
            "exp": expire,
            "name": "Dashboard User",
            "role": "admin"
        }
        return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    except Exception:
        # Fallback if settings/jose not available in the dashboard environment
        return None

# --- CONFIG PATH FIX ---
import sys
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

# Import settings if possible, otherwise use dummy values for token generation
try:
    from src.config import settings
    from datetime import timedelta
except ImportError:
    # Fallback if src is still not in path
    class DummySettings:
        JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default_secret")
        JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        ACCESS_TOKEN_EXPIRE_MINUTES = 60
    settings = DummySettings()
    from datetime import timedelta

def create_gauge(value, title, unit="s", max_val=15):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'color': 'white'}},
        number = {'suffix': unit, 'font': {'color': PRIMARY_COLOR}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': PRIMARY_COLOR},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 5], 'color': 'rgba(0, 255, 0, 0.1)'},
                {'range': [5, 10], 'color': 'rgba(255, 255, 0, 0.1)'},
                {'range': [10, 15], 'color': 'rgba(255, 0, 0, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        height=220,
        margin=dict(l=30, r=30, t=50, b=20)
    )
    return fig

# --- INITIAL STATE ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if 'last_perf' not in st.session_state:
    st.session_state.last_perf = {"latency": 0.0, "confidence": 0.0}

# --- SIDEBAR: Telemetry & Config ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/normani-avatars/main/avatars/female/12.png", width=60)
    st.title("System Nexus")
    st.write(f"Logged in as: **Pilot User**")
    
    st.divider()
    
    # Telemetry Gauges
    st.subheader("Real-time Performance")
    latency_gauge = create_gauge(st.session_state.last_perf['latency'], "Latency", "s")
    st.plotly_chart(latency_gauge, use_container_width=True, key="latency_chart")
    
    st.divider()
    
    st.subheader("Session Intelligence")
    st.info(f"Connected to GovGig Backend v2.4\nStatus: Online ✅")
    
    if st.button("Reset Session Memory", type="secondary", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    

# --- MAIN UI ---
col_main, col_evidence = st.columns([0.65, 0.35])

with col_main:
    st.markdown(f"## <span style='color:{PRIMARY_COLOR};'>Gov</span>Gig Insight AI", unsafe_allow_html=True)
    st.markdown("---")

    # Chat History Display
    chat_container = st.container(height=500, border=False)
    with chat_container:
        for chat in st.session_state.history:
            if chat['role'] == 'user':
                st.markdown(f'<div class="user-msg"><b>You:</b><br>{chat["text"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-msg"><b>AI Assistant:</b><br>{chat["text"]}</div>', unsafe_allow_html=True)
                if 'documents' in chat and chat['documents']:
                    with st.expander("Show Linked Regulations & Evidence"):
                        for doc in chat['documents']:
                            st.markdown(f"""
                            <div class="doc-card">
                                <small style='color:{PRIMARY_COLOR}'>{doc.get('regulation_type', 'UNK')} Section {doc.get('section', 'N/A')}</small><br>
                                <b>Source:</b> {doc.get('source', 'Unknown File')}<br>
                                <p style='font-size: 0.85rem;'>{doc.get('content', '')[:200]}...</p>
                            </div>
                            """, unsafe_allow_html=True)

    # Input Area
    prompt = st.chat_input("Ask a regulatory question (e.g., 'What is FAR 52.219-8?')")

    if prompt:
        # Add user msg to state
        st.session_state.history.append({"role": "user", "text": prompt})
        
        # ── STREAMING LOGIC (WebSockets) ──
        async def stream_query():
            t0 = time.perf_counter()
            token = generate_internal_token()
            
            clean_history = [
                {"role": h["role"], "text": h.get("text", "")}
                for h in st.session_state.history[:-1]
            ]
            
            try:
                async with websockets.connect(WS_URL, ping_timeout=REQUEST_TIMEOUT_SECONDS) as ws:
                    # 1. Authenticate & Query
                    payload = {
                        "token": token,
                        "query": prompt,
                        "history": clean_history,
                        "thread_id": st.session_state.thread_id,
                        "cot": False
                    }
                    await ws.send(json.dumps(payload))
                    
                    # 2. Setup UI placeholders
                    full_response = ""
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                    
                    # 3. Stream loop
                    while True:
                        msg_raw = await ws.recv()
                        msg = json.loads(msg_raw)
                        
                        if msg["type"] == "token":
                            full_response += msg["data"]
                            message_placeholder.markdown(full_response + "▌")
                        
                        elif msg["type"] == "complete":
                            res_data = msg["data"]
                            latency = time.perf_counter() - t0
                            
                            # Final update for this message
                            message_placeholder.markdown(full_response)
                            
                            st.session_state.last_perf = {
                                "latency": round(latency, 2),
                                "confidence": res_data.get('confidence', 0.85)
                            }
                            
                            st.session_state.history.append({
                                "role": "ai", 
                                "text": full_response, 
                                "documents": res_data.get('documents', []),
                                "agent_path": res_data.get('agent_path', [])
                            })
                        
                        elif msg["type"] == "done":
                            break
                        
                        elif msg["type"] == "error":
                            st.error(f"Backend Error: {msg['data'].get('message')}")
                            break
                            
                    return True

            except Exception as e:
                st.error(f"WebSocket Connection Failed: {str(e)}")
                return False

        # Run the async stream
        with st.status("🔍 Analyzing documents (Streaming)...", expanded=True) as status:
            success = asyncio.run(stream_query())
            if success:
                status.update(label="✅ Analysis Completed", state="complete", expanded=False)
                st.rerun()
            else:
                status.update(label="❌ Connection Failed", state="error")

with col_evidence:
    st.markdown("### 🗃️ Evidence Viewer")
    st.write("Click on a citation in the chat to view the original text snippet here.")
    
    # Display top documents from last AI response
    if st.session_state.history and st.session_state.history[-1]['role'] == 'ai':
        last_ai = st.session_state.history[-1]
        docs = last_ai.get('documents') or []
        if docs:
            for i, doc in enumerate(docs[:3]):
                with st.container(border=True):
                    st.markdown(f"**Rank {doc.get('rank', i+1)}: {doc.get('regulation_type', 'UNK')} {doc.get('section', '')}**")
                    st.caption(f"File: {doc.get('source', 'N/A')}")
                    st.info(doc.get('content', ''))
                    st.progress(float(doc.get('score', 0)) if doc.get('score') else 0.0, text=f"Relevance: {float(doc.get('score', 0))*100:.1f}%")
        else:
            st.info("No evidence retrieved for this query.")
    else:
        st.info("Retrieve information to see evidence breakdown here.")

# --- FOOTER ---
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: grey;'>GovGig AI Command Center © {datetime.now().year} | Pilot Testing Phase v2.4</p>", unsafe_allow_html=True)


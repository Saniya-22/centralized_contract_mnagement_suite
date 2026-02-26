import streamlit as st
import httpx
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime
import json

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/api/v1/query"
ST_TITLE = "GovGig AI - Regulatory Command Center"
PRIMARY_COLOR = "#00D1FF"
SECONDARY_COLOR = "#7000FF"
BG_COLOR = "#0E1117"

st.set_page_config(
    page_title=ST_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Glassmorphism & Premium Look) ---
st.markdown(f"""
    <style>
    /* Main Background */
    .stApp {{
        background: radial-gradient(circle at 50% 50%, #1a1e2e 0%, #0e1117 100%);
    }}
    
    /* Glassmorphism Cards */
    [data-testid="stMetric"] {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }}
    
    .stMarkdown div[data-testid="stMarkdownContainer"] blockquote {{
        background: rgba(255, 255, 255, 0.03);
        border-left: 5px solid {PRIMARY_COLOR};
        border-radius: 5px;
        padding: 10px 20px;
    }}
    
    /* Chat bubbles styling */
    .user-msg {{
        background: rgba(0, 209, 255, 0.12);
        padding: 15px;
        border-radius: 15px 15px 0 15px;
        border: 1px solid rgba(0, 209, 255, 0.2);
        margin-bottom: 15px;
    }}
    
    .ai-msg {{
        background: rgba(112, 0, 255, 0.08);
        padding: 15px;
        border-radius: 15px 15px 15px 0;
        border: 1px solid rgba(112, 0, 255, 0.2);
        margin-bottom: 15px;
    }}
    
    /* Performance Section */
    .perf-card {{
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0) 100%);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 20px;
        margin: 10px 0;
    }}
    
    /* Document Card */
    .doc-card {{
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        transition: transform 0.2s;
    }}
    .doc-card:hover {{
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.06);
    }}
    
    /* Hide default Streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{background: transparent !important;}}
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
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
    
    st.metric("Avg Rank Accuracy", f"{st.session_state.last_perf['confidence']*100:.1f}%")
    
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
        
        # Call API
        with st.status("🔍 Analyzing documents...", expanded=True) as status:
            t0 = time.perf_counter()
            try:
                response = httpx.post(API_URL, json={
                    "query": prompt,
                    "history": st.session_state.history[:-1], # pass history
                    "cot": True
                }, timeout=30.0)
                
                res_data = response.json()
                latency = time.perf_counter() - t0
                
                # Update agent path status
                if 'agent_path' in res_data:
                    for step in res_data['agent_path']:
                        st.write(f"✅ {step}")
                
                # Update state
                st.session_state.last_perf = {
                    "latency": round(latency, 2),
                    "confidence": res_data.get('confidence', 0.85)
                }
                
                ai_text = res_data.get('response', 'No response received.')
                docs = res_data.get('documents', [])
                
                st.session_state.history.append({
                    "role": "ai", 
                    "text": ai_text, 
                    "documents": docs,
                    "agent_path": res_data.get('agent_path', [])
                })
                
                status.update(label="✅ Analysis Completed", state="complete", expanded=False)
                st.rerun()

            except Exception as e:
                status.update(label="❌ Error in Retrieval", state="error")
                st.error(f"Backend Error: {str(e)}")

with col_evidence:
    st.markdown("### 🗃️ Evidence Evidence Viewer")
    st.write("Click on a citation in the chat to view the original text snippet here.")
    
    # Display top documents from last AI response
    if st.session_state.history and st.session_state.history[-1]['role'] == 'ai':
        last_ai = st.session_state.history[-1]
        if 'documents' in last_ai:
            for i, doc in enumerate(last_ai['documents'][:3]):
                with st.container(border=True):
                    st.markdown(f"**Rank {doc.get('rank', i+1)}: {doc.get('regulation_type', 'UNK')} {doc.get('section', '')}**")
                    st.caption(f"File: {doc.get('source', 'N/A')}")
                    st.info(doc.get('content', ''))
                    st.progress(float(doc.get('score', 0)) if doc.get('score') else 0.0, text=f"Relevance: {float(doc.get('score', 0))*100:.1f}%")
        else:
            st.warning("No linked documents found for the last query.")
    else:
        st.info("Retrieve information to see evidence breakdown here.")

# --- FOOTER ---
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: grey;'>GovGig AI Command Center © {datetime.now().year} | Pilot Testing Phase v2.4</p>", unsafe_allow_html=True)

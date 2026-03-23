import streamlit as st
import httpx
import plotly.graph_objects as go
import time
import json
import os
import uuid
import websockets
from jose import jwt
import sys
import asyncio
from datetime import datetime, timedelta

# --- CONFIGURATION ---
# Set API_BASE_URL to match your backend (e.g. http://localhost:8001 if backend runs on 8001)
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_URL = f"{API_BASE}/api/v1/query"
FEEDBACK_URL = f"{API_BASE}/api/v1/feedback"
LOGIN_URL = f"{API_BASE}/api/v1/auth/login"
SIGNUP_URL = f"{API_BASE}/api/v1/auth/signup"
THREADS_URL = f"{API_BASE}/api/v1/chat/threads"
HISTORY_URL = f"{API_BASE}/api/v1/chat/history"
WS_URL = (
    API_BASE.replace("http://", "ws://").replace("https://", "wss://")
) + "/ws/chat"
REQUEST_TIMEOUT_SECONDS = float(os.getenv("DASHBOARD_REQUEST_TIMEOUT", "120"))
ST_TITLE = "GovGig AI - Regulatory Command Center"
PRIMARY_COLOR = "#00D1FF"
SECONDARY_COLOR = "#7000FF"
BG_COLOR = "#F0F2F6"  # Lighter background for better contrast
ACCENT_COLOR = "#0E1117"

st.set_page_config(
    page_title=ST_TITLE, page_icon="🤖", layout="wide", initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Glassmorphism & Premium Look) ---
st.markdown(
    f"""
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
""",
    unsafe_allow_html=True,
)

# --- CONFIG PATH FIX ---

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

# Import settings if possible, otherwise use dummy values for token generation
try:
    from src.config import settings
except ImportError:
    # Fallback if src is still not in path (align with backend default 24h)
    class DummySettings:
        JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
        if not JWT_SECRET_KEY:
            raise RuntimeError("JWT_SECRET_KEY environment variable is required")
        JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        ACCESS_TOKEN_EXPIRE_MINUTES = int(
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")
        )
        APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

    settings = DummySettings()
    from datetime import timedelta


# --- HELPER FUNCTIONS ---
def generate_internal_token():
    """Generate a valid JWT token for internal dashboard use."""
    try:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
        to_encode = {
            "sub": "internal-dashboard-user",
            "exp": expire,
            "name": "Dashboard User",
            "role": "admin",
        }
        return jwt.encode(
            to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )
    except Exception:
        # Fallback if settings/jose not available in the dashboard environment
        return None


def create_gauge(value, title, unit="s", max_val=15):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 20, "color": "white"}},
            number={"suffix": unit, "font": {"color": PRIMARY_COLOR}},
            gauge={
                "axis": {
                    "range": [None, max_val],
                    "tickwidth": 1,
                    "tickcolor": "white",
                },
                "bar": {"color": PRIMARY_COLOR},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 2,
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, 5], "color": "rgba(0, 255, 0, 0.1)"},
                    {"range": [5, 10], "color": "rgba(255, 255, 0, 0.1)"},
                    {"range": [10, 15], "color": "rgba(255, 0, 0, 0.1)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=220,
        margin=dict(l=30, r=30, t=50, b=20),
    )
    return fig


# --- INITIAL STATE ---
if "history" not in st.session_state:
    st.session_state.history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "last_perf" not in st.session_state:
    st.session_state.last_perf = {"latency": 0.0, "confidence": 0.0}
if "feedback_sent" not in st.session_state:
    st.session_state.feedback_sent = set()
if "thread_list" not in st.session_state:
    st.session_state.thread_list = []
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "auth_view" not in st.session_state:
    st.session_state.auth_view = "login"  # "login" or "signup"

# --- AUTH GATE: User cannot query without logging in ---
if not st.session_state.access_token:
    st.markdown(
        f"## <span style='color:{PRIMARY_COLOR};'>Gov</span>Gig Insight AI",
        unsafe_allow_html=True,
    )
    st.markdown(
        "You must **log in** or **sign up** to access the AI modules. You cannot submit queries without an account."
    )
    st.markdown("---")

    # Sign up and Log in buttons
    btn_col1, btn_col2, _ = st.columns([1, 1, 2])
    with btn_col1:
        if st.button(
            "Log in", type="primary", use_container_width=True, key="btn_show_login"
        ):
            st.session_state.auth_view = "login"
            st.rerun()
    with btn_col2:
        if st.button(
            "Sign up", type="secondary", use_container_width=True, key="btn_show_signup"
        ):
            st.session_state.auth_view = "signup"
            st.rerun()

    st.markdown("---")
    if st.session_state.auth_view == "login":
        st.subheader("Log in")
        with st.form("login_form"):
            login_email = st.text_input(
                "Email (username)", placeholder="you@example.com", key="login_email"
            )
            login_password = st.text_input(
                "Password", type="password", key="login_password"
            )
            login_submit = st.form_submit_button("Log in")
        if login_submit:
            if not login_email or not login_password:
                st.error("Please enter email and password.")
            else:
                try:
                    r = httpx.post(
                        LOGIN_URL,
                        json={"email": login_email.strip(), "password": login_password},
                        timeout=15,
                    )
                    data = (
                        r.json()
                        if r.headers.get("content-type", "").startswith(
                            "application/json"
                        )
                        else {}
                    )
                    status_msg = data.get("status", "")
                    if r.status_code == 200 and status_msg == "Login successful":
                        st.session_state.access_token = data.get("access_token")
                        st.session_state.user_email = data.get("email") or login_email
                        st.session_state.user_name = data.get("full_name") or ""
                        st.success("Login successful.")
                        st.rerun()
                    else:
                        detail = data.get("detail", "Invalid email or password.")
                        if r.status_code == 423:
                            sec = data.get("remaining_seconds", 0)
                            st.error(
                                f"Account locked. Try again in {sec // 60} minutes."
                            )
                        else:
                            st.error(f"Login unsuccessful: {detail}")
                except Exception as e:
                    st.error(f"Could not connect: {e}")
        st.caption("Don't have an account? Click **Sign up** above.")
    else:
        st.subheader("Sign up")
        with st.form("signup_form"):
            signup_name = st.text_input(
                "Full name", placeholder="Your name", key="signup_name"
            )
            signup_email = st.text_input(
                "Email (username)", placeholder="you@example.com", key="signup_email"
            )
            signup_password = st.text_input(
                "Password",
                type="password",
                key="signup_password",
                help="At least 8 characters",
            )
            signup_confirm = st.text_input(
                "Confirm password", type="password", key="signup_confirm"
            )
            signup_submit = st.form_submit_button("Sign up")
        if signup_submit:
            if (
                not signup_name
                or not signup_email
                or not signup_password
                or not signup_confirm
            ):
                st.error("Please fill all fields.")
            elif signup_password != signup_confirm:
                st.error("Passwords do not match.")
            elif len(signup_password) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                try:
                    r = httpx.post(
                        SIGNUP_URL,
                        json={
                            "full_name": signup_name.strip(),
                            "email": signup_email.strip(),
                            "password": signup_password,
                            "confirm_password": signup_confirm,
                        },
                        timeout=15,
                    )
                    data = (
                        r.json()
                        if r.headers.get("content-type", "").startswith(
                            "application/json"
                        )
                        else {}
                    )
                    status_msg = data.get("status", "")
                    if r.status_code == 201 and status_msg == "Signup successful":
                        st.success(
                            f"Signup successful for **{data.get('full_name', signup_name)}**. Please log in."
                        )
                    else:
                        st.error(
                            f"Signup unsuccessful: {data.get('detail', 'Please try again.')}"
                        )
                except Exception as e:
                    st.error(f"Could not connect: {e}")
        st.caption("Already have an account? Click **Log in** above.")

    st.stop()

# --- SIDEBAR: Telemetry & Config (only when logged in) ---
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/streamlit/normani-avatars/main/avatars/female/12.png",
        width=60,
    )
    st.title("System Nexus")
    st.write(f"Logged in as: **{st.session_state.user_email or 'User'}**")
    if st.button("Log out", type="secondary", use_container_width=True):
        st.session_state.access_token = None
        st.session_state.user_email = None
        st.session_state.user_name = None
        st.rerun()
    st.divider()
    st.subheader("Real-time Performance")
    latency_gauge = create_gauge(st.session_state.last_perf["latency"], "Latency", "s")
    st.plotly_chart(latency_gauge, use_container_width=True, key="latency_chart")
    st.divider()
    st.subheader("Session Intelligence")
    st.info(f"Connected to GovGig Backend v{settings.APP_VERSION}\nStatus: Online ✅")
    if st.button("Reset Session Memory", type="secondary", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.subheader("Chat History")
    if st.button(
        "New Chat", type="primary", use_container_width=True, key="btn_new_chat"
    ):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    if st.button(
        "Chat history",
        type="secondary",
        use_container_width=True,
        key="btn_refresh_threads",
    ):
        token = st.session_state.access_token
        if token:
            try:
                r = httpx.get(
                    THREADS_URL,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10,
                )
                if r.status_code == 200:
                    data = r.json()
                    st.session_state.thread_list = data.get("threads") or []
                    st.success(
                        f"Found {len(st.session_state.thread_list)} conversation(s)."
                    )
                else:
                    st.error("Could not load conversations.")
            except Exception as e:
                st.error(f"Failed to load: {e}")
        st.rerun()
    threads = st.session_state.thread_list
    if threads:

        def _thread_label(i):
            t = threads[i]
            date = (t.get("updated_at") or "")[:16].replace("T", " ")
            prev = (t.get("preview") or "No preview")[:40]
            return f"{date} — {prev}..."

        idx = st.selectbox(
            "Select conversation",
            range(len(threads)),
            format_func=_thread_label,
            key="thread_select",
        )
        if st.button(
            "Open this thread",
            type="secondary",
            use_container_width=True,
            key="btn_open_thread",
        ):
            thread_id = threads[idx]["thread_id"]
            token = st.session_state.access_token
            if token and thread_id:
                try:
                    r = httpx.get(
                        HISTORY_URL,
                        params={"thread_id": thread_id},
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=10,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        hist = data.get("history") or []
                        st.session_state.history = [
                            {"role": h["role"], "text": h.get("text", "")} for h in hist
                        ]
                        st.session_state.thread_id = thread_id
                        st.success(f"Loaded {len(hist)} message(s).")
                    else:
                        st.error("Could not load conversation.")
                except Exception as e:
                    st.error(f"Failed to load: {e}")
            st.rerun()
    else:
        st.caption("Click **Chat history** to see past conversations.")

# --- MAIN UI ---
col_main, col_evidence = st.columns([0.65, 0.35])

with col_main:
    st.markdown(
        f"## <span style='color:{PRIMARY_COLOR};'>Gov</span>Gig Insight AI",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Chat History Display
    chat_container = st.container(height=500, border=False)
    with chat_container:
        for i, chat in enumerate(st.session_state.history):
            text = chat.get("text") or chat.get("content") or ""
            if chat.get("role") == "user":
                st.markdown(
                    f'<div class="user-msg"><b>You:</b><br>{text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="ai-msg"><b>AI Assistant:</b><br>{text}</div>',
                    unsafe_allow_html=True,
                )
                query_id = chat.get("query_id")
                if query_id and query_id not in st.session_state.feedback_sent:
                    col_up, col_down, _ = st.columns([1, 1, 4])
                    with col_up:
                        if st.button("👍 Good", key=f"fb_up_{i}_{query_id[:8]}"):
                            token = st.session_state.access_token
                            if token:
                                try:
                                    r = httpx.post(
                                        FEEDBACK_URL,
                                        json={"query_id": query_id, "response": "good"},
                                        headers={"Authorization": f"Bearer {token}"},
                                        timeout=10,
                                    )
                                    if r.status_code == 201:
                                        st.session_state.feedback_sent.add(query_id)
                                        st.success("Thanks for your feedback!")
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Could not send feedback: {e}")
                    with col_down:
                        if st.button("👎 Bad", key=f"fb_down_{i}_{query_id[:8]}"):
                            token = st.session_state.access_token
                            if token:
                                try:
                                    r = httpx.post(
                                        FEEDBACK_URL,
                                        json={"query_id": query_id, "response": "bad"},
                                        headers={"Authorization": f"Bearer {token}"},
                                        timeout=10,
                                    )
                                    if r.status_code == 201:
                                        st.session_state.feedback_sent.add(query_id)
                                        st.success("Thanks for your feedback!")
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Could not send feedback: {e}")
                elif query_id and query_id in st.session_state.feedback_sent:
                    st.caption("✓ Feedback recorded")
                if "documents" in chat and chat["documents"]:
                    with st.expander("Show Linked Regulations & Evidence"):
                        for doc in chat["documents"]:
                            st.markdown(
                                f"""
                            <div class="doc-card">
                                <small style='color:{PRIMARY_COLOR}'>{doc.get('regulation_type', 'UNK')} Section {doc.get('section', 'N/A')}</small><br>
                                <b>Source:</b> {doc.get('source', 'Unknown File')}<br>
                                <p style='font-size: 0.85rem;'>{doc.get('content', '')[:200]}...</p>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

    # Input Area
    prompt = st.chat_input("Ask a regulatory question (e.g., 'What is FAR 52.219-8?')")

    if prompt:
        if not st.session_state.access_token:
            st.error("Session expired. Please log in again.")
            st.stop()
        # Add user msg to state
        st.session_state.history.append({"role": "user", "text": prompt})

        async def stream_query():
            t0 = time.perf_counter()
            token = st.session_state.access_token

            clean_history = [
                {"role": h["role"], "text": h.get("text", "")}
                for h in st.session_state.history[:-1]
            ]

            try:
                async with websockets.connect(
                    WS_URL, ping_timeout=REQUEST_TIMEOUT_SECONDS
                ) as ws:
                    # 1. Authenticate & Query
                    payload = {
                        "token": token,
                        "query": prompt,
                        "history": clean_history,
                        "thread_id": st.session_state.thread_id,
                        "cot": False,
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
                                "confidence": res_data.get("confidence", 0.85),
                            }

                            st.session_state.history.append(
                                {
                                    "role": "ai",
                                    "text": full_response,
                                    "documents": res_data.get("documents", []),
                                    "agent_path": res_data.get("agent_path", []),
                                    "query_id": res_data.get("query_id"),
                                }
                            )

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
        with st.status(
            "🔍 Analyzing documents (Streaming)...", expanded=True
        ) as status:
            success = asyncio.run(stream_query())
            if success:
                status.update(
                    label="✅ Analysis Completed", state="complete", expanded=False
                )
                st.rerun()
            else:
                status.update(label="❌ Connection Failed", state="error")

with col_evidence:
    st.markdown("### 🗃️ Evidence Viewer")
    st.write("Click on a citation in the chat to view the original text snippet here.")

    # Display top documents from last AI response
    if st.session_state.history and st.session_state.history[-1]["role"] == "ai":
        last_ai = st.session_state.history[-1]
        docs = last_ai.get("documents") or []
        if docs:
            for i, doc in enumerate(docs[:3]):
                with st.container(border=True):
                    st.markdown(
                        f"**Rank {doc.get('rank', i+1)}: {doc.get('regulation_type', 'UNK')} {doc.get('section', '')}**"
                    )
                    st.caption(f"File: {doc.get('source', 'N/A')}")
                    st.info(doc.get("content", ""))
                    raw_score = (
                        float(doc.get("score", 0))
                        if doc.get("score") is not None
                        else 0.0
                    )
                    progress_val = min(
                        1.0, max(0.0, raw_score)
                    )  # st.progress() requires [0, 1]
                    pct = min(100.0, max(0.0, raw_score * 100))
                    st.progress(progress_val, text=f"Relevance: {pct:.1f}%")
        else:
            st.info("No evidence retrieved for this query.")
    else:
        st.info("Retrieve information to see evidence breakdown here.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: grey;'>GovGig AI Command Center © {datetime.now().year} | v{settings.APP_VERSION}</p>",
    unsafe_allow_html=True,
)

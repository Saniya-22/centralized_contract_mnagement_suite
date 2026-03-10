"""
Admin Feedback Dashboard — TOTAL / POSITIVE / NEGATIVE feedback counts from analytics summary.
Run: streamlit run dashboard/feedback_admin.py
"""
import streamlit as st
import httpx
import os
import sys

# Add project root for src.config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set API_BASE_URL to match your backend (e.g. http://localhost:8001 if backend runs on 8001)
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
SUMMARY_URL = f"{API_BASE}/api/v1/analytics/summary"
PRIMARY_COLOR = "#00D1FF"
GREEN = "#22c55e"
NEGATIVE_COLOR = "#f97316"  # reddish-orange

st.set_page_config(
    page_title="Feedback Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1A1C2C 0%, #2D1B3D 50%, #1A1C2C 100%);
        background-attachment: fixed;
    }
    .feedback-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    }
    .feedback-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }
    .feedback-value {
        font-size: 2.8rem;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)


def get_token():
    """JWT for dashboard (same as chat dashboard)."""
    try:
        from src.config import settings
        from datetime import datetime, timedelta
        from jose import jwt
        expire = datetime.utcnow() + timedelta(minutes=getattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES", 30))
        to_encode = {
            "sub": "internal-dashboard-admin",
            "exp": expire,
            "name": "Dashboard Admin",
            "role": "admin",
        }
        return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    except Exception:
        return os.getenv("DASHBOARD_JWT") or None


def fetch_summary():
    """GET /api/v1/analytics/summary and return feedback counts."""
    token = get_token()
    if not token:
        return None, "No JWT available. Set DASHBOARD_JWT or ensure src.config is loadable."
    try:
        r = httpx.get(
            SUMMARY_URL,
            params={"hours": 24},
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        if r.status_code != 200:
            return None, f"API returned {r.status_code}"
        data = r.json()
        return data, None
    except Exception as e:
        return None, str(e)


st.title("Feedback Dashboard")
st.markdown("---")

data, err = fetch_summary()
if err:
    st.error(f"Cannot load feedback summary: {err}")
    st.info("Ensure the backend is running and JWT is configured (e.g. .env with JWT_SECRET_KEY).")
    st.stop()

total = data.get("feedback_total", 0)
positive = data.get("feedback_positive", 0)
negative = data.get("feedback_negative", 0)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="feedback-card">
            <div class="feedback-label">Total</div>
            <div class="feedback-value" style="color: white;">{total}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="feedback-card">
            <div class="feedback-label">Positive</div>
            <div class="feedback-value" style="color: {GREEN};">{positive}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="feedback-card">
            <div class="feedback-label">Negative</div>
            <div class="feedback-value" style="color: {NEGATIVE_COLOR};">{negative}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
if st.button("Refresh"):
    st.rerun()

# core/ui.py
import streamlit as st

def apply_theme():
    st.set_page_config(page_title="AI Stock Sniper (Auto)", page_icon="🦄", layout="wide")
    st.markdown("""
    <style>
      .stApp { background: #0f1116; color: #e7eaf0; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1200px; }

      /* inputs */
      div[data-testid="stTextInput"] input {
        background: #171a22 !important;
        color: #e7eaf0 !important;
        border: 1px solid #2a2f3b !important;
        border-radius: 10px !important;
        padding: 10px 12px !important;
      }

      /* tabs */
      button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        padding: 10px 14px !important;
      }

      /* tables */
      div[data-testid="stDataFrame"] {
        border: 1px solid #2a2f3b;
        border-radius: 12px;
        overflow: hidden;
      }

      /* cards */
      .card {
        background: #141824;
        border: 1px solid #2a2f3b;
        border-radius: 14px;
        padding: 14px 14px;
      }
      .muted { color: #98a2b3; }
      .big { font-size: 1.6rem; font-weight: 800; }
      .pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        border: 1px solid #2a2f3b;
        background: #101321;
        font-size: 0.85rem;
        color: #cbd5e1;
      }
      a { color: #60a5fa !important; text-decoration: none; }
      a:hover { text-decoration: underline; }
    </style>
    """, unsafe_allow_html=True)

def card(title: str, value: str, subtitle: str = ""):
    st.markdown(f"""
      <div class="card">
        <div class="muted" style="font-weight:700;">{title}</div>
        <div class="big">{value}</div>
        <div class="muted" style="margin-top:6px;">{subtitle}</div>
      </div>
    """, unsafe_allow_html=True)

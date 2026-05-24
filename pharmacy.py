"""
PharmaDash — Pharmacy Inventory & Sales Analytics
Optimized version: faster reruns, cached aggregations, vectorised ops,
lightweight charts, category dtypes, minimal CSS, lazy data loading.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import io

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── AI / Groq ─────────────────────────────────────────────────────────────────
try:
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# Initialise Groq client once (None if key/package missing)
_groq_client = None
if _GROQ_AVAILABLE:
    _api_key = os.getenv("GROQ_API_KEY", "")
    if _api_key:
        try:
            _groq_client = Groq(api_key=_api_key)
        except Exception:
            _groq_client = None

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PharmaDash | Inventory & Sales",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS — all styles inline (no external file needed)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800;900&family=Rajdhani:wght@500;600;700&display=swap');

:root{--brand:#1565C0;--brand-lt:#42A5F5;--accent:#00BCD4;
      --danger:#E53935;--warn:#FB8C00;--ok:#43A047;--txt:#0d2f52}

html,body,[class*="css"]{font-family:'Nunito',sans-serif;color:var(--txt)}

/* ══ ANIMATED MEDICAL BACKGROUND ══ */
.stApp {
  min-height: 100vh;
  background:
    radial-gradient(ellipse 85% 65% at 12% 8%,  rgba(100,200,255,0.40) 0%, transparent 65%),
    radial-gradient(ellipse 65% 85% at 88% 6%,  rgba(173,216,255,0.34) 0%, transparent 60%),
    radial-gradient(ellipse 75% 55% at 50% 92%, rgba(186,235,255,0.32) 0%, transparent 65%),
    radial-gradient(ellipse 55% 45% at 92% 58%, rgba(224,247,255,0.26) 0%, transparent 55%),
    radial-gradient(ellipse 45% 60% at 5%  75%, rgba(150,210,255,0.20) 0%, transparent 55%),
    linear-gradient(148deg, #cce8f7 0%, #b6d9f0 18%, #c8e5f8 42%, #d8eefb 68%, #e8f5fd 100%);
  background-attachment: fixed;
}

/* Hex + cross tiled overlay */
.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background-image:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='130' height='150'%3E%3Cpolygon points='65,7 118,36 118,94 65,123 12,94 12,36' fill='none' stroke='rgba(255,255,255,0.38)' stroke-width='1.1'/%3E%3Crect x='57' y='53' width='16' height='5' rx='2.5' fill='rgba(255,255,255,0.32)'/%3E%3Crect x='62' y='48' width='5' height='16' rx='2.5' fill='rgba(255,255,255,0.32)'/%3E%3Ccircle cx='65' cy='150' r='2' fill='rgba(255,255,255,0.15)'/%3E%3C%2Fsvg%3E"),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='50' height='50'%3E%3Ccircle cx='25' cy='25' r='1.8' fill='rgba(255,255,255,0.18)'/%3E%3C%2Fsvg%3E");
  background-size: 130px 150px, 50px 50px;
  background-position: 0 0, 25px 25px;
  animation: hexdrift 35s ease-in-out infinite alternate;
}

@keyframes hexdrift {
  0%   { background-position: 0px 0px,   25px 25px; }
  100% { background-position: 26px 20px, 51px 45px; }
}

/* Floating ambient orb top-right */
.stApp::after {
  content: "";
  position: fixed;
  width: 600px; height: 600px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(66,165,245,0.14) 0%, transparent 70%);
  top: -140px; right: -140px;
  pointer-events: none;
  z-index: 0;
  animation: orb 22s ease-in-out infinite alternate;
}
@keyframes orb {
  0%   { transform: translate(0,0)   scale(1.0); opacity:1; }
  100% { transform: translate(-90px,130px) scale(1.35); opacity:0.7; }
}

.main .block-container{position:relative;z-index:1;padding-top:1.4rem;padding-bottom:2rem;padding-left:2rem;padding-right:2rem}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{
  background:linear-gradient(170deg,#082d54 0%,#0f4a8a 30%,#1565C0 65%,#0b3d6e 100%) !important;
  border-right:1px solid rgba(255,255,255,0.14);
  box-shadow:4px 0 32px rgba(13,45,84,0.45);
  position:relative;overflow:hidden}
[data-testid="stSidebar"]::before{
  content:"";position:absolute;inset:0;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='92'%3E%3Cpolygon points='40,5 74,24 74,62 40,81 6,62 6,24' fill='none' stroke='rgba(255,255,255,0.07)' stroke-width='1'/%3E%3C/svg%3E");
  background-size:80px 92px;pointer-events:none}
[data-testid="stSidebar"] *{color:#fff !important}
[data-testid="stSidebar"] h2{color:#fff !important;text-shadow:0 2px 8px rgba(0,0,0,0.35)}

[data-testid="stSidebar"] [data-baseweb="radio"]{
  background:rgba(255,255,255,0.10) !important;
  border:1px solid rgba(255,255,255,0.20) !important;
  border-radius:10px !important;padding:9px 13px !important;margin-bottom:4px}
[data-testid="stSidebar"] [data-baseweb="radio"]:hover{
  background:rgba(255,255,255,0.20) !important;border-color:rgba(255,255,255,0.40) !important}
[data-testid="stSidebar"] [data-baseweb="radio"]>div+div{
  font-weight:700 !important;font-size:0.88rem !important;
  color:#e0f2ff !important;letter-spacing:0.2px}
[data-testid="stSidebar"] .stSelectbox>div>div{
  background:rgba(255,255,255,0.13) !important;
  border:1px solid rgba(255,255,255,0.28) !important;border-radius:9px !important}
[data-testid="stSidebar"] label{
  color:#a8d4f0 !important;font-size:0.75rem !important;
  font-weight:800 !important;letter-spacing:1px}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,0.18) !important}

/* Date-picker dark style */
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="base-input"] input{
  background:#0a1929 !important;color:#e0f2ff !important;
  border:1px solid rgba(255,255,255,0.28) !important;border-radius:8px !important}
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"]{
  background:#0a1929 !important;
  border:1px solid rgba(255,255,255,0.28) !important;border-radius:8px !important}

/* Calendar popup */
div[data-baseweb="popover"]:has(div[data-baseweb="calendar"]),
div[data-baseweb="popover"]:has(div[data-baseweb="calendar"])>div,
div[data-baseweb="popover"]:has(div[data-baseweb="calendar"])>div>div{
  background:#071c2e !important;border:1px solid rgba(255,255,255,0.18) !important;
  border-radius:14px !important;box-shadow:0 8px 32px rgba(0,0,0,0.45) !important}
div[data-baseweb="calendar"],div[data-baseweb="calendar"]>div,
div[data-baseweb="datepicker"]{background:#071c2e !important}
div[data-baseweb="calendar"] *:not(button[aria-selected="true"] *){
  background:transparent !important;color:#c8e6ff !important}
div[data-baseweb="calendar"] button{color:#e0f2ff !important;background:transparent !important}
div[data-baseweb="calendar"] [role="columnheader"] div{color:#64B5F6 !important;font-weight:800 !important}
div[data-baseweb="calendar"] [role="gridcell"] button div,
div[data-baseweb="calendar"] [role="option"] button div{color:#e0f2ff !important}
div[data-baseweb="calendar"] [role="gridcell"] button:hover div,
div[data-baseweb="calendar"] [role="option"] button:hover div{
  background:rgba(25,118,210,0.40) !important;border-radius:50% !important}
div[data-baseweb="calendar"] [aria-selected="true"] div,
div[data-baseweb="calendar"] [data-selected="true"] div{
  background:#1976D2 !important;border-radius:50% !important;color:#fff !important}
div[data-baseweb="calendar"] [data-range-highlight="true"] div{background:rgba(25,118,210,0.20) !important}
div[data-baseweb="calendar"] [data-baseweb="select"]>div{
  background:#0d2540 !important;border-color:rgba(255,255,255,0.25) !important;
  border-radius:8px !important;color:#c8e6ff !important}
div[data-baseweb="calendar"] ul[role="listbox"]{background:#071c2e !important}
div[data-baseweb="calendar"] ul[role="listbox"] li{color:#c8e6ff !important;background:transparent !important}
div[data-baseweb="calendar"] ul[role="listbox"] li:hover{background:rgba(25,118,210,0.35) !important}

/* Sidebar nav buttons — horizontal-tab style */
[data-testid="stSidebar"] [data-testid="stButton"]>button{
  background:rgba(255,255,255,0.10) !important;
  border:1px solid rgba(255,255,255,0.22) !important;
  border-radius:10px !important;
  color:#e0f2ff !important;
  font-weight:700 !important;
  font-size:0.86rem !important;
  text-align:left !important;
  padding:9px 14px !important;
  margin-bottom:3px !important;
  letter-spacing:0.2px;
  transition:background 0.15s,border-color 0.15s}
[data-testid="stSidebar"] [data-testid="stButton"]>button:hover{
  background:rgba(255,255,255,0.22) !important;
  border-color:rgba(255,255,255,0.50) !important}
[data-testid="stSidebar"] [data-testid="stButton"]>button:focus,
[data-testid="stSidebar"] [data-testid="stButton"]>button:active{
  background:rgba(255,255,255,0.28) !important;
  border:2px solid rgba(255,255,255,0.72) !important;
  color:#ffffff !important}

/* Apply Filters + Download buttons */
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"]>button,
[data-testid="stSidebar"] [data-testid="stDownloadButton"]>button{
  background:linear-gradient(135deg,rgba(0,96,100,0.90),rgba(0,188,212,0.88)) !important;
  border:1px solid rgba(0,230,250,0.55) !important;color:#ffffff !important;
  border-radius:10px !important;font-weight:700 !important;font-size:0.84rem !important;
  box-shadow:0 3px 10px rgba(0,188,212,0.20) !important;width:100%}
[data-testid="stSidebar"] [data-testid="stDownloadButton"]>button:hover{
  background:linear-gradient(135deg,rgba(0,121,107,0.95),rgba(0,229,255,0.90)) !important}

/* Page title */
.page-title{
  font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;
  letter-spacing:1.5px;color:#0d3d6e;text-align:center;margin-bottom:18px;
  text-shadow:0 2px 14px rgba(255,255,255,0.95),0 1px 3px rgba(21,101,192,0.18)}

/* KPI cards — glassmorphism with hover */
.kpi-row{display:flex;gap:14px;margin-bottom:22px;flex-wrap:wrap}
.kpi-card{
  flex:1;min-width:130px;
  background:linear-gradient(135deg,rgba(21,101,192,0.88) 0%,rgba(30,136,229,0.86) 100%);
  backdrop-filter:blur(14px);
  border:1px solid rgba(255,255,255,0.35);border-radius:16px;
  padding:16px 18px;text-align:center;
  box-shadow:0 6px 26px rgba(21,101,192,0.26),inset 0 1px 0 rgba(255,255,255,0.28);
  position:relative;overflow:hidden;
  transition:transform 0.2s,box-shadow 0.2s}
.kpi-card:hover{transform:translateY(-3px);box-shadow:0 10px 32px rgba(21,101,192,0.35)}
.kpi-card::after{content:"";position:absolute;top:-28px;right:-28px;
  width:80px;height:80px;border-radius:50%;background:rgba(255,255,255,0.10)}
.kpi-card .val{font-family:'Rajdhani',sans-serif;font-size:1.95rem;font-weight:700;color:#fff;line-height:1.1}
.kpi-card .lbl{font-size:0.67rem;font-weight:800;color:rgba(255,255,255,0.87);
  text-transform:uppercase;letter-spacing:0.9px;margin-top:5px}
.kpi-card.warn{background:linear-gradient(135deg,rgba(183,28,28,0.88),rgba(239,108,0,0.86))}
.kpi-card.ok  {background:linear-gradient(135deg,rgba(27,94,32,0.88),rgba(67,160,71,0.86))}
.kpi-card.teal{background:linear-gradient(135deg,rgba(0,96,100,0.88),rgba(0,188,212,0.86))}

/* Section title */
.sec-title{
  font-family:'Rajdhani',sans-serif;font-size:1.02rem;font-weight:700;
  color:#0b3d6e;letter-spacing:0.6px;border-left:4px solid #42A5F5;
  padding:4px 10px;background:rgba(255,255,255,0.52);
  border-radius:0 8px 8px 0;margin-bottom:10px;margin-top:18px}

/* Fill column gaps — Streamlit horizontal blocks */
[data-testid="stHorizontalBlock"]{gap:1.2rem !important;margin-bottom:0.5rem}
[data-testid="stColumn"]{gap:0}

/* Chart panels — glassmorphism */
[data-testid="stPlotlyChart"]>div{
  background:rgba(255,255,255,0.65) !important;border-radius:14px !important;
  border:1px solid rgba(255,255,255,0.52) !important;
  box-shadow:0 4px 20px rgba(21,101,192,0.09) !important;padding:4px !important;
  margin-bottom:6px !important}

/* Insight box — glassmorphism */
.insight-box{
  background:rgba(255,255,255,0.72);
  border:1px solid rgba(255,255,255,0.55);border-radius:14px;
  padding:18px 22px;margin-bottom:14px;
  box-shadow:0 3px 16px rgba(21,101,192,0.08)}
.insight-box h4{font-family:'Rajdhani',sans-serif;color:#0b3d6e;margin:0 0 9px;font-size:1rem}
.insight-box ul{margin:0;padding-left:17px}
.insight-box li{font-size:0.85rem;color:#1a3a55;margin-bottom:5px;line-height:1.45}

/* Dataframe */
[data-testid="stDataFrame"]{
  background:rgba(255,255,255,0.80) !important;border-radius:12px !important;
  border:1px solid rgba(255,255,255,0.52) !important;
  margin-bottom:10px !important}

/* Dropdown / popover menus */
div[data-baseweb="popover"]>div,div[data-baseweb="popover"]>div>div,
div[data-baseweb="tooltip"],div[data-baseweb="menu"]{
  background:rgba(235,245,255,0.97) !important;
  border:1px solid rgba(21,101,192,0.25) !important;
  border-radius:12px !important;box-shadow:0 6px 24px rgba(21,101,192,0.14) !important}
div[data-baseweb="menu"] [role="option"],div[data-baseweb="menu"] li,
div[data-baseweb="popover"] [role="option"],div[data-baseweb="popover"] li{
  color:#0d3d6e !important;background:transparent !important;
  font-weight:600 !important;font-size:0.84rem !important}
div[data-baseweb="menu"] [role="option"]:hover,div[data-baseweb="menu"] li:hover{
  background:rgba(21,101,192,0.12) !important;color:#1565C0 !important;border-radius:7px !important}
div[data-baseweb="menu"] [role="option"] svg,div[data-baseweb="menu"] li svg{
  fill:#1565C0 !important;color:#1565C0 !important}

/* Sidebar selectbox list */
[data-baseweb="popover"] ul[role="listbox"],ul[role="listbox"]{
  background:rgba(235,245,255,0.98) !important;
  border:1px solid rgba(21,101,192,0.25) !important;
  border-radius:10px !important;box-shadow:0 4px 18px rgba(21,101,192,0.12) !important}
ul[role="listbox"] li,ul[role="listbox"] [role="option"]{
  color:#0d3d6e !important;background:transparent !important;
  font-weight:600 !important;font-size:0.85rem !important}
ul[role="listbox"] li:hover,ul[role="listbox"] [role="option"]:hover{
  background:rgba(21,101,192,0.12) !important;color:#1565C0 !important}

::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-thumb{background:rgba(21,101,192,0.35);border-radius:3px}

/* ── FILL STREAMLIT DEFAULT VERTICAL GAPS ── */
.stMarkdown{margin-bottom:0 !important}
[data-testid="stVerticalBlock"]>div{gap:0.55rem}
[data-testid="element-container"]{margin-bottom:0 !important}
.block-container .stAlert{margin-bottom:10px}
.block-container [data-testid="stMetric"]{
  background:rgba(255,255,255,0.65);border-radius:12px;
  border:1px solid rgba(255,255,255,0.50);padding:12px 16px;
  box-shadow:0 3px 12px rgba(21,101,192,0.07)}

/* ── AI ASSISTANT STYLES ── */
.ai-section-title{
  font-family:'Rajdhani',sans-serif;font-size:1.3rem;font-weight:700;
  color:#0b3d6e;letter-spacing:1px;margin:28px 0 14px;
  display:flex;align-items:center;gap:8px}
.ai-section-title::after{
  content:"";flex:1;height:2px;
  background:linear-gradient(90deg,rgba(21,101,192,0.35),transparent)}

.ai-response-card{
  background:linear-gradient(135deg,rgba(227,242,253,0.95),rgba(245,251,255,0.92));
  border:1px solid rgba(21,101,192,0.22);
  border-left:5px solid #1565C0;
  border-radius:16px;padding:18px 24px;margin-top:8px;margin-bottom:12px;
  box-shadow:0 4px 24px rgba(21,101,192,0.12);
  font-size:0.88rem;color:#0d2f52;line-height:1.7}
.ai-response-card b,.ai-response-card strong{color:#0b3d6e}

.ai-chat-card{
  background:linear-gradient(135deg,rgba(232,245,255,0.95),rgba(248,253,255,0.92));
  border:1px solid rgba(21,101,192,0.20);
  border-left:5px solid #00BCD4;
  border-radius:16px;padding:18px 24px;margin-top:8px;margin-bottom:12px;
  box-shadow:0 4px 24px rgba(0,188,212,0.10);
  font-size:0.88rem;color:#0d2f52;line-height:1.7}

.ai-status-pill{
  display:inline-block;background:rgba(21,101,192,0.12);
  border:1px solid rgba(21,101,192,0.25);border-radius:20px;
  padding:3px 12px;font-size:0.72rem;font-weight:700;
  color:#1565C0;letter-spacing:0.5px;margin-bottom:10px}

/* ── Q&A INSIGHT CARDS ── */
.qa-grid{display:flex;flex-direction:column;gap:10px;margin-top:12px;margin-bottom:18px}
.qa-card{
  display:grid;grid-template-columns:38% 62%;
  background:linear-gradient(135deg,rgba(227,242,253,0.95),rgba(245,251,255,0.92));
  border:1px solid rgba(21,101,192,0.18);
  border-radius:14px;overflow:hidden;
  box-shadow:0 3px 16px rgba(21,101,192,0.09)}
.qa-question{
  background:linear-gradient(135deg,rgba(21,101,192,0.88),rgba(30,136,229,0.84));
  padding:16px 18px;display:flex;align-items:center;
  font-family:'Rajdhani',sans-serif;font-size:0.92rem;font-weight:700;
  color:#fff;line-height:1.45;letter-spacing:0.2px}
.qa-answer{
  padding:12px 16px;font-size:0.84rem;color:#0d2f52;line-height:1.6;
  border-left:3px solid rgba(21,101,192,0.20);display:flex;flex-direction:column;gap:8px}
.qa-answer ul{margin:0;padding-left:16px}
.qa-answer li{margin-bottom:4px}

/* Problem block */
.qa-problem{
  background:linear-gradient(135deg,rgba(229,57,53,0.10),rgba(183,28,28,0.06));
  border:1px solid rgba(229,57,53,0.32);border-left:4px solid #E53935;
  border-radius:9px;padding:10px 13px}
.qa-problem-label{
  display:block;font-size:0.68rem;font-weight:900;letter-spacing:1.4px;
  color:#C62828;text-transform:uppercase;margin-bottom:5px}
.qa-problem-text{margin:0;font-size:0.84rem;color:#5a1010;line-height:1.6;font-weight:600}

/* Solution block */
.qa-solution{
  background:linear-gradient(135deg,rgba(67,160,71,0.10),rgba(27,94,32,0.06));
  border:1px solid rgba(67,160,71,0.32);border-left:4px solid #43A047;
  border-radius:9px;padding:10px 13px}
.qa-solution-label{
  display:block;font-size:0.68rem;font-weight:900;letter-spacing:1.4px;
  color:#2E7D32;text-transform:uppercase;margin-bottom:5px}
.qa-solution-list{margin:0;padding-left:15px;color:#1a3d1a;font-size:0.84rem;font-weight:600}
.qa-solution-list li{margin-bottom:5px;line-height:1.55}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (defined once at module level — never recomputed on rerun)
# ═══════════════════════════════════════════════════════════════════════════════

CAT_PALETTE: dict[str, str] = {
    "Analgesics": "#1976D2", "Antibiotics": "#7B1FA2", "Diabetes": "#F57C00",
    "Vitamins": "#388E3C",   "Cardiac": "#D32F2F",     "Antiseptics": "#0097A7",
}

BLUES: list[str] = [
    "#0D47A1","#1565C0","#1976D2","#1E88E5","#2196F3",
    "#42A5F5","#64B5F6","#0288D1","#01579B","#003c8f",
]

# Stock-status coordinate lookup — used in Business Insights map
STORE_COORDS: dict[str, tuple[float, float]] = {
    "MedPlus Ameerpet":       (17.4344, 78.4487),
    "Apollo Kukatpally":      (17.4924, 78.3918),
    "WellCare Banjara Hills": (17.4138, 78.4349),
    "HealthFirst Madhapur":   (17.4474, 78.3910),
    "CityMed Dilsukhnagar":   (17.3688, 78.5268),
}

# Stock-status conditions used in batch-expiry table (vectorised)
_STS_CONDS    = None   # lazily populated after load_data()
_STS_CHOICES  = ["Out of Stock", "Low Stock"]
_STS_DEFAULT  = "In Stock"

# Pre-built filter option lists (populated directly from load_data below)
CATEGORIES: list[str] = []
STORES:     list[str] = []
SUPPLIERS:  list[str] = []
DATE_MIN    = None
DATE_MAX    = None
# NOTE: these are assigned immediately after load_data() below — no empty-list risk.


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  — cached for the lifetime of the server process
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading pharmacy data…")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all CSVs once, apply dtype optimisations, and return frozen DataFrames.
    All downstream operations filter/aggregate these — never mutate them.
    """
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        _required = {
            "Fact_Sales_20k.csv": "Sales transactions",
            "Fact_Stock_20k.csv": "Stock records",
            "Dim_Products_20k.csv": "Product master",
            "Dim_Store.csv": "Store data",
            "Dim_Suppliers_20k.csv": "Supplier data",
        }
        for fname, label in _required.items():
            fpath = os.path.join(base, fname)
            if not os.path.exists(fpath):
                st.error(
                    f"❌ **Missing dataset file:** `{fname}` ({label})\n\n"
                    f"Expected at: `{fpath}`\n\n"
                    "Please ensure the `dataset/` folder is present next to `pharmacy_dash.py`."
                )
                st.stop()
        sales = pd.read_csv(os.path.join(base, "Fact_Sales_20k.csv"))
        stock = pd.read_csv(os.path.join(base, "Fact_Stock_20k.csv"))
        prod  = pd.read_csv(os.path.join(base, "Dim_Products_20k.csv"))
        store = pd.read_csv(os.path.join(base, "Dim_Store.csv"))
        sup   = pd.read_csv(os.path.join(base, "Dim_Suppliers_20k.csv"))
    except st.runtime.scriptrunner.StopException:
        raise  # let st.stop() propagate
    except FileNotFoundError as exc:
        st.error(f"❌ **Dataset file not found:** {exc}\n\nCheck the `dataset/` folder.")
        st.stop()
    except Exception as exc:
        st.error(f"❌ **Failed to load pharmacy data:** {exc}")
        st.stop()

    # ── Dates ──────────────────────────────────────────────────────────────────
    sales["Date"]      = pd.to_datetime(sales["Date"], dayfirst=False, errors="coerce")
    sales["Month"]     = sales["Date"].dt.strftime("%B")
    sales["Month_Num"] = sales["Date"].dt.month

    # ── Joins ──────────────────────────────────────────────────────────────────
    sales = sales.merge(store[["Store_ID", "Store_Name"]], on="Store_ID", how="left")
    sales = sales.merge(
        prod[["ProductID", "UnitCost"]],
        on="ProductID", how="left")
    stock = stock.merge(store[["Store_ID", "Store_Name"]], on="Store_ID", how="left")
    stock = stock.merge(
        prod[["ProductID", "ProductName", "Category", "UnitCost", "RetailPrice", "ReorderPoint", "SafetyStock"]],
        on="ProductID", how="left")
    stock = stock.merge(
        sup[["SupplierID", "Supplier_Name", "QualityRating", "LeadTimeDays"]],
        on="SupplierID", how="left")

    # ── Downcast numeric columns → lower memory, faster ops ───────────────────
    # Note: pd.to_numeric(downcast=...) is deprecated in pandas 2.x — use astype() directly.
    for col in ["QuantitySold", "TotalAmount", "Month_Num", "UnitCost"]:
        if col in sales.columns:
            sales[col] = pd.to_numeric(sales[col], errors="coerce").astype("float32")

    for col in ["QuantityOnHand", "UnitCost", "RetailPrice", "ReorderPoint", "SafetyStock",
                "QualityRating", "LeadTimeDays", "DaysToExpiry"]:
        if col in stock.columns:
            stock[col] = pd.to_numeric(stock[col], errors="coerce").astype("float32")

    # ── Convert high-cardinality object columns → category dtype ──────────────
    # Dramatically reduces memory and speeds up groupby/filter
    for col in ["Category", "Store_Name", "CustomerType", "Medicine_Name"]:
        if col in sales.columns:
            sales[col] = sales[col].astype("category")

    for col in ["Category", "Store_Name", "Supplier_Name",
                "ProductName", "ExpiryStatus", "BatchNumber"]:
        if col in stock.columns:
            stock[col] = stock[col].astype("category")

    return sales, stock, prod, store, sup


# ── Load once and populate module-level constants ────────────────────────────
sales_df, stock_df, prod_df, store_df, sup_df = load_data()

# Populate filter-option lists (sorted, immutable after first run)
CATEGORIES = sorted(sales_df["Category"].dropna().unique().tolist())
STORES     = sorted(sales_df["Store_Name"].dropna().unique().tolist())
SUPPLIERS  = sorted(stock_df["Supplier_Name"].dropna().unique().tolist())
DATE_MIN   = sales_df["Date"].min().date()
DATE_MAX   = sales_df["Date"].max().date()

# Pre-build option lists with "All" prefix — avoids repeated list concat in sidebar
_STORE_OPTS    = ["All"] + STORES
_CAT_OPTS      = ["All"] + CATEGORIES
_SUPPLIER_OPTS = ["All"] + SUPPLIERS
_STATUS_OPTS   = ["All", "Low Stock", "Out of Stock", "Available"]


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED FILTER + AGGREGATION FUNCTIONS
# Keyed by (scalar) filter params so results survive reruns when params unchanged.
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_filtered_sales(
    dr_min, dr_max, sel_store: str, sel_cat: str
) -> pd.DataFrame:
    """Return a filtered view of sales_df — no copy unless a filter applies."""
    d = sales_df
    d = d[(d["Date"].dt.date >= dr_min) & (d["Date"].dt.date <= dr_max)]
    if sel_store != "All": d = d[d["Store_Name"] == sel_store]
    if sel_cat   != "All": d = d[d["Category"]   == sel_cat]
    return d


@st.cache_data(show_spinner=False)
def get_filtered_stock(
    sel_store: str, sel_cat: str, sel_supplier: str, sel_status: str
) -> pd.DataFrame:
    """Return a filtered view of stock_df."""
    d = stock_df
    if sel_store    != "All": d = d[d["Store_Name"]    == sel_store]
    if sel_cat      != "All": d = d[d["Category"]      == sel_cat]
    if sel_supplier != "All": d = d[d["Supplier_Name"] == sel_supplier]
    if sel_status == "Low Stock":
        d = d[(d["QuantityOnHand"] > 0) & (d["QuantityOnHand"] < d["ReorderPoint"])]
    elif sel_status == "Out of Stock":
        d = d[d["QuantityOnHand"] == 0]
    elif sel_status == "Available":
        d = d[d["QuantityOnHand"] >= d["ReorderPoint"]]
    return d


@st.cache_data(show_spinner=False)
def get_overview_kpis(dr_min, dr_max, sel_store, sel_cat, sel_supplier, sel_status) -> dict:
    fs = get_filtered_sales(dr_min, dr_max, sel_store, sel_cat)
    fk = get_filtered_stock(sel_store, sel_cat, sel_supplier, sel_status)
    qty   = fk["QuantityOnHand"].to_numpy()
    cost  = fk["UnitCost"].to_numpy()
    rp    = fk["ReorderPoint"].to_numpy()
    expst = fk["ExpiryStatus"].to_numpy(dtype=str)       # str comparison is fast on numpy
    return {
        "total_sv":    float(np.dot(qty, cost)),
        "total_rev":   float(fs["TotalAmount"].sum()),
        "total_units": float(fs["QuantitySold"].sum()),
        "expired_cnt": int((expst == "Expired").sum()),
        "low_stock":   int((qty < rp).sum()),
        "safe_pct":    round(float((expst == "Valid").sum()) / max(len(fk), 1) * 100, 1),
    }


@st.cache_data(show_spinner=False)
def get_monthly_revenue(dr_min, dr_max, sel_store, sel_cat) -> pd.DataFrame:
    fs = get_filtered_sales(dr_min, dr_max, sel_store, sel_cat)
    return (fs.groupby(["Month_Num", "Month"], observed=True)["TotalAmount"]
              .sum().reset_index().sort_values("Month_Num"))


@st.cache_data(show_spinner=False)
def get_category_revenue(dr_min, dr_max, sel_store, sel_cat) -> pd.DataFrame:
    fs = get_filtered_sales(dr_min, dr_max, sel_store, sel_cat)
    return fs.groupby("Category", observed=True)["TotalAmount"].sum().reset_index()


@st.cache_data(show_spinner=False)
def get_top_medicines(dr_min, dr_max, sel_store, sel_cat, n: int = 10) -> pd.DataFrame:
    fs = get_filtered_sales(dr_min, dr_max, sel_store, sel_cat)
    return (fs.groupby("Medicine_Name", observed=True)["QuantitySold"]
              .sum().sort_values(ascending=False).head(n).reset_index())


@st.cache_data(show_spinner=False)
def get_store_revenue(dr_min, dr_max, sel_store, sel_cat) -> pd.DataFrame:
    fs = get_filtered_sales(dr_min, dr_max, sel_store, sel_cat)
    return (fs.groupby("Store_Name", observed=True)["TotalAmount"]
              .sum().sort_values().reset_index())


@st.cache_data(show_spinner=False)
def get_supplier_stock(sel_store, sel_cat, sel_supplier, sel_status) -> pd.DataFrame:
    fk = get_filtered_stock(sel_store, sel_cat, sel_supplier, sel_status)
    return (fk.groupby("Supplier_Name", observed=True)["QuantityOnHand"]
              .sum().sort_values(ascending=False).head(14).reset_index())


@st.cache_data(show_spinner=False)
def get_inventory_expiry_kpis(sel_store, sel_cat, sel_supplier, sel_status) -> dict:
    fk      = get_filtered_stock(sel_store, sel_cat, sel_supplier, sel_status)
    expst   = fk["ExpiryStatus"].to_numpy(dtype=str)
    qty     = fk["QuantityOnHand"].to_numpy()
    cost    = fk["UnitCost"].to_numpy()
    exp_msk = expst == "Expired"
    return {
        "expired":  int(exp_msk.sum()),
        "near_exp": int((expst == "Expiring Soon").sum()),
        "safe_cnt": int((expst == "Valid").sum()),
        "low_s":    int((qty < fk["ReorderPoint"].to_numpy()).sum()),
        "wastage":  float(np.dot(qty[exp_msk], cost[exp_msk])),
    }


@st.cache_data(show_spinner=False)
def get_batch_expiry_table(sel_store, sel_cat, sel_supplier, sel_status) -> pd.DataFrame:
    """
    Build and return the full batch expiry detail table.
    Uses np.select (vectorised) instead of apply(axis=1).
    """
    fk = get_filtered_stock(sel_store, sel_cat, sel_supplier, sel_status)
    cols = ["ProductName", "BatchNumber", "ExpiryDate", "DaysToExpiry",
            "ExpiryStatus", "QuantityOnHand", "ReorderPoint"]
    df = fk[cols].sort_values("DaysToExpiry").reset_index(drop=True)

    qty, rp = df["QuantityOnHand"].to_numpy(), df["ReorderPoint"].to_numpy()
    df["StockStatus"] = np.select(
        [qty == 0, qty < rp],
        ["Out of Stock", "Low Stock"],
        default="In Stock",
    )
    return df.drop(columns=["ReorderPoint"])


@st.cache_data(show_spinner=False)
def get_store_performance(dr_min, dr_max, sel_store, sel_cat, sel_supplier, sel_status) -> pd.DataFrame:
    fs  = get_filtered_sales(dr_min, dr_max, sel_store, sel_cat)
    fk  = get_filtered_stock(sel_store, sel_cat, sel_supplier, sel_status)

    sps = fs.groupby("Store_Name", observed=True).agg(
        Revenue=("TotalAmount", "sum"),
        UnitsSold=("QuantitySold", "sum"),
        Transactions=("TransactionID", "count"),
    ).reset_index()

    inv_s = (fk.groupby("Store_Name", observed=True)["QuantityOnHand"]
               .sum().reset_index().rename(columns={"QuantityOnHand": "Stock_OnHand"}))

    sps = sps.merge(inv_s, on="Store_Name", how="left")

    # Vectorised string formatting — avoids row-wise apply
    sps["Revenue"]      = sps["Revenue"].map("₹{:,.0f}".format)
    sps["UnitsSold"]    = sps["UnitsSold"].map("{:,}".format)
    sps["Stock_OnHand"] = sps["Stock_OnHand"].map("{:,.0f}".format)
    return sps


@st.cache_data(show_spinner=False)
def get_top15_medicines(dr_min, dr_max, sel_store, sel_cat) -> pd.DataFrame:
    fs = get_filtered_sales(dr_min, dr_max, sel_store, sel_cat)
    top = (fs.groupby(["Medicine_Name", "Category"], observed=True)
             .agg(Revenue=("TotalAmount", "sum"),
                  Units=("QuantitySold", "sum"),
                  Transactions=("TransactionID", "count"))
             .sort_values("Revenue", ascending=False).head(15).reset_index())
    top["Revenue"] = top["Revenue"].map("₹{:,.0f}".format)
    top["Units"]   = top["Units"].map("{:,}".format)
    return top


@st.cache_data(show_spinner=False)
def get_category_stock_health(sel_store, sel_cat, sel_supplier, sel_status) -> pd.DataFrame:
    """
    Replaces the groupby+apply(lambda) pattern with pure vectorised ops.
    Returns per-category Safe / LowStock counts.
    """
    fk  = get_filtered_stock(sel_store, sel_cat, sel_supplier, sel_status)
    qty = fk["QuantityOnHand"].to_numpy()
    rp  = fk["ReorderPoint"].to_numpy()
    cats = fk["Category"].to_numpy(dtype=str)

    safe_mask = qty >= rp
    # Build a DataFrame of one-hot columns, then groupby
    tmp = pd.DataFrame({"Category": cats, "Safe": safe_mask.astype(int),
                        "LowStock": (~safe_mask).astype(int)})
    return tmp.groupby("Category")[["Safe", "LowStock"]].sum().reset_index()


@st.cache_data(show_spinner=False)
def get_margin_by_category(dr_min, dr_max, sel_store, sel_cat) -> pd.DataFrame:
    """Compute average margin % per category using UnitPrice from sales and UnitCost joined from products."""
    fs = get_filtered_sales(dr_min, dr_max, sel_store, sel_cat)
    fs = fs.copy()
    fs["MarginPct"] = ((fs["UnitPrice"] - fs["UnitCost"]) / fs["UnitPrice"].replace(0, np.nan) * 100).round(1)
    return fs.groupby("Category", observed=True)["MarginPct"].mean().round(1).reset_index()


@st.cache_data(show_spinner=False)
def get_customer_type_revenue(dr_min, dr_max, sel_store, sel_cat) -> pd.DataFrame:
    """Revenue split by CustomerType (Hospital vs Retail)."""
    fs = get_filtered_sales(dr_min, dr_max, sel_store, sel_cat)
    return fs.groupby("CustomerType", observed=True)["TotalAmount"].sum().reset_index()


@st.cache_data(show_spinner=False)
def get_safety_stock_kpis(sel_store, sel_cat, sel_supplier, sel_status) -> dict:
    """Return counts for items below SafetyStock, below ReorderPoint, and out of stock."""
    fk  = get_filtered_stock(sel_store, sel_cat, sel_supplier, sel_status)
    qty = fk["QuantityOnHand"].to_numpy()
    ss  = fk["SafetyStock"].to_numpy()
    rp  = fk["ReorderPoint"].to_numpy()
    return {
        "below_safety":  int((qty < ss).sum()),
        "below_reorder": int((qty < rp).sum()),
        "out_of_stock":  int((qty == 0).sum()),
    }


@st.cache_data(show_spinner=False)
def get_supplier_risk(sel_store, sel_cat, sel_supplier, sel_status) -> pd.DataFrame:
    """Compute a risk score per supplier = (5 - QualityRating) * LeadTimeDays."""
    fk = get_filtered_stock(sel_store, sel_cat, sel_supplier, sel_status)
    risk = (fk.groupby("Supplier_Name", observed=True)
              .agg(QualityRating=("QualityRating", "mean"),
                   LeadTimeDays=("LeadTimeDays", "mean"))
              .reset_index())
    risk["RiskScore"] = ((5 - risk["QualityRating"]) * risk["LeadTimeDays"]).round(2)
    risk = risk.sort_values("RiskScore", ascending=False).head(12).reset_index(drop=True)
    risk["QualityRating"] = risk["QualityRating"].round(2)
    risk["LeadTimeDays"]  = risk["LeadTimeDays"].round(1)
    return risk


@st.cache_data(show_spinner=False)
def get_medicine_velocity(dr_min, dr_max, sel_store, sel_cat) -> pd.DataFrame:
    """Compute real fast vs slow moving split by medicine based on units/day sold."""
    fs = get_filtered_sales(dr_min, dr_max, sel_store, sel_cat)
    days = max((fs["Date"].max() - fs["Date"].min()).days, 1)
    vel  = (fs.groupby("Medicine_Name", observed=True)["QuantitySold"]
              .sum().reset_index())
    vel["UnitsPerDay"] = (vel["QuantitySold"] / days).round(3)
    median_vel = vel["UnitsPerDay"].median()
    vel["Segment"] = np.where(vel["UnitsPerDay"] >= median_vel, "Fast Moving", "Slow Moving")
    return vel


@st.cache_data(show_spinner=False)
def get_map_data(dr_min, dr_max, sel_store, sel_cat) -> pd.DataFrame:
    """Pre-compute store map data including lat/lon and size scaling."""
    fs = get_filtered_sales(dr_min, dr_max, sel_store, sel_cat)
    rev_map = fs.groupby("Store_Name", observed=True).agg(
        TotalAmount=("TotalAmount", "sum"),
        Units=("QuantitySold", "sum"),
    ).reset_index()

    # Cast to float64 explicitly — Store_Name is category dtype and .map() can
    # inherit it, causing Plotly's internal .mean() call to raise TypeError.
    rev_map["lat"] = rev_map["Store_Name"].map(
        lambda s: STORE_COORDS.get(s, (17.38, 78.46))[0]).astype("float64")
    rev_map["lon"] = rev_map["Store_Name"].map(
        lambda s: STORE_COORDS.get(s, (17.38, 78.46))[1]).astype("float64")

    max_rev = rev_map["TotalAmount"].max()
    rev_map["size"] = ((rev_map["TotalAmount"] / max_rev * 55) + 14).astype("float64")
    return rev_map


@st.cache_data(show_spinner=False)
def build_report(dr_min, dr_max, sel_store, sel_cat, sel_supplier, sel_status) -> bytes:
    """Rebuild Excel report only when filter params change."""
    fs  = get_filtered_sales(dr_min, dr_max, sel_store, sel_cat)
    fk  = get_filtered_stock(sel_store, sel_cat, sel_supplier, sel_status)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        fs.to_excel(writer, sheet_name="Sales", index=False)
        fk.to_excel(writer, sheet_name="Stock", index=False)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# AI INSIGHTS — Groq-powered Llama integration
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_ai_insights(summary_data: str, user_question: str | None = None,
                         chat_history: tuple = ()) -> str:
    """
    Call Groq API with pharmacy context data.
    Cached by (summary_data, user_question, chat_history) so repeat calls are free.
    chat_history: tuple of alternating (user_msg, assistant_msg) strings for multi-turn context.
    Returns the AI response string, or an error message.
    """
    if _groq_client is None:
        if not _GROQ_AVAILABLE:
            return "⚠️ **Groq package not installed.** Run: `pip install groq python-dotenv`"
        return "⚠️ **GROQ_API_KEY not found.** Add it to your `.env` file and restart the app."

    if user_question:
        system_prompt = (
            "You are a seasoned pharmacy owner who has run a multi-store pharmacy chain in Hyderabad for 15 years. "
            "You think in terms of cash flow, patient trust, staff accountability, and monthly profit margins. "
            "You've personally dealt with expired stock write-offs, supplier disputes, slow-moving SKUs, and peak-season demand spikes. "
            "Answer the user's question based strictly on the pharmacy dashboard data provided. "
            "Speak like an owner — direct, no-nonsense, grounded in rupees and real consequences. "
            "Use bullet points where appropriate and always tie your answer to the numbers given."
        )
        user_content = (
            f"My Pharmacy Dashboard Data:\n{summary_data}\n\n"
            f"My Question: {user_question}"
        )
    else:
        system_prompt = (
            "You are a seasoned pharmacy owner who has run a 5-store pharmacy chain in Hyderabad for 15 years. "
            "You think in terms of cash flow, patient trust, staff accountability, supplier relationships, and monthly profit. "
            "You've personally dealt with: expired stock write-offs that hurt your balance sheet, "
            "suppliers who deliver late and damage your fill rate, slow-moving SKUs tying up working capital, "
            "and seasonal demand spikes that catch you understocked. "
            "You are reviewing your own dashboard and identifying what needs to be fixed TODAY. "
            "\n\n"
            "Analyse the pharmacy metrics below and respond with exactly 8 Q&A pairs. "
            "Format EVERY pair EXACTLY like this:\n"
            "Q: <a sharp, owner-mindset question — the kind you'd ask your store manager>\n"
            "A:\n"
            "🔴 PROBLEM: <1–2 sentences — what is bleeding money or hurting the business RIGHT NOW, with exact numbers and ₹ values>\n"
            "✅ SOLUTION: <2–3 bullet points of owner-level actions — specific, immediate, and profit-focused. "
            "Think: negotiate with supplier, call the distributor, run a promotion, fire a reorder, transfer stock between stores, "
            "train staff, renegotiate terms, write off and claim GST input credit, implement auto-reorder triggers, etc.>\n\n"
            "Cover these 8 topics in order: "
            "1) Cash & Revenue Health, 2) Dead & Slow Stock, 3) Expiry Crisis & Write-offs, "
            "4) Stockout & Lost Sales Risk, 5) Supplier Performance, "
            "6) Top SKU Demand Gaps, 7) Store-level Underperformance, 8) 30-Day Action Plan. "
            "Every PROBLEM must feel urgent and real — use the actual numbers. "
            "Every SOLUTION must be something you as the owner would personally instruct your team to do this week. "
            "Use • for bullet points. No generic advice — make it specific to this pharmacy's data."
        )
        user_content = f"My Pharmacy Dashboard (review this as if it is your own business):\n\n{summary_data}"

    _MODELS = ["llama-3.3-70b-versatile", "llama3-8b-8192"]
    last_err = ""

    # Build messages array — system + optional history + current user turn
    _messages: list[dict] = [{"role": "system", "content": system_prompt}]
    # Replay prior turns (tuple of (user, assistant) pairs)
    for _u, _a in zip(chat_history[::2], chat_history[1::2]):
        _messages.append({"role": "user",      "content": _u})
        _messages.append({"role": "assistant", "content": _a})
    _messages.append({"role": "user", "content": user_content})

    for model in _MODELS:
        try:
            resp = _groq_client.chat.completions.create(
                model=model,
                messages=_messages,
                temperature=0.4,
                max_tokens=2000,
            )
            text = resp.choices[0].message.content
            if text and text.strip():
                return text.strip()
        except Exception as e:
            last_err = str(e)
            continue   # try fallback model

    # All models failed
    if "invalid_api_key" in last_err.lower() or "401" in last_err:
        return "⚠️ **Invalid API Key.** Check your GROQ_API_KEY in `.env`."
    if "quota" in last_err.lower() or "429" in last_err:
        return "⚠️ **API quota exceeded.** Please try again later or upgrade your Groq plan."
    if "timeout" in last_err.lower() or "connection" in last_err.lower():
        return "⚠️ **Connection error.** Check your internet connection and retry."
    return f"⚠️ **AI unavailable.** Error: {last_err or 'Unknown error'}. Please try again."


@st.cache_data(show_spinner=False)
def _build_ai_context(
    dr_min, dr_max, sel_store, sel_cat, sel_supplier, sel_status,
    total_rev, total_sv, low_stock, safe_pct, wastage, exp_kpis,
) -> str:
    """Assemble a compact text summary of current dashboard metrics for the AI."""
    top_meds     = get_top_medicines(dr_min, dr_max, sel_store, sel_cat, n=10)
    store_rev    = get_store_revenue(dr_min, dr_max, sel_store, sel_cat)
    sup_stock    = get_supplier_stock(sel_store, sel_cat, sel_supplier, sel_status)
    cat_rev      = get_category_revenue(dr_min, dr_max, sel_store, sel_cat)
    monthly_rev  = get_monthly_revenue(dr_min, dr_max, sel_store, sel_cat)

    lines = [
        "=== PHARMACY DASHBOARD SUMMARY ===",
        f"Active Filters → Store: {sel_store} | Category: {sel_cat} "
        f"| Supplier: {sel_supplier} | Status: {sel_status}",
        "",
        "--- KEY KPIs ---",
        f"Total Revenue:        ₹{total_rev:,.0f}",
        f"Total Stock Value:    ₹{total_sv:,.0f}",
        f"Low Stock Items:      {low_stock:,}",
        f"Inventory Health:     {safe_pct}%",
        f"Wastage Value:        ₹{wastage:,.0f}",
        f"Expired Medicines:    {exp_kpis['expired']:,}",
        f"Near-Expiry Meds:     {exp_kpis['near_exp']:,}",
        "",
        "--- TOP 10 MEDICINES BY UNITS SOLD ---",
        top_meds.to_string(index=False),
        "",
        "--- STORE REVENUE ---",
        store_rev.to_string(index=False),
        "",
        "--- TOP SUPPLIER STOCK CONTRIBUTION ---",
        sup_stock.to_string(index=False),
        "",
        "--- CATEGORY REVENUE ---",
        cat_rev.to_string(index=False),
        "",
        "--- MONTHLY REVENUE TREND ---",
        monthly_rev[["Month", "TotalAmount"]].to_string(index=False),
    ]

    # Add margin analysis
    mg_data = get_margin_by_category(dr_min, dr_max, sel_store, sel_cat)
    lines += [
        "",
        "--- MARGIN % BY CATEGORY ---",
        mg_data.rename(columns={"MarginPct": "Avg_Margin_%"}).to_string(index=False),
    ]
    _neg_margin = mg_data[mg_data["MarginPct"] < 0]
    if len(_neg_margin) > 0:
        lines += [
            "",
            "--- ⚠️ BELOW-COST CATEGORIES (SELLING AT A LOSS) ---",
            _neg_margin.rename(columns={"MarginPct": "Avg_Margin_%"}).to_string(index=False),
            "ACTION REQUIRED: These categories lose money on every unit sold.",
        ]

    # Add safety stock KPIs
    ss_kpis = get_safety_stock_kpis(sel_store, sel_cat, sel_supplier, sel_status)
    lines += [
        "",
        "--- SAFETY STOCK KPIs ---",
        f"Below Safety Stock:  {ss_kpis['below_safety']:,}",
        f"Below Reorder Point: {ss_kpis['below_reorder']:,}",
        f"Out of Stock:        {ss_kpis['out_of_stock']:,}",
    ]

    # Add net profit estimate
    _ctx_fs   = get_filtered_sales(dr_min, dr_max, sel_store, sel_cat)
    _ctx_cogs = float((_ctx_fs["QuantitySold"] * _ctx_fs["UnitCost"]).sum())
    _ctx_net  = total_rev - _ctx_cogs - wastage
    lines += [
        "",
        "--- PROFITABILITY ESTIMATE ---",
        f"Total Revenue: ₹{total_rev:,.0f}",
        f"Est. COGS:     ₹{_ctx_cogs:,.0f}",
        f"Wastage:       ₹{wastage:,.0f}",
        f"Est. Net Profit: ₹{_ctx_net:,.0f}",
        f"Est. GST Liability (~12%): ₹{total_rev * 0.12:,.0f}",
    ]

    # Add customer type split
    ct_data = get_customer_type_revenue(dr_min, dr_max, sel_store, sel_cat)
    if len(ct_data) > 0:
        total_ct = ct_data["TotalAmount"].sum()
        ct_data  = ct_data.copy()
        ct_data["Revenue_%"] = (ct_data["TotalAmount"] / max(total_ct, 1) * 100).round(1)
        lines += [
            "",
            "--- CUSTOMER TYPE REVENUE SPLIT ---",
            ct_data.to_string(index=False),
        ]

    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def get_expiring_soon_n(days: int = 7, sel_store: str = "All", sel_cat: str = "All",
                        sel_supplier: str = "All", sel_status: str = "All") -> dict:
    """
    Return expiry counts for items expiring within `days` days.
    BUG FIX: previously used ProductName.nunique() which returned only 6 because
    the dataset has 6 medicine name templates shared across 170 unique ProductIDs.
    Now returns batches (stock rows), SKUs (unique ProductIDs), and qty at risk.
    """
    fk = get_filtered_stock(sel_store, sel_cat, sel_supplier, sel_status)
    fk = fk[fk["DaysToExpiry"].between(0, days, inclusive="both")]
    return {
        "batches":   int(len(fk)),
        "skus":      int(fk["ProductID"].nunique()),
        "med_names": int(fk["ProductName"].nunique()),
        "total_qty": int(fk["QuantityOnHand"].sum()),
    }


@st.cache_data(show_spinner=False)
def get_out_of_stock_by_store(sel_store: str = "All", sel_cat: str = "All",
                               sel_supplier: str = "All", sel_status: str = "All") -> pd.DataFrame:
    """
    Return store + product combos that are Out of Stock, respecting sidebar filters.

    BUG FIX: was using [Store_Name, ProductName].drop_duplicates() which collapsed
    all ProductIDs sharing the same name into one row — only 6 medicine names exist
    in the dataset, giving a max of 30 combos instead of the real 93+ unique SKU+store OOS.
    Now uses ProductID for deduplication and keeps ProductName for display only.
    """
    fk  = get_filtered_stock(sel_store, sel_cat, sel_supplier, sel_status)
    oos = (fk[fk["QuantityOnHand"] == 0]
           [["Store_Name", "ProductID", "ProductName", "Category", "BatchNumber"]]
           .drop_duplicates(subset=["Store_Name", "ProductID"]))   # ← dedupe on SKU, not name
    return oos


@st.cache_data(show_spinner=False)
def get_supplier_last_delivery_days(sel_store: str = "All", sel_cat: str = "All",
                                    sel_supplier: str = "All") -> pd.DataFrame:
    """
    Rank suppliers by a composite Risk Score that prioritises ACTUAL stockout impact:

      RiskScore = OOS_batches        × 10   ← dominant: supplier causing real stockouts
               + LowStock_batches    × 3    ← urgent: about to go OOS
               + Expiring_batches    × 1    ← secondary: stock at risk of write-off
               + delay_days          × 0.3  ← delivery recency proxy
               + lead_time_days      × 0.5  ← recovery difficulty
               + (5 − quality)       × 2    ← chronic quality risk

    Previously sorted purely by EstDaysSinceDelivery, which surfaced suppliers
    with 0 OOS batches (e.g. Wockhardt with 49-day delay but no stockouts) while
    hiding suppliers like Abbott India (2 OOS + 109 low-stock batches) that are
    actually causing inventory problems right now.
    """
    df = stock_df.copy()

    if sel_store != "All" and "Store_Name" in df.columns:
        df = df[df["Store_Name"] == sel_store]
    if sel_cat != "All" and "Category" in df.columns:
        df = df[df["Category"] == sel_cat]
    if sel_supplier != "All" and "Supplier_Name" in df.columns:
        df = df[df["Supplier_Name"] == sel_supplier]

    if df.empty:
        return pd.DataFrame(columns=["Supplier_Name", "EstDaysSinceDelivery", "OOSRate",
                                     "OOSBatches", "LowStockBatches", "LeadTimeDays",
                                     "QualityRating", "RiskScore"])

    df["_expiry"]    = pd.to_datetime(df["ExpiryDate"], errors="coerce")
    df["_is_oos"]    = df["QuantityOnHand"] == 0
    df["_is_low"]    = (df["QuantityOnHand"] > 0) & (df["QuantityOnHand"] < df["ReorderPoint"])
    df["_is_expiring"] = df["DaysToExpiry"].between(0, 30)

    SHELF = 730
    today = pd.Timestamp.today().normalize()

    grp = df.groupby("Supplier_Name", observed=True).agg(
        MaxExpiry       = ("_expiry",       "max"),
        TotalBatches    = ("BatchNumber",   "count"),
        OOSBatches      = ("_is_oos",       "sum"),
        LowStockBatches = ("_is_low",       "sum"),
        ExpiringBatches = ("_is_expiring",  "sum"),
        LeadTimeDays    = ("LeadTimeDays",  "first"),
        QualityRating   = ("QualityRating", "first"),
    ).reset_index()

    grp["EstDeliveryDate"]      = grp["MaxExpiry"] - pd.Timedelta(days=SHELF)
    grp["EstDaysSinceDelivery"] = (today - grp["EstDeliveryDate"]).dt.days.clip(lower=0).astype(int)
    grp["OOSRate"]              = (grp["OOSBatches"] / grp["TotalBatches"].clip(lower=1) * 100).round(1)
    grp["LowStockRate"]         = (grp["LowStockBatches"] / grp["TotalBatches"].clip(lower=1) * 100).round(1)

    grp["RiskScore"] = (
        grp["OOSBatches"]           * 10
      + grp["LowStockBatches"]      * 3
      + grp["ExpiringBatches"]      * 1
      + grp["EstDaysSinceDelivery"] * 0.3
      + grp["LeadTimeDays"]         * 0.5
      + (5 - grp["QualityRating"])  * 2
    ).round(1)

    return (grp[["Supplier_Name", "EstDaysSinceDelivery", "OOSRate", "OOSBatches",
                 "LowStockBatches", "LowStockRate", "LeadTimeDays", "QualityRating", "RiskScore"]]
            .sort_values("RiskScore", ascending=False)
            .head(5)
            .reset_index(drop=True))


@st.cache_data(show_spinner=False)
def get_reorder_list(sel_store: str = "All", sel_cat: str = "All",
                     sel_supplier: str = "All", sel_status: str = "All") -> pd.DataFrame:
    """Return items below ReorderPoint with estimated reorder quantity, respecting sidebar filters."""
    fk = get_filtered_stock(sel_store, sel_cat, sel_supplier, sel_status)
    mask = fk["QuantityOnHand"] < fk["ReorderPoint"]
    low  = fk[mask][["ProductName", "Store_Name", "Supplier_Name",
                      "QuantityOnHand", "ReorderPoint", "SafetyStock",
                      "UnitCost", "Category"]].copy()
    low["SuggestedOrder"] = (low["ReorderPoint"] * 1.5 - low["QuantityOnHand"]).clip(lower=1).round(0).astype(int)
    low["EstOrderValue"]  = (low["SuggestedOrder"] * low["UnitCost"]).round(2)
    low = low.sort_values("QuantityOnHand").reset_index(drop=True)
    return low


@st.cache_data(show_spinner=False)
def get_expiry_action_table(days: int = 30, sel_store: str = "All", sel_cat: str = "All",
                             sel_supplier: str = "All", sel_status: str = "All") -> pd.DataFrame:
    """Return items expiring within `days` days for expiry action decisions, respecting sidebar filters."""
    fk = get_filtered_stock(sel_store, sel_cat, sel_supplier, sel_status)
    fk = fk[fk["DaysToExpiry"].between(0, days, inclusive="both")].copy()
    cols = ["ProductName", "BatchNumber", "Store_Name", "Supplier_Name",
            "DaysToExpiry", "ExpiryDate", "QuantityOnHand", "UnitCost", "RetailPrice"]
    available_cols = [c for c in cols if c in fk.columns]
    df = fk[available_cols].sort_values("DaysToExpiry").reset_index(drop=True)
    df["StockValue"] = (df["QuantityOnHand"] * df["UnitCost"]).round(2)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# UI HELPERS  — thin, reusable, zero Streamlit state inside
# ═══════════════════════════════════════════════════════════════════════════════

def glass_fig(fig: go.Figure, h: int = 320) -> go.Figure:
    """Apply shared chart layout (transparent background, branded grid)."""
    fig.update_layout(
        height=h,
        margin=dict(l=0, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        font=dict(family="Nunito", color="#0d2f52", size=12),
        legend=dict(
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(200,220,240,0.6)",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(180,210,235,0.45)",
        linecolor="rgba(180,210,235,0.4)",
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="rgba(180,210,235,0.45)",
        linecolor="rgba(180,210,235,0.4)",
        zeroline=False,
    )
    return fig


def kpi(val: str, lbl: str, cls: str = "") -> str:
    """Return KPI card HTML fragment."""
    return (f'<div class="kpi-card {cls}">'
            f'<div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>')


def fmt(n: float) -> str:
    """Compact number formatter: 1234567 → '1.23M', 12345 → '12.3K'."""
    n = abs(n)
    if n >= 1_000_000: return f"{n/1e6:.2f}M"
    if n >= 1_000:     return f"{n/1e3:.1f}K"
    return str(int(n))


def sec(title: str) -> None:
    """Render a section-title bar."""
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)


def render_chart(fig: go.Figure, h: int = 320, key: str | None = None) -> None:
    """Render a Plotly chart with shared glass layout and static rendering hint."""
    st.plotly_chart(
        glass_fig(fig, h),
        use_container_width=True,
        # config disables heavy mode-bar hover calculations
        config={"displayModeBar": False, "staticPlot": False},
        key=key,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  — Navigation + Filters inside st.form (single rerun on Apply)
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:15px 0 10px'>
      <img src='https://cdn-icons-png.flaticon.com/512/2966/2966486.png' width='72'
           style='filter:drop-shadow(0 4px 10px rgba(0,0,0,0.30));margin-bottom:8px'>
      <div style='font-family:Rajdhani;font-size:1.75rem;font-weight:700;
                  color:#fff;letter-spacing:2.5px;text-shadow:0 2px 10px rgba(0,0,0,0.35)'>
        💊 PHARMA<span style='color:#64B5F6'>DASH</span>
      </div>
      <div style='font-size:0.66rem;color:rgba(255,255,255,0.55);letter-spacing:1.5px;margin-top:2px'>
        INVENTORY &amp; SALES ANALYTICS
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(
        "<p style='color:#7ec8e8;font-size:0.72rem;font-weight:800;"
        "letter-spacing:1.5px;margin:0 0 7px'>📌 NAVIGATION</p>",
        unsafe_allow_html=True,
    )
    _NAV_ITEMS = [
        "🏠  Home",
        "☀️  Morning Briefing",
        "📊  Overview",
        "🧪  Inventory & Expiry",
        "📈  Sales & Demand",
        "🏪  Supplier & Store",
        "💡  Business Insights",
        "ℹ️  About",
    ]
    if "active_page" not in st.session_state:
        st.session_state.active_page = _NAV_ITEMS[0]
    for _item in _NAV_ITEMS:
        _active = st.session_state.active_page == _item
        st.markdown(f"""
        <style>
        div[data-testid="stButton"] button[kind="secondary"].nav-btn-{_item[:3].strip()}{{
            background:{'rgba(255,255,255,0.25)' if _active else 'rgba(255,255,255,0.08)'} !important;
            border:{'2px solid rgba(255,255,255,0.70)' if _active else '1px solid rgba(255,255,255,0.18)'} !important;
        }}
        </style>""", unsafe_allow_html=True)
        if st.button(
            _item,
            key=f"nav_{_item}",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.active_page = _item
            st.rerun()
    page = st.session_state.active_page

    st.markdown("---")
    st.markdown(
        "<p style='color:#7ec8e8;font-size:0.72rem;font-weight:800;"
        "letter-spacing:1.5px;margin:0 0 7px'>🔽 FILTERS</p>",
        unsafe_allow_html=True,
    )

    # ── Session-state defaults (written once) ─────────────────────────────────
    _DEFAULTS: dict = {
        "dr_min": DATE_MIN, "dr_max": DATE_MAX,
        "sel_store": "All", "sel_cat": "All",
        "sel_supplier": "All", "sel_status": "All",
    }
    for _k, _v in _DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # AI assistant state — persists across reruns, never reset by filters
    for _ai_key in ("ai_insights_result", "ai_chat_result", "ai_chat_question"):
        if _ai_key not in st.session_state:
            st.session_state[_ai_key] = None
    # Multi-turn chat history: list of {"role": "user"|"assistant", "content": str}
    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []

    # Expiry action log — persists across reruns
    if "expiry_actions" not in st.session_state:
        st.session_state["expiry_actions"] = {}   # key: (ProductName, BatchNumber, Store_Name) → action str

    def _clear_filters() -> None:
        """Reset all filter values AND widget keys to defaults so Streamlit
        renders them showing 'All' / full date range on next rerun."""
        # Reset backing state values
        for _k, _v in _DEFAULTS.items():
            st.session_state[_k] = _v
        # Overwrite widget keys directly — this is what Streamlit actually
        # reads to render the widget, so it must match the default value
        st.session_state["sel_store_widget"]    = "All"
        st.session_state["sel_cat_widget"]      = "All"
        st.session_state["sel_supplier_widget"] = "All"
        st.session_state["sel_status_widget"]   = "All"
        st.session_state["date_range_input"]    = (DATE_MIN, DATE_MAX)

    date_range   = st.date_input("📅 Date Range",
                                 value=(st.session_state.dr_min, st.session_state.dr_max),
                                 min_value=DATE_MIN,
                                 max_value=DATE_MAX,
                                 key="date_range_input")
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        st.session_state.dr_min = date_range[0]
        st.session_state.dr_max = date_range[1]

    sel_store    = st.selectbox("🏪 Store Name",    _STORE_OPTS,
                                key="sel_store_widget")
    sel_cat      = st.selectbox("💊 Category",      _CAT_OPTS,
                                key="sel_cat_widget")
    sel_supplier = st.selectbox("🏭 Supplier Name", _SUPPLIER_OPTS,
                                key="sel_supplier_widget")
    sel_status   = st.selectbox("📦 Stock Status",  _STATUS_OPTS,
                                key="sel_status_widget")

    st.session_state.sel_store    = sel_store
    st.session_state.sel_cat      = sel_cat
    st.session_state.sel_supplier = sel_supplier
    st.session_state.sel_status   = sel_status

    st.button("🗑️ Clear All Filters", on_click=_clear_filters,
              use_container_width=True, type="primary")
    # Download — cached by filter params; only re-builds when filters change
    st.download_button(
        label="⬇ Download Filtered Report",
        data=build_report(
            st.session_state.dr_min, st.session_state.dr_max,
            st.session_state.sel_store, st.session_state.sel_cat,
            st.session_state.sel_supplier, st.session_state.sel_status,
        ),
        file_name="PharmaDash_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.markdown("---")
    st.markdown(
        "<p style='color:rgba(255,255,255,0.30);font-size:0.67rem;"
        "text-align:center'>© 2026 PharmaDash Analytics</p>",
        unsafe_allow_html=True,
    )


# ── Resolve active filter values from session state (single source of truth) ──
DR_MIN    = st.session_state.dr_min
DR_MAX    = st.session_state.dr_max
SEL_STORE = st.session_state.sel_store
SEL_CAT   = st.session_state.sel_cat
SEL_SUP   = st.session_state.sel_supplier
SEL_STS   = st.session_state.sel_status


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div style="text-align:center;padding:30px 0 10px">
      <div style="font-family:'Rajdhani',sans-serif;font-size:3.2rem;font-weight:700;
                  color:#0d3d6e;letter-spacing:3px;
                  text-shadow:0 2px 18px rgba(255,255,255,0.95),0 1px 4px rgba(21,101,192,0.20)">
        💊 PHARMA<span style="color:#1976D2">DASH</span>
      </div>
      <div style="font-size:1.05rem;color:#1a4a70;margin-top:8px;font-weight:600;letter-spacing:1px">
        Pharmacy Inventory &amp; Sales Analytics Platform
      </div>
      <div style="width:80px;height:4px;background:linear-gradient(90deg,#1565C0,#00BCD4);
                  border-radius:2px;margin:16px auto 0"></div>
    </div>
    """, unsafe_allow_html=True)

    # Hero banner
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(21,101,192,0.88),rgba(0,188,212,0.82));border:1px solid rgba(255,255,255,0.35);
                border-radius:20px;padding:32px 36px;margin:18px 0 24px;
                box-shadow:0 8px 32px rgba(21,101,192,0.28);position:relative;overflow:hidden">
      <div style="position:absolute;top:-30px;right:-30px;width:180px;height:180px;border-radius:50%;
                  background:rgba(255,255,255,0.08)"></div>
      <div style="position:absolute;bottom:-40px;left:-20px;width:140px;height:140px;border-radius:50%;
                  background:rgba(255,255,255,0.06)"></div>
      <div style="font-family:'Rajdhani',sans-serif;font-size:1.6rem;font-weight:700;color:#fff;
                  margin-bottom:10px;letter-spacing:0.8px">
        🚀 Welcome to PharmaDash
      </div>
      <div style="font-size:0.95rem;color:rgba(255,255,255,0.90);line-height:1.7;max-width:720px">
        A powerful, real-time analytics dashboard designed for pharmacy chains across Hyderabad.
        Monitor inventory health, track expiry risks, analyse sales trends, and make data-driven
        procurement decisions — all from one unified platform.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards row 1
    c1, c2, c3 = st.columns(3)
    _FEATURES = [
        ("📊", "Overview", "#1565C0", "#42A5F5",
         "High-level KPIs covering stock value, revenue, units sold, and inventory health across all stores and categories."),
        ("🧪", "Inventory & Expiry", "#6A1B9A", "#AB47BC",
         "Track expired, near-expiry, and safe stock. Drill into batch-level expiry details with interactive filters."),
        ("📈", "Sales & Demand", "#00695C", "#26A69A",
         "Monthly revenue trends, top-selling medicines, category performance, and customer-type breakdowns."),
    ]
    for col, (icon, title, c1c, c2c, desc) in zip([c1, c2, c3], _FEATURES):
        col.markdown(f"""
        <div style="background:linear-gradient(135deg,{c1c}CC,{c2c}BB);border:1px solid rgba(255,255,255,0.30);
                    border-radius:16px;padding:22px 20px;height:100%;
                    box-shadow:0 4px 20px rgba(0,0,0,0.12);
                    transition:transform 0.2s">
          <div style="font-size:2rem;margin-bottom:10px">{icon}</div>
          <div style="font-family:'Rajdhani',sans-serif;font-size:1.1rem;font-weight:700;
                      color:#fff;margin-bottom:8px">{title}</div>
          <div style="font-size:0.82rem;color:rgba(255,255,255,0.88);line-height:1.6">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Feature cards row 2
    c4, c5, c6 = st.columns(3)
    _FEATURES2 = [
        ("🏪", "Supplier & Store", "#BF360C", "#FF7043",
         "Compare store revenues on an interactive Hyderabad map. Evaluate supplier quality ratings and lead times."),
        ("💡", "Business Insights", "#1B5E20", "#66BB6A",
         "Actionable recommendations powered by live data — expiry alerts, reorder triggers, and demand forecasting guidance."),
        ("🔽", "Smart Filters", "#0277BD", "#29B6F6",
         "Filter by date range, store, category, supplier, and stock status. Clear all filters instantly with one click."),
    ]
    for col, (icon, title, c1c, c2c, desc) in zip([c4, c5, c6], _FEATURES2):
        col.markdown(f"""
        <div style="background:linear-gradient(135deg,{c1c}CC,{c2c}BB);border:1px solid rgba(255,255,255,0.30);
                    border-radius:16px;padding:22px 20px;height:100%;
                    box-shadow:0 4px 20px rgba(0,0,0,0.12)">
          <div style="font-size:2rem;margin-bottom:10px">{icon}</div>
          <div style="font-family:'Rajdhani',sans-serif;font-size:1.1rem;font-weight:700;
                      color:#fff;margin-bottom:8px">{title}</div>
          <div style="font-size:0.82rem;color:rgba(255,255,255,0.88);line-height:1.6">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box" style="text-align:center;margin-top:8px">
      <div style="font-size:0.88rem;color:#1a3a55">
        👈 Use the <b>sidebar navigation</b> to explore each section of the dashboard.
        Apply <b>filters</b> to slice data by store, category, supplier, or date range.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 0B — MORNING BRIEFING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "☀️  Morning Briefing":
    import datetime as _dt
    _now = _dt.datetime.now()
    _greeting = "Good Morning" if _now.hour < 12 else ("Good Afternoon" if _now.hour < 17 else "Good Evening")

    st.markdown(f"""
    <div style="text-align:center;padding:20px 0 6px">
      <div style="font-family:'Rajdhani',sans-serif;font-size:2.4rem;font-weight:700;
                  color:#0d3d6e;letter-spacing:2px;
                  text-shadow:0 2px 18px rgba(255,255,255,0.95)">
        ☀️ {_greeting}, Pharmacy Owner
      </div>
      <div style="font-size:0.85rem;color:#4a6a8a;margin-top:4px;letter-spacing:0.5px">
        {_now.strftime("%A, %d %B %Y  ·  %I:%M %p")}  ·  Here's what needs your attention today
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pull briefing data ──────────────────────────────────────────────────────
    _exp7   = get_expiring_soon_n(7,  SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
    _exp30  = get_expiring_soon_n(30, SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
    _oos_df = get_out_of_stock_by_store(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
    _sup_df = get_supplier_last_delivery_days(SEL_STORE, SEL_CAT, SEL_SUP)
    _reorder_df = get_reorder_list(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
    _exp_action_df = get_expiry_action_table(30, SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)

    # ── Unpack expiry dicts (now returns batches/skus/qty, not just one int) ──
    _exp7_batches  = _exp7["batches"]    # total stock-batch rows ≤7 days
    _exp7_skus     = _exp7["skus"]       # unique product IDs ≤7 days
    _exp7_qty      = _exp7["total_qty"]  # total units at risk ≤7 days
    _exp30_batches = _exp30["batches"]
    _exp30_skus    = _exp30["skus"]

    _oos_count  = len(_oos_df)
    # Top delayed supplier — used for checklist task
    _top_sup      = _sup_df.iloc[0]["Supplier_Name"]       if len(_sup_df) > 0 else "N/A"
    _top_sup_days = int(_sup_df.iloc[0]["EstDaysSinceDelivery"]) if len(_sup_df) > 0 else 0
    _reorder_count = len(_reorder_df)
    _reorder_val   = float(_reorder_df["EstOrderValue"].sum()) if len(_reorder_df) > 0 else 0.0
    _logged_actions = len(st.session_state["expiry_actions"])

    # ── ALERT STRIP — cards with titles ───────────────────────────────────────
    def _alert_pill(icon, title, text, bg, border, title_color):
        return (
            f'<div style="flex:1;min-width:220px;background:{bg};border:1.5px solid {border};'
            f'border-radius:14px;padding:16px 20px 18px;display:flex;flex-direction:column;gap:10px;'
            f'box-shadow:0 3px 14px rgba(0,0,0,0.08)">'
            # Card header row: icon + title
            f'<div style="display:flex;align-items:center;gap:10px;border-bottom:1px solid {border};padding-bottom:10px">'
            f'<span style="font-size:1.6rem;line-height:1">{icon}</span>'
            f'<span style="font-size:0.80rem;font-weight:900;color:{title_color};'
            f'text-transform:uppercase;letter-spacing:1px">{title}</span>'
            f'</div>'
            # Card body
            f'<span style="font-size:0.88rem;color:#1a2a3a;font-weight:600;line-height:1.6">{text}</span>'
            f'</div>'
        )

    # Card 1 — Expiry Alert  (FIX: now shows SKU count not medicine-name count)
    _pill1_text = (
        f"<b style='color:#B71C1C;font-size:1.1rem'>{_exp7_skus:,} SKUs</b> "
        f"expiring in the next 7 days.<br>"
        f"<span style='color:#7f0000;font-size:0.82rem'>"
        f"{_exp7_batches:,} batch-lots · {_exp7_qty:,} units at risk</span><br>"
        f"<span style='color:#6a0000;font-size:0.80rem'>"
        f"{_exp30_skus:,} SKUs ({_exp30_batches:,} batches) within 30 days</span>"
    )

    # Card 2 — Out of Stock  (FIX: now counts unique ProductID+Store combos, not ProductName+Store)
    _oos_detail = ""
    if len(_oos_df) > 0:
        _by_store = _oos_df.groupby("Store_Name", observed=True)["ProductID"].count()
        _oos_detail = "<br>".join(
            [f"<span style='color:#7a3000;font-size:0.79rem'>{s}: {c} SKU{'s' if c>1 else ''} OOS</span>"
             for s, c in _by_store.items()]
        )
    _pill2_text = (
        f"<b style='color:#E65100;font-size:1.1rem'>{_oos_count:,} SKU–store combos</b><br>"
        f"are currently out of stock.<br>{_oos_detail}"
    )

    # Card 3 — Reorder Alert
    _pill3_text = (
        f"<b style='color:#1B5E20;font-size:1.1rem'>{_reorder_count:,} SKUs</b>"
        f" need reordering now.<br>"
        f"Estimated PO value: <b style='color:#1B5E20'>₹{_reorder_val:,.0f}</b>"
    )

    # Card 4 — Supplier Delay (OOS-weighted risk: suppliers actually causing stockouts)
    if len(_sup_df) > 0:
        _sup_rows = ""
        for _, row in _sup_df.iterrows():
            score = row["RiskScore"]
            oos_b = int(row["OOSBatches"])
            low_b = int(row["LowStockBatches"])
            if score >= 300:
                badge_col, badge = "#B71C1C", "🔴 CRITICAL"
            elif score >= 150:
                badge_col, badge = "#E65100", "🟠 HIGH"
            else:
                badge_col, badge = "#F57F17", "🟡 MED"

            _sup_rows += (
                f'<div style="border-bottom:1px solid rgba(74,20,140,0.12);padding:6px 0">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-weight:700;color:#2d0060;font-size:0.82rem">{row["Supplier_Name"]}</span>'
                f'<span style="font-size:0.72rem;font-weight:700;color:{badge_col}">{badge}</span>'
                f'</div>'
                f'<div style="font-size:0.76rem;color:#4a2270;margin-top:3px;display:flex;gap:10px;flex-wrap:wrap">'
                f'<span>🚫 <b style="color:#B71C1C">{oos_b} OOS</b> batches</span>'
                f'<span>⚠️ <b style="color:#E65100">{low_b} low-stock</b></span>'
                f'<span>🚚 {int(row["LeadTimeDays"])}d lead</span>'
                f'<span>⭐ {row["QualityRating"]:.1f}</span>'
                f'</div>'
                f'</div>'
            )
        _pill4_text = (
            f"<b style='color:#4A148C;font-size:0.88rem'>Suppliers causing stockouts</b>"
            f"<span style='color:#7a50a0;font-size:0.74rem'> (ranked by OOS impact)</span>"
            f"<div style='margin-top:6px'>{_sup_rows}</div>"
        )
    else:
        _pill4_text = "<span style='color:#4A148C'>No supplier data for selected filters.</span>"

    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:12px;margin:22px 0 24px">
      {_alert_pill("💊", "⚠️ Expiry Alert", _pill1_text,
                   "linear-gradient(135deg,rgba(255,235,238,0.95),rgba(255,205,210,0.88))",
                   "rgba(183,28,28,0.35)", "#B71C1C")}
      {_alert_pill("🏪", "🔴 Out of Stock", _pill2_text,
                   "linear-gradient(135deg,rgba(255,243,224,0.95),rgba(255,224,178,0.88))",
                   "rgba(230,81,0,0.35)", "#BF360C")}
      {_alert_pill("📦", "🛒 Reorder Now", _pill3_text,
                   "linear-gradient(135deg,rgba(232,245,233,0.95),rgba(200,230,201,0.88))",
                   "rgba(27,94,32,0.35)", "#1B5E20")}
      {_alert_pill("🚚", "🏭 Supplier Delay", _pill4_text,
                   "linear-gradient(135deg,rgba(243,229,245,0.95),rgba(225,190,231,0.88))",
                   "rgba(74,20,140,0.35)", "#4A148C")}
    </div>
    """, unsafe_allow_html=True)

    # ── TODAY'S ACTION CHECKLIST — per store tabs ─────────────────────────────
    st.markdown('<div class="sec-title">✅ Today\'s 30-Second Action Checklist</div>', unsafe_allow_html=True)

    def _build_store_tasks(store_name: str) -> list[tuple[str, str]]:
        """Build task list for a specific store."""
        _s_exp7    = get_expiring_soon_n(7,  store_name, SEL_CAT, SEL_SUP, SEL_STS)
        _s_reorder = get_reorder_list(store_name, SEL_CAT, SEL_SUP, SEL_STS)
        _s_oos_df  = get_out_of_stock_by_store(store_name, SEL_CAT, SEL_SUP, SEL_STS)
        _s_sup_df  = get_supplier_last_delivery_days(store_name, SEL_CAT, SEL_SUP)

        tasks = []

        # Expiry
        if _s_exp7["skus"] > 0:
            tasks.append(("🔴", (
                f"<b>{_s_exp7['skus']:,} SKUs</b> expiring in 7 days "
                f"({_s_exp7['batches']:,} batches · {_s_exp7['total_qty']:,} units) "
                f"— discount, return, or write-off"
            )))

        # OOS
        _s_oos_count = len(_s_oos_df)
        if _s_oos_count > 0:
            # top 3 OOS product names for context
            _oos_names = ", ".join(_s_oos_df["ProductName"].value_counts().head(3).index.tolist())
            tasks.append(("🟠", (
                f"<b>{_s_oos_count:,} SKUs out of stock</b> — top items: {_oos_names}"
            )))

        # Reorder
        _s_reorder_count = len(_s_reorder)
        _s_reorder_val   = float(_s_reorder["EstOrderValue"].sum()) if _s_reorder_count > 0 else 0
        if _s_reorder_count > 0:
            tasks.append(("🟡", (
                f"<b>{_s_reorder_count:,} SKUs</b> below reorder point "
                f"— estimated PO ₹{_s_reorder_val:,.0f}"
            )))

        # Supplier delay — only for this store's suppliers
        if len(_s_sup_df) > 0:
            _s_top_sup  = _s_sup_df.iloc[0]["Supplier_Name"]
            _s_top_days = int(_s_sup_df.iloc[0]["EstDaysSinceDelivery"])
            if _s_top_days > 10:
                tasks.append(("🟣", (
                    f"<b>Call {_s_top_sup}</b> — no delivery in ~{_s_top_days} days "
                    f"(risk score: {_s_sup_df.iloc[0]['RiskScore']:.0f})"
                )))

        return tasks

    def _render_tasks_html(tasks: list) -> str:
        if not tasks:
            return '<div style="color:#2e7d32;font-weight:700;font-size:0.9rem;padding:10px">✅ All clear — no urgent actions flagged!</div>'
        return "".join([
            f'<div style="display:flex;align-items:flex-start;gap:11px;padding:10px 16px;'
            f'background:rgba(255,255,255,0.72);border-radius:10px;margin-bottom:7px;'
            f'border:1px solid rgba(200,220,240,0.70);font-size:0.88rem;color:#1a2d42;font-weight:500">'
            f'<span style="font-size:1.1rem;margin-top:1px">{sev}</span>'
            f'<span style="line-height:1.55">{txt}</span></div>'
            for sev, txt in tasks
        ])

    # All-stores tab + one tab per store
    _all_stores_list = sorted(stock_df["Store_Name"].dropna().unique().tolist())
    _tab_labels = ["🏬 All Stores"] + [f"🏪 {s.split()[0]}" for s in _all_stores_list]
    _checklist_tabs = st.tabs(_tab_labels)

    # Tab 0 — All stores (aggregated, current filter)
    with _checklist_tabs[0]:
        _all_tasks = []
        if _exp7_skus > 0:
            _all_tasks.append(("🔴", (
                f"<b>{_exp7_skus:,} SKUs</b> expiring in 7 days "
                f"({_exp7_batches:,} batches · {_exp7_qty:,} units) "
                f"— discount, return, or write-off"
            )))
        if _oos_count > 0:
            _top_oos_store = _oos_df["Store_Name"].value_counts().idxmax() if len(_oos_df) > 0 else ""
            _all_tasks.append(("🟠", f"<b>{_oos_count:,} SKU–store combos out of stock</b> — {_top_oos_store} most affected"))
        if _reorder_count > 0:
            _all_tasks.append(("🟡", f"<b>{_reorder_count:,} SKUs</b> need reordering — total PO ₹{_reorder_val:,.0f}"))
        if _top_sup_days > 10:
            _all_tasks.append(("🟣", f"<b>Call {_top_sup}</b> — no delivery in ~{_top_sup_days} days"))
        if _logged_actions > 0:
            _all_tasks.append(("🟢", f"<b>{_logged_actions} expiry actions</b> already logged today — review the Expiry Action table below"))
        st.markdown(f'<div style="margin:10px 0 18px">{_render_tasks_html(_all_tasks)}</div>', unsafe_allow_html=True)

    # Tabs 1-N — one per store
    for _ti, _sname in enumerate(_all_stores_list):
        with _checklist_tabs[_ti + 1]:
            _store_tasks = _build_store_tasks(_sname)
            st.markdown(
                f'<div style="margin:10px 0 18px">{_render_tasks_html(_store_tasks)}</div>',
                unsafe_allow_html=True,
            )

    # ── SECTION 2: REORDER LIST ────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📦 Reorder List — Items Below Reorder Point</div>', unsafe_allow_html=True)

    if len(_reorder_df) == 0:
        st.success("✅ No items below reorder point right now.")
    else:
        _ro_filtered = _reorder_df

        _ro = _ro_filtered[["ProductName", "Store_Name", "Supplier_Name", "Category",
                            "QuantityOnHand", "ReorderPoint", "SuggestedOrder", "EstOrderValue"]].copy()
        _ro.columns = ["Medicine", "Store", "Supplier", "Category",
                       "Qty On Hand", "Reorder Point", "Suggested Order", "Est. Order Value (₹)"]

        # Summary metrics
        _col1, _col2, _col3 = st.columns(3)
        _col1.metric("Total SKUs to Reorder", f"{len(_ro_filtered):,}")
        _col2.metric("Out-of-Stock Items", f"{int((_ro_filtered['QuantityOnHand'] == 0).sum()):,}")
        _col3.metric("Total PO Value", f"₹{float(_ro_filtered['EstOrderValue'].sum()):,.0f}")

        # Colour-code urgency
        def _urgency_color(row):
            if row["Qty On Hand"] == 0:
                return ["background-color:rgba(229,57,53,0.15)"] * len(row)
            elif row["Qty On Hand"] < row["Reorder Point"] / 2:
                return ["background-color:rgba(251,140,0,0.12)"] * len(row)
            return ["background-color:rgba(67,160,71,0.07)"] * len(row)

        _ro_display = _ro.copy()
        _ro_display["Est. Order Value (₹)"] = _ro_display["Est. Order Value (₹)"].map("₹{:,.0f}".format)
        styled_ro = _ro_display.style.apply(_urgency_color, axis=1)
        st.dataframe(styled_ro, use_container_width=True, height=360)

        # ── PDF DOWNLOAD ──────────────────────────────────────────────────────
        import io as _io
        try:
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib import colors as rl_colors
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            import datetime as _dt2

            def _build_reorder_pdf(df_in, store_label):
                _buf = _io.BytesIO()
                doc = SimpleDocTemplate(_buf, pagesize=landscape(A4),
                                        leftMargin=1.5*cm, rightMargin=1.5*cm,
                                        topMargin=1.5*cm, bottomMargin=1.5*cm)
                styles = getSampleStyleSheet()
                _title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                              fontSize=16, textColor=rl_colors.HexColor("#0d3d6e"),
                                              spaceAfter=4)
                _sub_style = ParagraphStyle("Sub", parent=styles["Normal"],
                                            fontSize=9, textColor=rl_colors.HexColor("#4a6a8a"),
                                            spaceAfter=12)

                elems = []
                elems.append(Paragraph("📦 Pharmacy Reorder List", _title_style))
                _now2 = _dt2.datetime.now().strftime("%d %b %Y, %I:%M %p")
                elems.append(Paragraph(f"Generated: {_now2}  ·  Store: {store_label}  ·  {len(df_in)} items", _sub_style))
                elems.append(Spacer(1, 0.3*cm))

                _headers = ["#", "Medicine", "Store", "Supplier", "Category",
                            "Qty\nOn Hand", "Reorder\nPoint", "Suggested\nOrder", "Est. Order\nValue (₹)"]
                _table_data = [_headers]
                for i, (_ridx, row) in enumerate(df_in.iterrows(), 1):
                    _table_data.append([
                        str(i),
                        str(row["Medicine"])[:30],
                        str(row["Store"])[:20],
                        str(row["Supplier"])[:22],
                        str(row["Category"])[:14],
                        str(int(row["Qty On Hand"])),
                        str(int(row["Reorder Point"])),
                        str(int(row["Suggested Order"])),
                        f"Rs.{float(str(row['Est. Order Value (₹)']).replace('₹','').replace(',','') or 0):,.0f}",
                    ])

                col_widths = [1*cm, 5.5*cm, 4*cm, 4.5*cm, 3*cm,
                              2*cm, 2.2*cm, 2.5*cm, 3.2*cm]
                tbl = Table(_table_data, colWidths=col_widths, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#0d3d6e")),
                    ("TEXTCOLOR",  (0, 0), (-1, 0), rl_colors.white),
                    ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE",   (0, 0), (-1, 0), 8),
                    ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTSIZE",   (0, 1), (-1, -1), 7.5),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [rl_colors.HexColor("#f0f6ff"), rl_colors.HexColor("#ffffff")]),
                    ("GRID",       (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#c0d4e8")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                # Red for OOS rows
                for i, (_ridx, row) in enumerate(df_in.iterrows(), 1):
                    if int(row["Qty On Hand"]) == 0:
                        tbl.setStyle(TableStyle([
                            ("BACKGROUND", (0, i), (-1, i), rl_colors.HexColor("#ffebee")),
                        ]))
                    elif int(row["Qty On Hand"]) < int(row["Reorder Point"]) // 2:
                        tbl.setStyle(TableStyle([
                            ("BACKGROUND", (0, i), (-1, i), rl_colors.HexColor("#fff3e0")),
                        ]))
                elems.append(tbl)
                doc.build(elems)
                _buf.seek(0)
                return _buf.getvalue()

            _pdf_bytes = _build_reorder_pdf(_ro_display, SEL_STORE if SEL_STORE != "All" else "All Stores")
            _fname = f"Reorder_List_{(SEL_STORE if SEL_STORE != 'All' else 'AllStores').replace(' ','_')}_{_dt.datetime.now().strftime('%d%b%Y')}.pdf"
            st.download_button(
                label="⬇️ Download Reorder List as PDF",
                data=_pdf_bytes,
                file_name=_fname,
                mime="application/pdf",
                use_container_width=True,
            )
        except ImportError:
            st.warning("📄 PDF export requires `reportlab`. Run: `pip install reportlab` then restart the app.")

    # ── SECTION 3: EXPIRY ACTION TABLE ────────────────────────────────────────
    st.markdown('<div class="sec-title" style="margin-top:28px">⏰ Expiry Action Table — Items Expiring in 30 Days</div>', unsafe_allow_html=True)

    if len(_exp_action_df) == 0:
        st.success("✅ No items expiring in the next 30 days.")
    else:
        _ea_filtered = _exp_action_df.copy()

        # ── Auto-suggest action based on days to expiry ────────────────────────
        def _auto_action(days):
            if days < 7:
                return "🔄 Return to Supplier"
            else:
                return "🏷️ Discount Sale"

        _ea_filtered["Suggested Action"] = _ea_filtered["DaysToExpiry"].apply(_auto_action)

        # ── Legend ─────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px">
          <div style="background:rgba(183,28,28,0.10);border:1px solid rgba(183,28,28,0.35);
                      border-radius:8px;padding:7px 14px;font-size:0.82rem;font-weight:700;color:#7f0000">
            🔴 &lt; 7 days → 🔄 Return to Supplier
          </div>
          <div style="background:rgba(230,81,0,0.10);border:1px solid rgba(230,81,0,0.35);
                      border-radius:8px;padding:7px 14px;font-size:0.82rem;font-weight:700;color:#7a3000">
            🟠 7–30 days → 🏷️ Discount Sale
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Summary metrics ────────────────────────────────────────────────────
        _ea1, _ea2, _ea3 = st.columns(3)
        _ea1.metric("Items Expiring ≤30 Days", f"{len(_ea_filtered):,}")
        _ea2.metric("Stock Value at Risk", f"₹{float(_ea_filtered['StockValue'].sum()):,.0f}")
        _auto_ret = int((_ea_filtered["Suggested Action"] == "🔄 Return to Supplier").sum())
        _ea3.metric("🔄 Return Suggested", f"{_auto_ret}")

        # ── Build display table ────────────────────────────────────────────────
        _ea_table = _ea_filtered[["ProductName", "Store_Name", "Supplier_Name",
                                   "DaysToExpiry", "ExpiryDate", "QuantityOnHand",
                                   "StockValue", "Suggested Action"]].copy()
        _ea_table.columns = ["Medicine", "Store", "Supplier",
                              "Days Left", "Expiry Date", "Qty",
                              "Stock Value (₹)", "Suggested Action"]
        _ea_table["Stock Value (₹)"] = _ea_table["Stock Value (₹)"].map("₹{:,.0f}".format)
        _ea_table["Qty"] = _ea_table["Qty"].astype(int)
        _ea_table["Days Left"] = _ea_table["Days Left"].astype(int)

        def _ea_row_color(row):
            d = row["Days Left"]
            if d < 7:
                return ["background-color:rgba(229,57,53,0.14)"] * len(row)
            return ["background-color:rgba(251,140,0,0.11)"] * len(row)

        _ea_styled = _ea_table.style.apply(_ea_row_color, axis=1)
        st.dataframe(_ea_styled, use_container_width=True, height=480)


# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Overview":
    st.markdown('<div class="page-title">📊 Pharmacy Inventory & Sales Dashboard</div>',
                unsafe_allow_html=True)

    kpis = get_overview_kpis(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
    _ss_kpis = get_safety_stock_kpis(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
    _ov_fk   = get_filtered_stock(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
    _ov_exp  = _ov_fk[_ov_fk["ExpiryStatus"] == "Expired"]
    _wastage_val = float(np.dot(_ov_exp["QuantityOnHand"].to_numpy(), _ov_exp["UnitCost"].to_numpy()))

    # ── Net Profit estimate (Revenue − COGS − Wastage) ────────────────────────
    _ov_fs      = get_filtered_sales(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
    _ov_cogs    = float((_ov_fs["QuantitySold"] * _ov_fs["UnitCost"]).sum())
    _ov_net_profit = kpis["total_rev"] - _ov_cogs - _wastage_val
    _ov_net_cls = "ok" if _ov_net_profit > 0 else "warn"
    # GST estimate: ~12% of revenue is a rough pharmacy GST assumption
    _ov_gst_est = kpis["total_rev"] * 0.12

    st.markdown(f"""<div class="kpi-row">
        {kpi(fmt(kpis["total_sv"]),                "Total Stock Value")}
        {kpi(f"₹{fmt(kpis['total_rev'])}",         "Total Revenue")}
        {kpi(f"₹{fmt(_ov_net_profit)}",            "Est. Net Profit",     _ov_net_cls)}
        {kpi(f"₹{fmt(_ov_gst_est)}",               "Est. GST Liability",  "teal")}
        {kpi(fmt(kpis["total_units"]),             "Total Units Sold")}
        {kpi(f"{kpis['expired_cnt']:,}",           "Expired Medicines",   "warn")}
        {kpi(f"₹{fmt(_wastage_val)}",              "Expired Wastage Value","warn")}
        {kpi(f"{kpis['safe_pct']}%",               "Inventory Health",    "ok")}
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        sec("🏆 Best Selling Medicines (Top 10)")
        top  = get_top_medicines(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT, 10)
        fig1 = px.bar(top, x="QuantitySold", y="Medicine_Name", orientation="h",
                      text="QuantitySold",
                      labels={"QuantitySold": "Units Sold", "Medicine_Name": ""})
        fig1.update_traces(marker_color=BLUES[:len(top)],
                           texttemplate="%{text:,}", textposition="outside")
        fig1.update_yaxes(categoryorder="total ascending")
        fig1.update_layout(xaxis=dict(range=[0, top["QuantitySold"].max() * 1.25]))
        render_chart(fig1, 370, key="ov_top_med")

    with c2:
        if SEL_STORE == "All":
            # ── All stores: show ranking chart ────────────────────────────────
            sec("🏪 Top Performing Stores")
            sr   = get_store_revenue(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
            fig2 = px.bar(sr, x="TotalAmount", y="Store_Name", orientation="h",
                          color="TotalAmount", color_continuous_scale="Blues",
                          text="TotalAmount",
                          labels={"TotalAmount": "Revenue (₹)", "Store_Name": ""})
            fig2.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
            fig2.update_layout(coloraxis_showscale=False,
                               xaxis=dict(range=[0, sr["TotalAmount"].max() * 1.28]))
            render_chart(fig2, 370, key="ov_store_rev")
        else:
            # ── Single store selected: show category breakdown for that store ─
            sec(f"📊 {SEL_STORE} — Revenue by Category")
            _sc_fs  = get_filtered_sales(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
            _sc_cat = (_sc_fs.groupby("Category", observed=True)["TotalAmount"]
                             .sum().reset_index().sort_values("TotalAmount", ascending=True))
            _sc_fig = px.bar(
                _sc_cat, x="TotalAmount", y="Category", orientation="h",
                color="TotalAmount", color_continuous_scale="Blues",
                text="TotalAmount",
                labels={"TotalAmount": "Revenue (₹)", "Category": ""},
            )
            _sc_fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
            _sc_fig.update_layout(
                coloraxis_showscale=False,
                xaxis=dict(range=[0, _sc_cat["TotalAmount"].max() * 1.32]),
            )
            render_chart(_sc_fig, 370, key="ov_store_cat_rev")

    c3, c4 = st.columns(2)

    with c3:
        sec("📅 Monthly Revenue Trend")
        mr   = get_monthly_revenue(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
        fig3 = px.area(mr, x="Month", y="TotalAmount", markers=True,
                       color_discrete_sequence=["#1976D2"],
                       labels={"TotalAmount": "Revenue (₹)", "Month": ""})
        fig3.update_traces(line_width=2.8, fillcolor="rgba(25,118,210,0.14)",
                           marker=dict(size=7, color="#1565C0",
                                       line=dict(width=2, color="#fff")))
        render_chart(fig3, 300, key="ov_monthly_rev")

    with c4:
        sec("🍩 Revenue by Category")
        cr   = get_category_revenue(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
        fig4 = px.pie(cr, names="Category", values="TotalAmount", hole=0.52,
                      color="Category", color_discrete_map=CAT_PALETTE)
        fig4.update_traces(textinfo="percent+label", pull=[0.03] * len(cr),
                           textfont_size=11)
        render_chart(fig4, 300, key="ov_cat_pie")

    sec("👥 Retail vs Hospital Revenue by Category")
    fs = get_filtered_sales(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
    ct = fs.groupby(["Category", "CustomerType"], observed=True)["TotalAmount"].sum().reset_index()
    fig5 = px.bar(ct, x="Category", y="TotalAmount", color="CustomerType", barmode="group",
                  color_discrete_map={"Retail": "#1976D2", "Hospital": "#00ACC1"},
                  labels={"TotalAmount": "Revenue (₹)", "Category": ""},
                  text="TotalAmount")
    fig5.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
    render_chart(fig5, 290, key="ov_retail_hospital")

    sec("📊 Category-wise Stock Health")
    ch   = get_category_stock_health(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
    fig6 = go.Figure([
        go.Bar(x=ch["Category"], y=ch["Safe"],     name="Safe Stock",
               marker_color="#43A047", text=ch["Safe"],
               texttemplate="%{text:,}", textposition="inside"),
        go.Bar(x=ch["Category"], y=ch["LowStock"], name="Low Stock",
               marker_color="#E53935", text=ch["LowStock"],
               texttemplate="%{text:,}", textposition="inside"),
    ])
    fig6.update_layout(barmode="stack", xaxis_title="", yaxis_title="SKU Count")
    render_chart(fig6, 290, key="ov_stock_health")

    c5, c6 = st.columns(2)

    with c5:
        sec("💰 Profit Margin % by Category")
        mg   = get_margin_by_category(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
        mg_colors = ["#E53935" if v < 0 else "#43A047" if v >= 50 else "#FB8C00"
                     for v in mg["MarginPct"]]
        fig_mg = px.bar(mg, x="Category", y="MarginPct",
                        text="MarginPct",
                        labels={"MarginPct": "Avg Margin %", "Category": ""})
        fig_mg.update_traces(marker_color=mg_colors,
                             texttemplate="%{text:.1f}%", textposition="outside")
        fig_mg.add_hline(y=0, line_dash="dash", line_color="#E53935", line_width=1.5)
        fig_mg.update_layout(yaxis_title="Margin %")
        render_chart(fig_mg, 300, key="ov_margin")

    # ── Negative margin alert ──────────────────────────────────────────────
    _neg_cats = mg[mg["MarginPct"] < 0]["Category"].tolist()
    if _neg_cats:
        _neg_list = ", ".join(f"<b>{c}</b>" for c in _neg_cats)
        st.markdown(
            f'<div style="background:rgba(255,235,238,0.95);border:1.5px solid rgba(183,28,28,0.45);'
            f'border-left:5px solid #B71C1C;border-radius:10px;padding:11px 16px;margin-top:6px;">'
            f'<span style="font-size:0.78rem;font-weight:900;color:#B71C1C;letter-spacing:1px;'
            f'text-transform:uppercase">⚠️ Negative Margin Alert</span><br>'
            f'<span style="font-size:0.86rem;color:#5a1010;font-weight:600">'
            f'{_neg_list} {"is" if len(_neg_cats)==1 else "are"} being sold <u>below cost</u>. '
            f'Every unit sold loses money. Raise retail price or delist immediately.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c6:
        sec("👥 Hospital vs Retail Revenue Split")
        ct_rev = get_customer_type_revenue(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
        fig_ct = px.pie(ct_rev, names="CustomerType", values="TotalAmount", hole=0.52,
                        color="CustomerType",
                        color_discrete_map={"Hospital": "#1565C0", "Retail": "#00ACC1"})
        fig_ct.update_traces(textinfo="percent+label+value",
                             texttemplate="%{label}<br>%{percent:.0%}<br>₹%{value:,.0f}",
                             pull=[0.04, 0.04])
        render_chart(fig_ct, 300, key="ov_customer_split")

    # ── Hospital dependency warning ────────────────────────────────────────────
    _ov_hosp_pct = 0.0
    if len(ct_rev) > 0:
        _ov_total_ct = ct_rev["TotalAmount"].sum()
        _ov_hosp_amt = float(ct_rev[ct_rev["CustomerType"] == "Hospital"]["TotalAmount"].sum()) if "Hospital" in ct_rev["CustomerType"].values else 0.0
        _ov_hosp_pct = round(_ov_hosp_amt / max(_ov_total_ct, 1) * 100, 1)
    if _ov_hosp_pct > 60:
        st.markdown(
            f'<div style="background:rgba(255,243,224,0.95);border:1.5px solid rgba(230,81,0,0.45);'
            f'border-left:5px solid #E65100;border-radius:10px;padding:11px 16px;margin-top:6px;">'
            f'<span style="font-size:0.78rem;font-weight:900;color:#BF360C;letter-spacing:1px;'
            f'text-transform:uppercase">⚠️ Hospital Dependency Risk</span><br>'
            f'<span style="font-size:0.86rem;color:#4a1a00;font-weight:600">'
            f'<b>{_ov_hosp_pct}% of revenue</b> comes from hospital accounts. '
            f'If even one contract lapses, cash flow will be severely impacted. '
            f'Launch a retail loyalty programme to rebalance.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Net Profit vs COGS by Store ────────────────────────────────────────────
    sec("💰 Net Profit vs COGS by Store")
    _ov_store_pnl = (
        _ov_fs.groupby("Store_Name", observed=True)
        .agg(Revenue=("TotalAmount","sum"), COGS=("UnitCost", lambda x: (x * _ov_fs.loc[x.index,"QuantitySold"]).sum()))
        .reset_index()
    )
    # Re-compute properly: COGS = sum(QuantitySold * UnitCost) per store
    _pnl_cogs = (_ov_fs.assign(COGS_line=_ov_fs["QuantitySold"]*_ov_fs["UnitCost"])
                       .groupby("Store_Name", observed=True)
                       .agg(Revenue=("TotalAmount","sum"), COGS=("COGS_line","sum"))
                       .reset_index())
    _pnl_cogs["Net Profit"] = _pnl_cogs["Revenue"] - _pnl_cogs["COGS"]
    _pnl_fig = go.Figure()
    _pnl_fig.add_trace(go.Bar(x=_pnl_cogs["Store_Name"], y=_pnl_cogs["Revenue"],
                               name="Revenue", marker_color="#1976D2",
                               text=_pnl_cogs["Revenue"].map("₹{:,.0f}".format),
                               textposition="inside"))
    _pnl_fig.add_trace(go.Bar(x=_pnl_cogs["Store_Name"], y=_pnl_cogs["COGS"],
                               name="COGS", marker_color="#90CAF9",
                               text=_pnl_cogs["COGS"].map("₹{:,.0f}".format),
                               textposition="inside"))
    _pnl_fig.add_trace(go.Bar(x=_pnl_cogs["Store_Name"], y=_pnl_cogs["Net Profit"],
                               name="Net Profit", marker_color="#43A047",
                               text=_pnl_cogs["Net Profit"].map("₹{:,.0f}".format),
                               textposition="outside"))
    _pnl_fig.update_layout(barmode="group", xaxis_title="", yaxis_title="Amount (₹)")
    render_chart(_pnl_fig, 320, key="ov_net_profit")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INVENTORY & EXPIRY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧪  Inventory & Expiry":
    st.markdown('<div class="page-title">🧪 Inventory & Expiry Monitoring</div>',
                unsafe_allow_html=True)

    fk  = get_filtered_stock(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
    inv = get_inventory_expiry_kpis(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)

    _inv_ss = get_safety_stock_kpis(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
    st.markdown(f"""<div class="kpi-row">
        {kpi(f"{inv['expired']:,}",         "Expired Medicines",    "warn")}
        {kpi(f"₹{fmt(inv['wastage'])}",     "Wastage Value",        "warn")}
        {kpi(f"{inv['safe_cnt']:,}",        "Safe Stock Items",     "ok")}
        {kpi(f"{inv['near_exp']:,}",        "Near Expiry (≤30d)",   "")}
        {kpi(f"{_inv_ss['below_safety']:,}","Below Safety Stock",   "warn")}
        {kpi(f"{_inv_ss['out_of_stock']:,}","Out of Stock Items",   "warn")}
        {kpi(f"{inv['low_s']:,}",           "Below Reorder Point",  "warn")}
    </div>""", unsafe_allow_html=True)

    BCOLORS = {"Valid": "#1976D2", "Expired": "#E53935", "Expiring Soon": "#FB8C00"}

    c1, c2 = st.columns([1.4, 1])

    with c1:
        sec("📦 Inventory Health by Category & Status")
        cb   = fk.groupby(["Category", "ExpiryStatus"], observed=True)["QuantityOnHand"].sum().reset_index()
        fig1 = px.bar(cb, x="QuantityOnHand", y="Category", color="ExpiryStatus",
                      orientation="h", color_discrete_map=BCOLORS,
                      labels={"QuantityOnHand": "Qty On Hand", "Category": ""})
        render_chart(fig1, 295, key="inv_health_cat")

    with c2:
        sec("🍩 Expiry Status Distribution")
        bc        = fk["ExpiryStatus"].value_counts().reset_index()
        bc.columns = ["Status", "Count"]
        fig3 = px.pie(bc, names="Status", values="Count", hole=0.52,
                      color="Status", color_discrete_map=BCOLORS)
        fig3.update_traces(pull=[0.05] * len(bc),
                           textinfo="percent+label+value", textfont_size=11)
        render_chart(fig3, 275, key="inv_expiry_pie")

    # Full-width low-stock chart — fills gap left by unequal column heights
    sec("⚠️ Most Critical Low-Stock Products")
    low_df = (fk[fk["QuantityOnHand"] < fk["ReorderPoint"]]
              .groupby("ProductName", observed=True)["QuantityOnHand"]
              .sum().sort_values().head(8).reset_index())
    fig2   = px.bar(low_df, x="QuantityOnHand", y="ProductName", orientation="h",
                    color="QuantityOnHand", color_continuous_scale="Reds_r",
                    text="QuantityOnHand",
                    labels={"QuantityOnHand": "Qty", "ProductName": ""})
    fig2.update_traces(textposition="outside")
    fig2.update_layout(coloraxis_showscale=False)
    render_chart(fig2, 310, key="inv_low_stock")

    sec("📋 Batch Expiry Detail")
    sample = get_batch_expiry_table(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)

    # Status colour maps — applied element-wise (not full-table applymap)
    _EXPIRY_COLORS = {
        "Expired":       "background-color:#FFEBEE;color:#C62828;font-weight:700",
        "Expiring Soon": "background-color:#FFF3E0;color:#E65100;font-weight:700",
        "Valid":         "background-color:#E8F5E9;color:#2E7D32;font-weight:700",
    }
    _STOCK_COLORS = {
        "Out of Stock": "background-color:#FFEBEE;color:#C62828;font-weight:700",
        "Low Stock":    "background-color:#FFF3E0;color:#E65100;font-weight:700",
        "In Stock":     "background-color:#E8F5E9;color:#2E7D32;font-weight:700",
    }

    styled = (
        sample.style
        .map(lambda v: _EXPIRY_COLORS.get(v, ""), subset=["ExpiryStatus"])
        .map(lambda v: _STOCK_COLORS.get(v, ""),  subset=["StockStatus"])
    )
    st.dataframe(styled, use_container_width=True, height=480)

    st.markdown(f"""<div class="insight-box"><h4>📌 Inventory Insights</h4><ul>
        <li><b>{inv['expired']:,} medicines expired</b> — immediate disposal & write-off required.</li>
        <li><b>{inv['near_exp']:,} near-expiry items</b> — accelerate via promotions or inter-store transfer.</li>
        <li>Wastage value: <b>₹{inv['wastage']:,.0f}</b> — reduce via demand-aligned procurement.</li>
        <li><b>{inv['low_s']:,} SKUs below reorder point</b> — trigger purchase orders immediately.</li>
        <li>Analgesics &amp; Antibiotics hold the largest on-hand inventory volumes.</li>
    </ul></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SALES & DEMAND
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Sales & Demand":
    st.markdown('<div class="page-title">📈 Sales & Demand Analysis</div>',
                unsafe_allow_html=True)

    # Fetch once — reuse across all charts on this page
    fs = get_filtered_sales(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
    fk = get_filtered_stock(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)

    # KPIs — vectorised, no repeated aggregation
    total_rev          = float(fs["TotalAmount"].sum())
    total_sv           = float(np.dot(fk["QuantityOnHand"].to_numpy(),
                                       fk["UnitCost"].to_numpy()))
    total_units        = float(fs["QuantitySold"].sum())
    avg_orders_per_day = round(
        float(fs.groupby("Date", observed=True)["TransactionID"].count().mean()), 2)

    st.markdown(f"""<div class="kpi-row">
        {kpi(f"₹{fmt(total_rev)}", "Total Revenue")}
        {kpi(fmt(total_sv),        "Total Stock Value")}
        {kpi(fmt(total_units),     "Total Units Sold")}
        {kpi(f"{avg_orders_per_day}", "Avg Orders Per Day", "teal")}
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    
    with c1:
        sec("⚡ Fast vs Slow Moving by Category")

        # Classify each category as Fast or Slow based on units/day vs median
        _fs_for_cat = get_filtered_sales(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
        _days = max((_fs_for_cat["Date"].max() - _fs_for_cat["Date"].min()).days, 1)

        # Total units sold per category → units per day
        _cat_units = (
            _fs_for_cat.groupby("Category", observed=True)["QuantitySold"]
            .sum().reset_index()
        )
        _cat_units["UnitsPerDay"] = (_cat_units["QuantitySold"] / _days).round(2)
        _cat_median = _cat_units["UnitsPerDay"].median()
        _cat_units["Segment"] = _cat_units["UnitsPerDay"].apply(
            lambda v: "Fast Moving" if v >= _cat_median else "Slow Moving"
        )
        _cat_units["QuantitySold"] = _cat_units["QuantitySold"].astype(int)

        # Separate fast and slow for two differently-coloured traces
        _fast_cats = _cat_units[_cat_units["Segment"] == "Fast Moving"]
        _slow_cats = _cat_units[_cat_units["Segment"] == "Slow Moving"]

        fig1 = go.Figure()

        fig1.add_trace(go.Bar(
            x=_fast_cats["Category"],
            y=_fast_cats["QuantitySold"],
            name="Fast Moving",
            marker_color="#1565C0",
            text=_fast_cats["QuantitySold"],
            texttemplate="%{text:,.0f}",
            textposition="inside",
            textangle=90,
            textfont=dict(color="white", size=11),
            insidetextanchor="middle",
        ))

        fig1.add_trace(go.Bar(
            x=_slow_cats["Category"],
            y=_slow_cats["QuantitySold"],
            name="Slow Moving",
            marker_color="#90CAF9",
            text=_slow_cats["QuantitySold"],
            texttemplate="%{text:,.0f}",
            textposition="inside",
            textangle=90,
            textfont=dict(color="white", size=11),
            insidetextanchor="middle",
        ))

        # Annotation: label each bar top with ⚡ or 🐢
        _annotations = []
        for _, _row in _cat_units.iterrows():
            _annotations.append(dict(
                x=str(_row["Category"]),
                y=_row["QuantitySold"] * 1.04,
                text="⚡" if _row["Segment"] == "Fast Moving" else "🐢",
                showarrow=False,
                font=dict(size=14),
                xanchor="center",
            ))

        fig1.update_layout(
            barmode="group",
            bargap=0.28,
            yaxis_title="Units Sold",
            xaxis_title="",
            height=360,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(tickangle=-30, categoryorder="total descending"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(21,101,192,0.10)"),
            annotations=_annotations,
            legend=dict(
                orientation="v",
                x=0.78, y=0.98,
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#ccc",
                borderwidth=1,
                font=dict(size=11),
            ),
        )
        st.plotly_chart(
            fig1,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        sec("🏃 Top 10 Fastest Moving Medicines")

        # Recreate velocity dataframe
        vel_df = get_medicine_velocity(
            DR_MIN,
            DR_MAX,
            SEL_STORE,
            SEL_CAT
        )
        top_fast = vel_df.sort_values("UnitsPerDay", ascending=False).head(10)
        fig1b = px.bar(top_fast, x="UnitsPerDay", y="Medicine_Name", orientation="h",
                           text="UnitsPerDay",
                           color="UnitsPerDay", color_continuous_scale="Blues",
                           labels={"UnitsPerDay": "Units/Day", "Medicine_Name": ""})
        fig1b.update_traces(texttemplate="%{text:.2f}/day", textposition="outside")
        fig1b.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
        render_chart(fig1b, 310, key="sd_top_fast")

        sec("🍩 Revenue by Category")
        cr   = get_category_revenue(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
        fig2 = px.pie(cr, names="Category", values="TotalAmount", hole=0.52,
                      color="Category", color_discrete_map=CAT_PALETTE)
        fig2.update_traces(textinfo="percent+label", pull=[0.03] * len(cr))
        render_chart(fig2, 300, key="sd_cat_pie")

    with c2:
        sec("📅 Monthly Demand Trend")
        mq   = (fs.groupby(["Month_Num", "Month"], observed=True)["QuantitySold"]
                  .sum().reset_index().sort_values("Month_Num"))
        fig3 = px.line(mq, x="Month", y="QuantitySold", markers=True,
                       color_discrete_sequence=["#1976D2"],
                       labels={"QuantitySold": "Units Sold", "Month": ""})
        fig3.update_traces(line_width=2.8,
                           marker=dict(size=8, color="#1565C0",
                                       line=dict(width=2, color="#fff")))
        render_chart(fig3, 285, key="sd_monthly_demand")

        sec("🗂️ Units Sold by Category")
        cu   = (fs.groupby("Category", observed=True)["QuantitySold"]
                      .sum().reset_index().sort_values("QuantitySold"))
        fig4 = px.bar(cu, x="QuantitySold", y="Category", orientation="h",
                      color="Category", color_discrete_map=CAT_PALETTE,
                      text="QuantitySold",
                      labels={"QuantitySold": "Units Sold", "Category": ""})
        fig4.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig4.update_layout(showlegend=False)
        render_chart(fig4, 295, key="sd_cat_units")

        sec("💊 Top 15 Medicines — Revenue & Volume")
        st.dataframe(get_top15_medicines(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT),
                     use_container_width=True, height=340)

    # ── Full-width: Hospital vs Retail — fills gap from unequal column heights ──
    sec("🏥 Hospital (Rx-proxy) vs Retail (OTC-proxy) Split")
    _sd_ct = fs.groupby(["Category", "CustomerType"], observed=True)["TotalAmount"].sum().reset_index()
    _hosp_total = float(_sd_ct[_sd_ct["CustomerType"] == "Hospital"]["TotalAmount"].sum())
    _retl_total = float(_sd_ct[_sd_ct["CustomerType"] == "Retail"]["TotalAmount"].sum())
    _grand_total = _hosp_total + _retl_total
    _sd_ct2 = fs.groupby("CustomerType", observed=True).agg(
        Revenue=("TotalAmount","sum"), Units=("QuantitySold","sum")).reset_index()
    _sd_ct2["Avg Price/Unit"] = (_sd_ct2["Revenue"] / _sd_ct2["Units"].replace(0,1)).round(2)

    _rx_c1, _rx_c2 = st.columns([1, 1])
    with _rx_c1:
        st.markdown(
            f'<div class="insight-box"><h4>🔍 Customer Channel Analysis</h4><ul>'
            f'<li>Hospital channel (prescription-driven): <b>₹{_hosp_total:,.0f}</b> '
            f'({_hosp_total/_grand_total*100:.1f}% of revenue)</li>'
            f'<li>Retail / OTC channel: <b>₹{_retl_total:,.0f}</b> '
            f'({_retl_total/_grand_total*100:.1f}% of revenue)</li>'
            f'<li><b>Note:</b> Hospital orders are typically prescription (Rx) medicines — '
            f'subject to Schedule H/H1 regulations. Retail includes both OTC and prescription '
            f'sales. Recommend tagging Rx vs OTC in product master for precise compliance tracking.</li>'
            f'</ul></div>',
            unsafe_allow_html=True,
        )
    with _rx_c2:
        st.dataframe(_sd_ct2.style.format({"Revenue": "₹{:,.0f}", "Units": "{:,.0f}", "Avg Price/Unit": "₹{:.2f}"}),
                     use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SUPPLIER & STORE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏪  Supplier & Store":
    st.markdown('<div class="page-title">🏪 Supplier & Store Performance</div>',
                unsafe_allow_html=True)

    fk = get_filtered_stock(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
    fs = get_filtered_sales(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)

    # KPIs — computed once, reused below
    n_sup    = int(fk["Supplier_Name"].nunique())
    n_stores = int(store_df.shape[0])
    avg_lead = round(float(fk["LeadTimeDays"].mean()), 1)
    avg_rat  = round(float(fk["QualityRating"].mean()), 2)

    st.markdown(f"""<div class="kpi-row">
        {kpi(f"{n_sup}",    "Active Suppliers")}
        {kpi(f"{n_stores}", "Total Stores")}
        {kpi(f"{avg_lead}", "Avg Lead Time (Days)", "teal")}
        {kpi(f"{avg_rat}",  "Avg Quality Rating",   "ok")}
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        sec("🏭 Supplier Stock Contribution")
        sc   = get_supplier_stock(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
        fig2 = px.bar(sc, x="QuantityOnHand", y="Supplier_Name", orientation="h",
                      color="QuantityOnHand", color_continuous_scale="Blues",
                      text="QuantityOnHand",
                      labels={"QuantityOnHand": "Stock On Hand", "Supplier_Name": ""})
        fig2.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig2.update_layout(coloraxis_showscale=False)
        render_chart(fig2, 320, key="ss_sup_stock")

    with c2:
        sec("⭐ Top Supplier Quality Ratings")
        rat = (fk.groupby("Supplier_Name", observed=True)["QualityRating"]
                 .mean().sort_values(ascending=False).head(8).reset_index())
        rat.columns = ["Supplier", "Rating"]
        fig3 = px.bar(rat, x="Rating", y="Supplier", orientation="h",
                      color="Rating", color_continuous_scale="Greens",
                      text="Rating",
                      labels={"Rating": "Quality Rating", "Supplier": ""})
        fig3.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig3.update_layout(coloraxis_showscale=False, xaxis_range=[0, 5.2])
        render_chart(fig3, 310, key="ss_qual_rat")

    # ── STORE PERFORMANCE OVERVIEW — per-store cards ──────────────────────────
    sec("🏪 Store Performance Overview")

    # Revenue per store
    _ss_rev = (fs.groupby("Store_Name", observed=True)["TotalAmount"]
                  .sum().reset_index().rename(columns={"TotalAmount": "Revenue"}))
    _ss_units = (fs.groupby("Store_Name", observed=True)["QuantitySold"]
                    .sum().reset_index())
    _ss_txn = (fs.groupby("Store_Name", observed=True)["TransactionID"]
                  .count().reset_index().rename(columns={"TransactionID": "Transactions"}))
    # ── All stock metrics use BATCH-ROW counts so every number tallies ──────────
    # Total Stock  = every batch row for that store (the universe)
    # OOS          = batches where QuantityOnHand == 0
    # Expired      = batches where ExpiryStatus == "Expired"
    # Near Expiry  = batches where DaysToExpiry in [0,30] AND not already Expired
    # Low Stock    = batches where 0 < QuantityOnHand < ReorderPoint AND not Expired
    # Current Stock = batches where QuantityOnHand > 0 AND not Expired
    # Relationship : TotalStock = OOS + Expired + CurrentStock
    #                CurrentStock ⊇ LowStock ⊇ (some) NearExpiry rows
    _grp = "Store_Name"
    def _cnt(mask):
        return (fk[mask].groupby(_grp, observed=True)
                .size().reset_index(name="_n"))

    _ss_total_stock = _cnt(pd.Series([True] * len(fk), index=fk.index)).rename(columns={"_n": "TotalStock"})
    _ss_total_units = (fk.groupby(_grp, observed=True)["QuantityOnHand"]
                         .sum().reset_index().rename(columns={"QuantityOnHand": "TotalUnits"}))
    _ss_oos         = _cnt(fk["QuantityOnHand"] == 0).rename(columns={"_n": "OOS_Items"})
    _ss_exp         = _cnt(fk["ExpiryStatus"] == "Expired").rename(columns={"_n": "ExpiredQty"})
    _ss_near        = _cnt(
        (fk["DaysToExpiry"].between(0, 30)) & (fk["ExpiryStatus"] != "Expired")
    ).rename(columns={"_n": "NearExpiry"})
    _ss_low         = _cnt(
        (fk["QuantityOnHand"] > 0) &
        (fk["QuantityOnHand"] < fk["ReorderPoint"]) &
        (fk["ExpiryStatus"] != "Expired")
    ).rename(columns={"_n": "LowStock"})
    _ss_current     = _cnt(
        (fk["QuantityOnHand"] > 0) & (fk["ExpiryStatus"] != "Expired")
    ).rename(columns={"_n": "CurrentStock"})
    _ss_sv = (fk.copy()
                 .assign(StockVal=lambda d: d["QuantityOnHand"] * d["UnitCost"])
                 .groupby("Store_Name", observed=True)["StockVal"]
                 .sum().reset_index().rename(columns={"StockVal": "StockValue"}))

    _ss_merged = _ss_rev.merge(_ss_units,        on="Store_Name", how="left")
    _ss_merged = _ss_merged.merge(_ss_txn,        on="Store_Name", how="left")
    _ss_merged = _ss_merged.merge(_ss_exp,        on="Store_Name", how="left")
    _ss_merged = _ss_merged.merge(_ss_near,       on="Store_Name", how="left")
    _ss_merged = _ss_merged.merge(_ss_oos,        on="Store_Name", how="left")
    _ss_merged = _ss_merged.merge(_ss_low,        on="Store_Name", how="left")
    _ss_merged = _ss_merged.merge(_ss_sv,         on="Store_Name", how="left")
    _ss_merged = _ss_merged.merge(_ss_total_stock,on="Store_Name", how="left")
    _ss_merged = _ss_merged.merge(_ss_total_units,on="Store_Name", how="left")
    _ss_merged = _ss_merged.merge(_ss_current,    on="Store_Name", how="left")
    for _col in ["ExpiredQty", "NearExpiry", "OOS_Items", "LowStock",
                 "Transactions", "QuantitySold", "TotalStock", "TotalUnits", "CurrentStock"]:
        _ss_merged[_col] = _ss_merged[_col].fillna(0).astype(int)
    _ss_merged["StockValue"] = _ss_merged["StockValue"].fillna(0)
    _ss_merged = _ss_merged.sort_values("Revenue", ascending=False).reset_index(drop=True)

    _ss_rev_max    = _ss_merged["Revenue"].max() if len(_ss_merged) > 0 else 1
    _ss_rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    _ss_store_icons = {"MedPlus": "💊", "Apollo": "🏥", "WellCare": "🩺",
                       "HealthFirst": "💉", "CityMed": "🧬"}

    _ss_store_rows = [_ss_merged.iloc[i:i+2] for i in range(0, len(_ss_merged), 2)]
    for _ss_row_df in _ss_store_rows:
        _ss_cols = st.columns(len(_ss_row_df))
        for _ss_ci, (_, _ss_sr) in enumerate(zip(_ss_cols, _ss_row_df.itertuples())):
            _ss_rank    = _ss_rank_emojis[_ss_sr.Index] if _ss_sr.Index < len(_ss_rank_emojis) else "🏪"
            _ss_icon    = next((v for k, v in _ss_store_icons.items() if k in _ss_sr.Store_Name), "🏪")
            _ss_bar_pct = int(_ss_sr.Revenue / _ss_rev_max * 100)
            if _ss_sr.OOS_Items == 0 and _ss_sr.ExpiredQty == 0:
                _ss_hcol = "#1B5E20"; _ss_hlbl = "✅ Healthy"
            elif _ss_sr.OOS_Items > 5 or _ss_sr.ExpiredQty > 20:
                _ss_hcol = "#B71C1C"; _ss_hlbl = "🔴 Needs Attention"
            else:
                _ss_hcol = "#E65100"; _ss_hlbl = "🟠 Monitor Closely"

            _ss_cols[_ss_ci].markdown(f"""
<div style="background:linear-gradient(145deg,rgba(255,255,255,0.88),rgba(235,245,255,0.82));
            border:1px solid rgba(200,220,240,0.70);
            border-top:4px solid {_ss_hcol};
            border-radius:16px;padding:18px 20px;margin-bottom:12px;
            box-shadow:0 4px 20px rgba(21,101,192,0.10)">

  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
    <div style="display:flex;align-items:center;gap:10px">
      <span style="font-size:1.8rem">{_ss_icon}</span>
      <div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:1.05rem;font-weight:700;
                    color:#0b3d6e;letter-spacing:0.4px">{_ss_sr.Store_Name}</div>
        <div style="font-size:0.72rem;font-weight:700;color:{_ss_hcol};
                    letter-spacing:0.5px;margin-top:1px">{_ss_hlbl}</div>
      </div>
    </div>
    <span style="font-size:1.6rem">{_ss_rank}</span>
  </div>

  <div style="margin-bottom:14px">
    <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">
      <span style="font-size:0.72rem;font-weight:800;color:#4a6a8a;letter-spacing:0.8px;text-transform:uppercase">Revenue</span>
      <span style="font-family:'Rajdhani',sans-serif;font-size:1.3rem;font-weight:700;color:#0d5c1e">₹{_ss_sr.Revenue:,.0f}</span>
    </div>
    <div style="height:7px;background:rgba(21,101,192,0.12);border-radius:4px">
      <div style="height:7px;width:{_ss_bar_pct}%;background:linear-gradient(90deg,#1565C0,#42A5F5);border-radius:4px"></div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-bottom:12px">
    <div style="background:rgba(21,101,192,0.07);border-radius:10px;padding:8px 10px;text-align:center">
      <div style="font-size:1.05rem;font-weight:700;color:#0b3d6e">{_ss_sr.QuantitySold:,}</div>
      <div style="font-size:0.66rem;font-weight:800;color:#4a6a8a;text-transform:uppercase;letter-spacing:0.5px">Units Sold</div>
    </div>
    <div style="background:rgba(21,101,192,0.07);border-radius:10px;padding:8px 10px;text-align:center">
      <div style="font-size:1.05rem;font-weight:700;color:#0b3d6e">{_ss_sr.Transactions:,}</div>
      <div style="font-size:0.66rem;font-weight:800;color:#4a6a8a;text-transform:uppercase;letter-spacing:0.5px">Transactions</div>
    </div>
    <div style="background:rgba(21,101,192,0.07);border-radius:10px;padding:8px 10px;text-align:center">
      <div style="font-size:1.05rem;font-weight:700;color:#0b3d6e">₹{fmt(_ss_sr.StockValue)}</div>
      <div style="font-size:0.66rem;font-weight:800;color:#4a6a8a;text-transform:uppercase;letter-spacing:0.5px">Stock Value</div>
    </div>
    <div style="background:rgba(0,188,212,0.09);border-radius:10px;padding:8px 10px;text-align:center">
      <div style="font-size:1.05rem;font-weight:700;color:#006064">{_ss_sr.TotalUnits:,}</div>
      <div style="font-size:0.66rem;font-weight:800;color:#00696e;text-transform:uppercase;letter-spacing:0.5px">Total Stock (Units)</div>
    </div>
  </div>

  <!-- Stock tally bar: Total = OOS + Expired + Current Stock -->
  <div style="background:rgba(21,101,192,0.05);border:1px solid rgba(21,101,192,0.15);
              border-radius:10px;padding:8px 12px;margin-bottom:10px">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
      <span style="font-size:0.65rem;font-weight:800;color:#4a6a8a;text-transform:uppercase;letter-spacing:0.7px">SKU Status Breakdown: Available + Expired + Out of Stock</span>
      <span style="font-size:0.75rem;font-weight:800;color:#0b3d6e">{_ss_sr.TotalStock:,} SKUs = {_ss_sr.CurrentStock:,} + {_ss_sr.ExpiredQty} + {_ss_sr.OOS_Items}</span>
    </div>
    <div style="height:8px;border-radius:4px;overflow:hidden;display:flex;gap:1px">
      <div style="flex:{_ss_sr.OOS_Items};background:#E53935;min-width:{2 if _ss_sr.OOS_Items>0 else 0}px" title="OOS"></div>
      <div style="flex:{_ss_sr.ExpiredQty};background:#7B1FA2;min-width:{2 if _ss_sr.ExpiredQty>0 else 0}px" title="Expired"></div>
      <div style="flex:{_ss_sr.CurrentStock};background:linear-gradient(90deg,#1565C0,#42A5F5);min-width:{2 if _ss_sr.CurrentStock>0 else 0}px" title="Current"></div>
    </div>
    <div style="display:flex;gap:12px;margin-top:4px">
      <span style="font-size:0.63rem;color:#E53935;font-weight:700">● OOS</span>
      <span style="font-size:0.63rem;color:#7B1FA2;font-weight:700">● Expired</span>
      <span style="font-size:0.63rem;color:#1565C0;font-weight:700">● Current Stock</span>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:6px">
    <div style="background:rgba(229,57,53,0.10);border:1px solid rgba(229,57,53,0.30);
                border-radius:8px;padding:6px 8px;text-align:center">
      <div style="font-size:1.0rem;font-weight:800;color:#C62828">{_ss_sr.OOS_Items}</div>
      <div style="font-size:0.62rem;font-weight:700;color:#a33;text-transform:uppercase">OOS</div>
    </div>
    <div style="background:rgba(251,140,0,0.10);border:1px solid rgba(251,140,0,0.30);
                border-radius:8px;padding:6px 8px;text-align:center">
      <div style="font-size:1.0rem;font-weight:800;color:#E65100">{_ss_sr.LowStock}</div>
      <div style="font-size:0.62rem;font-weight:700;color:#bf4000;text-transform:uppercase">Low Stk</div>
    </div>
    <div style="background:rgba(106,27,154,0.10);border:1px solid rgba(106,27,154,0.30);
                border-radius:8px;padding:6px 8px;text-align:center">
      <div style="font-size:1.0rem;font-weight:800;color:#6A1B9A">{_ss_sr.ExpiredQty}</div>
      <div style="font-size:0.62rem;font-weight:700;color:#4a1270;text-transform:uppercase">Expired</div>
    </div>
    <div style="background:rgba(183,28,28,0.08);border:1px solid rgba(183,28,28,0.25);
                border-radius:8px;padding:6px 8px;text-align:center">
      <div style="font-size:1.0rem;font-weight:800;color:#B71C1C">{_ss_sr.NearExpiry}</div>
      <div style="font-size:0.62rem;font-weight:700;color:#900;text-transform:uppercase">Near Exp</div>
    </div>
  </div>
  <div style="margin-top:6px;padding:5px 8px;background:rgba(0,188,212,0.07);
              border-radius:8px;border:1px solid rgba(0,188,212,0.20);text-align:center">
    <span style="font-size:0.68rem;font-weight:800;color:#006064;text-transform:uppercase;letter-spacing:0.5px">
      Current Stock (active batches): </span>
    <span style="font-size:0.85rem;font-weight:800;color:#006064">{_ss_sr.CurrentStock:,}</span>
    <span style="font-size:0.65rem;color:#4a7a7a"> &nbsp;|&nbsp; Low Stk &amp; Near Exp are subsets of Current Stock</span>
  </div>

</div>""", unsafe_allow_html=True)

    sec("🗺️ Store Revenue — Hyderabad Map")
    rev_map  = get_map_data(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
    fig_map  = px.scatter_mapbox(
        rev_map, lat="lat", lon="lon", size="size",
        hover_name="Store_Name",
        hover_data={"TotalAmount": True, "Units": True,
                    "lat": False, "lon": False, "size": False},
        color_discrete_sequence=["#1976D2"], zoom=11, height=400,
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_map, use_container_width=True,
                    config={"displayModeBar": False})

    # ── Dynamic Supplier & Store Insights — all scoped to active filters ───────
    _ins_fs  = get_filtered_sales(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
    _ins_fk  = get_filtered_stock(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
    _ins_sup = get_supplier_last_delivery_days(SEL_STORE, SEL_CAT, SEL_SUP)

    # 1. Top revenue store (from filtered sales)
    _ins_sr        = _ins_fs.groupby("Store_Name", observed=True)["TotalAmount"].sum()
    _ins_top_store = _ins_sr.idxmax() if not _ins_sr.empty else "N/A"
    _ins_top_rev   = f"₹{_ins_sr.max():,.0f}" if not _ins_sr.empty else "N/A"

    # 2. Avg supplier lead time (from filtered supplier data → respects store/cat/sup filters)
    _ins_lead     = _ins_sup["LeadTimeDays"].mean() if not _ins_sup.empty and "LeadTimeDays" in _ins_sup.columns else None
    _ins_lead_str = f"~{_ins_lead:.1f} days" if _ins_lead is not None else "N/A"

    # 3. Store with highest on-hand inventory (from filtered stock)
    _ins_stk     = _ins_fk.groupby("Store_Name", observed=True)["QuantityOnHand"].sum()
    _ins_top_inv = _ins_stk.idxmax() if not _ins_stk.empty else "N/A"
    _ins_inv_qty = f"{_ins_stk.max():,.0f} units" if not _ins_stk.empty else "N/A"

    # 4. Top-rated supplier + count ≥ 4.5 (from filtered supplier data)
    _ins_rat_ser      = _ins_sup.set_index("Supplier_Name")["QualityRating"] if not _ins_sup.empty and "QualityRating" in _ins_sup.columns else pd.Series(dtype=float)
    _ins_top_sup      = _ins_rat_ser.idxmax() if not _ins_rat_ser.empty else "N/A"
    _ins_top_score    = f"{_ins_rat_ser.max():.2f} / 5.0" if not _ins_rat_ser.empty else "N/A"
    _ins_high_cnt     = int((_ins_rat_ser >= 4.5).sum())

    # 5. Slowest supplier by lead time (from filtered supplier data)
    _ins_lt_ser   = _ins_sup.set_index("Supplier_Name")["LeadTimeDays"] if not _ins_sup.empty and "LeadTimeDays" in _ins_sup.columns else pd.Series(dtype=float)
    _ins_slow_sup = _ins_lt_ser.idxmax() if not _ins_lt_ser.empty else "N/A"
    _ins_slow_d   = f"{_ins_lt_ser.max():.1f} days" if not _ins_lt_ser.empty else "N/A"

    st.markdown(f"""<div class="insight-box"><h4>📌 Supplier & Store Insights</h4><ul>
        <li><b>{_ins_top_store}</b> generates the highest revenue at <b>{_ins_top_rev}</b> across filtered stores.</li>
        <li>Average supplier lead time is <b>{_ins_lead_str}</b> — dual-source critical SKUs to reduce risk.</li>
        <li><b>{_ins_top_inv}</b> carries the highest on-hand inventory with <b>{_ins_inv_qty}</b>.</li>
        <li><b>{_ins_high_cnt}</b> supplier(s) exceed <b>4.5 / 5.0</b> — top-rated is <b>{_ins_top_sup}</b> ({_ins_top_score}).</li>
        <li>Supplier <b>{_ins_slow_sup}</b> has the longest avg lead time at <b>{_ins_slow_d}</b> — a primary delay risk.</li>
    </ul></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡  Business Insights":
    st.markdown('<div class="page-title">💡 Business Insights & Recommendations</div>',
                unsafe_allow_html=True)

    # Fetch once — shared across both columns
    fk = get_filtered_stock(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
    fs = get_filtered_sales(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)

    # Vectorised KPI computation
    qty      = fk["QuantityOnHand"].to_numpy()
    cost     = fk["UnitCost"].to_numpy()
    exp_st   = fk["ExpiryStatus"].to_numpy(dtype=str)
    exp_msk  = exp_st == "Expired"

    total_sv  = float(np.dot(qty, cost))
    wastage   = float(np.dot(qty[exp_msk], cost[exp_msk]))
    low_stock = int((qty < fk["ReorderPoint"].to_numpy()).sum())
    safe_pct  = round(float((exp_st == "Valid").sum()) / max(len(fk), 1) * 100, 1)

    st.markdown(f"""<div class="kpi-row">
        {kpi(fmt(total_sv),      "Total Stock Value")}
        {kpi(f"₹{wastage:,.0f}", "Wastage Value",    "warn")}
        {kpi(f"{low_stock:,}",   "Low Stock Items",  "warn")}
        {kpi(f"{safe_pct}%",     "Inventory Health", "ok")}
    </div>""", unsafe_allow_html=True)

    c1, = st.columns([1])

    with c1:
        # Compute live values for dynamic insights
        _bi_fs       = get_filtered_sales(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
        _bi_inv      = get_inventory_expiry_kpis(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
        _bi_top_cat  = (_bi_fs.groupby("Category", observed=True)["TotalAmount"]
                        .sum().idxmax() if len(_bi_fs) > 0 else "N/A")
        _bi_top_med  = (_bi_fs.groupby("Medicine_Name", observed=True)["QuantitySold"]
                        .sum().idxmax() if len(_bi_fs) > 0 else "N/A")
        _bi_ct       = (_bi_fs.groupby("CustomerType", observed=True)["TotalAmount"]
                        .sum().to_dict())
        _bi_hosp_pct = round(_bi_ct.get("Hospital", 0) /
                             max(sum(_bi_ct.values()), 1) * 100, 1)
        _bi_near_exp = _bi_inv["near_exp"]
        _bi_ss       = get_safety_stock_kpis(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
        _bi_risk     = get_supplier_risk(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
        _bi_top_risk_sup = _bi_risk.iloc[0]["Supplier_Name"] if len(_bi_risk) > 0 else "N/A"
        _bi_top_risk_score = _bi_risk.iloc[0]["RiskScore"] if len(_bi_risk) > 0 else 0

        _INSIGHT_BOXES = [
            ("📦 Inventory Insights", [
                f"<b>{_bi_inv['expired']:,} medicines expired</b> — review cold chain & storage protocols.",
                f"<b>{_bi_near_exp:,} medicines near expiry</b> — prioritise in sales & promotions.",
                f"<b>{_bi_ss['below_safety']:,} items below safety stock</b> — replenish before reaching reorder level.",
                f"<b>{_bi_ss['out_of_stock']:,} items fully out of stock</b> — raise purchase orders immediately.",
            ]),
            ("⚠️ Risk Indicators", [
                f"Expired stock wastage = <b>₹{wastage:,.0f}</b> — demand-aligned procurement can reduce this.",
                f"<b>{low_stock:,} SKUs</b> below reorder point — shortage risk is elevated.",
                f"Hospital channel drives <b>{_bi_hosp_pct}% of revenue</b> — high concentration risk if hospital contracts change.",
            ]),
            ("📊 Sales Insights", [
                f"<b>{_bi_top_cat}</b> is the top revenue-generating category under current filters.",
                f"<b>{_bi_top_med}</b> is the best-selling medicine by units sold.",
                f"Retail vs Hospital revenue: <b>₹{_bi_ct.get('Retail',0):,.0f}</b> retail | <b>₹{_bi_ct.get('Hospital',0):,.0f}</b> hospital.",
            ]),
            ("🏭 Supplier Insights", [
                f"<b>{_bi_top_risk_sup}</b> has the highest risk score ({_bi_top_risk_score:.1f}) — review contract urgently.",
                "Dual-source any SKU where a single supplier holds &gt;60% of stock.",
                "<b>Supplier lead times</b> directly drive expiry risk — prioritise suppliers with lead time &lt;7 days.",
            ]),
        ]
        for title, items in _INSIGHT_BOXES:
            li_html = "".join(f"<li>{i}</li>" for i in items)
            st.markdown(
                f'<div class="insight-box"><h4>{title}</h4><ul>{li_html}</ul></div>',
                unsafe_allow_html=True,
            )

        st.markdown("""<div class="insight-box"
            style="border-left:5px solid #1976D2;background:rgba(227,242,253,0.88)">
        <h4>✅ RECOMMENDATIONS</h4><ul>
            <li>Implement <b>automated 30/60-day expiry alerts</b>.</li>
            <li>Deploy <b>ML demand forecasting</b> for top-20 SKUs per store.</li>
            <li>Reduce supplier lead time via <b>dual-sourcing agreements</b>.</li>
            <li>Enable <b>inter-store stock transfers</b> for near-expiry surplus.</li>
            <li>Use <b>dynamic pricing</b> to clear slow-moving inventory.</li>
            <li>Set <b>per-store × per-SKU</b> automated reorder triggers.</li>
        </ul></div>""", unsafe_allow_html=True)

        # ── Store underperformance alert ──────────────────────────────────────
        _bi_store_rev = get_store_revenue(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
        if len(_bi_store_rev) >= 2:
            _bi_top_rev  = float(_bi_store_rev["TotalAmount"].max())
            _bi_low_rev  = float(_bi_store_rev["TotalAmount"].min())
            _bi_low_store= _bi_store_rev.loc[_bi_store_rev["TotalAmount"].idxmin(), "Store_Name"]
            _bi_gap      = _bi_top_rev - _bi_low_rev
            _bi_gap_pct  = round(_bi_gap / max(_bi_top_rev, 1) * 100, 1)
            if _bi_gap_pct > 30:
                st.markdown(
                    f'<div class="insight-box" style="border-left:5px solid #E53935;'
                    f'background:rgba(255,235,238,0.88)">'
                    f'<h4>🏪 Store Underperformance Alert</h4><ul>'
                    f'<li><b>{_bi_low_store}</b> generates <b>₹{_bi_low_rev:,.0f}</b> — '
                    f'<b>{_bi_gap_pct}% below</b> the top-performing store '
                    f'(gap = ₹{_bi_gap:,.0f}).</li>'
                    f'<li>Assign a store manager review: check footfall, local competition, '
                    f'pricing strategy, and product mix.</li>'
                    f'<li>Consider running store-specific promotions or expanding the '
                    f'retail (non-hospital) customer base.</li>'
                    f'</ul></div>',
                    unsafe_allow_html=True,
                )

        # ── Negative margin category alert ────────────────────────────────────
        _bi_margin = get_margin_by_category(DR_MIN, DR_MAX, SEL_STORE, SEL_CAT)
        _bi_neg    = _bi_margin[_bi_margin["MarginPct"] < 0]
        if len(_bi_neg) > 0:
            _bi_neg_html = "".join(
                f"<li><b>{row['Category']}</b>: avg margin = <b style='color:#C62828'>"
                f"{row['MarginPct']:.1f}%</b> — selling below cost. "
                f"Raise retail price or stop procurement until margin turns positive.</li>"
                for _, row in _bi_neg.iterrows()
            )
            st.markdown(
                f'<div class="insight-box" style="border-left:5px solid #B71C1C;'
                f'background:rgba(255,235,238,0.88)">'
                f'<h4>⚠️ Below-Cost Categories</h4><ul>{_bi_neg_html}</ul></div>',
                unsafe_allow_html=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 🤖 AI PHARMACY ASSISTANT
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="ai-section-title">🤖 AI Pharmacy Assistant</div>',
                unsafe_allow_html=True)

    # Groq availability indicator
    if _groq_client is not None:
        st.markdown(
            '<span class="ai-status-pill">⚡ Groq · Llama-3.3-70B · Live</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="ai-status-pill" style="color:#E53935;border-color:#E53935;'
            'background:rgba(229,57,53,0.08)">⚠️ Groq not configured</span>',
            unsafe_allow_html=True,
        )
        st.info(
            "**To enable the AI assistant:**\n"
            "1. `pip install groq python-dotenv`\n"
            "2. Create `.env` → add `GROQ_API_KEY=your_key_here`\n"
            "3. Restart the app\n\n"
            "Get a free API key at [console.groq.com](https://console.groq.com)",
            icon="🔑",
        )

    # Build shared context (always, so it's ready for both the button and chat)
    exp_kpis_ai = get_inventory_expiry_kpis(SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS)
    _ai_context = _build_ai_context(
        DR_MIN, DR_MAX, SEL_STORE, SEL_CAT, SEL_SUP, SEL_STS,
        total_rev=float(fs["TotalAmount"].sum()),
        total_sv=total_sv,
        low_stock=low_stock,
        safe_pct=safe_pct,
        wastage=wastage,
        exp_kpis=exp_kpis_ai,
    )

    # ── Button: Generate full AI insights ────────────────────────────────────
    ai_col1, ai_col2 = st.columns([1, 3])
    with ai_col1:
        gen_btn = st.button(
            "🤖 Generate AI Insights",
            key="btn_gen_ai",
            use_container_width=True,
            type="primary",
        )

    if gen_btn:
        st.session_state["ai_insights_result"] = None   # clear stale cache display
        with st.spinner("🧠 Llama is analysing your pharmacy data…"):
            result = generate_ai_insights(_ai_context, user_question=None)
        st.session_state["ai_insights_result"] = result

    # Display persisted AI insights result as Q&A cards
    if st.session_state.get("ai_insights_result"):
        raw = st.session_state["ai_insights_result"]

        # Parse Q&A pairs — supports "Q:" / "A:" format returned by the model
        import re as _re
        pairs = _re.findall(
            r"Q:\s*(.+?)\nA:\s*([\s\S]+?)(?=\nQ:|\Z)",
            raw,
            _re.IGNORECASE,
        )

        if pairs:
            cards_html = '<div class="qa-grid">'
            for q, a in pairs:
                answer_text = a.strip()
                prob_match = _re.search(
                    r'(?:\U0001f534\s*)?PROBLEM[:\s]+([\s\S]+?)(?=(?:\u2705\s*)?SOLUTION|$)',
                    answer_text, _re.IGNORECASE
                )
                sol_match = _re.search(
                    r'(?:\u2705\s*)?SOLUTION[:\s]+([\s\S]+?)$',
                    answer_text, _re.IGNORECASE
                )
                if prob_match and sol_match:
                    prob_text = prob_match.group(1).strip()
                    sol_text  = sol_match.group(1).strip()
                    _bullet_re = r'^[•\-\*]\s*'
                    sol_lines = [ln.strip() for ln in sol_text.splitlines() if ln.strip()]
                    sol_items = "".join(
                        "<li>" + _re.sub(_bullet_re, "", ln) + "</li>"
                        for ln in sol_lines
                    )
                    answer_block = (
                        f'<div class="qa-problem">'
                        f'<span class="qa-problem-label">\U0001f534 Problem</span>'
                        f'<p class="qa-problem-text">{prob_text}</p>'
                        f'</div>'
                        f'<div class="qa-solution">'
                        f'<span class="qa-solution-label">\u2705 Solution</span>'
                        f'<ul class="qa-solution-list">{sol_items}</ul>'
                        f'</div>'
                    )
                else:
                    answer_lines = [ln.strip() for ln in answer_text.splitlines() if ln.strip()]
                    _bullet_re = r'^[•\-\*]\s*'
                    li_items = "".join(
                        "<li>" + _re.sub(_bullet_re, "", ln) + "</li>"
                        for ln in answer_lines
                    )
                    answer_block = f'<ul>{li_items}</ul>'
                cards_html += (
                    f'<div class="qa-card">'
                    f'<div class="qa-question">{q.strip()}</div>'
                    f'<div class="qa-answer">{answer_block}</div>'
                    f'</div>'
                )
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)
        else:
            # Fallback: plain card if parsing fails
            st.markdown(
                f'<div class="ai-response-card">'
                f'{raw.replace(chr(10), "<br>")}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Chat Assistant ────────────────────────────────────────────────────────
    # ── helper: convert basic markdown → HTML ───────────────────────────────
    import re as _re
    def _md_to_html(text: str) -> str:
        """Convert **bold**, *italic*, bullet lines and newlines to HTML."""
        # escape angle brackets first so raw text is safe
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # headings  ### ## #
        text = _re.sub(r"^### (.+)$",
            r'<h4 style="margin:14px 0 4px;color:#0b3d6e;font-size:0.95rem">\1</h4>',
            text, flags=_re.MULTILINE)
        text = _re.sub(r"^## (.+)$",
            r'<h3 style="margin:16px 0 6px;color:#0b3d6e;font-size:1.05rem">\1</h3>',
            text, flags=_re.MULTILINE)
        text = _re.sub(r"^# (.+)$",
            r'<h2 style="margin:16px 0 6px;color:#0b3d6e;font-size:1.15rem">\1</h2>',
            text, flags=_re.MULTILINE)
        # bold+italic  ***text***
        text = _re.sub(r"\*\*\*(.+?)\*\*\*",
            r'<strong><em>\1</em></strong>', text)
        # bold  **text**
        text = _re.sub(r"\*\*(.+?)\*\*",
            r'<strong style="color:#0b3d6e">\1</strong>', text)
        # italic  *text*
        text = _re.sub(r"\*(.+?)\*",
            r'<em style="color:#1a5276">\1</em>', text)
        # bullet lines  "* item" or "- item"
        def _make_li(m):
            return (f'<li style="margin:4px 0;padding-left:4px;'
                    f'border-left:3px solid #42A5F5;padding-left:10px;'
                    f'list-style:none">{m.group(1)}</li>')
        text = _re.sub(r"^[*\-] (.+)$", _make_li, text, flags=_re.MULTILINE)
        # wrap consecutive <li> blocks in <ul>
        text = _re.sub(r"((?:<li[^>]*>.*?</li>\n?)+)",
            r'<ul style="margin:8px 0;padding:0">\1</ul>', text, flags=_re.DOTALL)
        # line breaks
        text = text.replace("\n", "<br>")
        # clean up <br> inside / around block elements
        text = _re.sub(r"<br>\s*(<(?:ul|h[2-4]|li))", r"\1", text)
        text = _re.sub(r"(</(?:ul|h[2-4]|li)>)\s*<br>", r"\1", text)
        return text

    # ── header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
      <div style="width:36px;height:36px;background:linear-gradient(135deg,#1565C0,#42A5F5);
                  border-radius:10px;display:flex;align-items:center;justify-content:center;
                  font-size:1.2rem">🤖</div>
      <div>
        <div style="font-size:1.05rem;font-weight:800;color:#0b3d6e">AI Assistant</div>
        <div style="font-size:0.72rem;color:#4a6a8a">Powered by your live pharmacy data</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── suggestion chips ──────────────────────────────────────────────────────
    _suggestions = [
        "Which medicines need reordering?",
        "Which supplier is underperforming?",
        "Which category has highest expiry risk?",
        "How can revenue improve?",
    ]
    _chip_html = "".join([
        f'<span style="display:inline-block;background:rgba(21,101,192,0.09);'
        f'border:1px solid rgba(21,101,192,0.25);border-radius:20px;'
        f'padding:4px 12px;font-size:0.72rem;font-weight:600;color:#1565C0;'
        f'margin:3px 4px 3px 0;cursor:default">{s}</span>'
        for s in _suggestions
    ])
    st.markdown(
        f'<div style="margin-bottom:14px"><span style="font-size:0.7rem;font-weight:700;'
        f'color:#4a6a8a;text-transform:uppercase;letter-spacing:0.5px">Try asking: </span>'
        f'{_chip_html}</div>',
        unsafe_allow_html=True
    )

    # ── conversation history ──────────────────────────────────────────────────
    _chat_history = st.session_state.get("ai_chat_history", [])

    if not _chat_history:
        st.markdown(
            '<div style="text-align:center;padding:28px;background:rgba(21,101,192,0.04);'
            'border-radius:12px;border:1.5px dashed rgba(21,101,192,0.2);margin-bottom:14px">'
            '<div style="font-size:1.6rem;margin-bottom:6px">💬</div>'
            '<div style="font-size:0.85rem;color:#4a6a8a;font-weight:600">No conversation yet</div>'
            '<div style="font-size:0.75rem;color:#7a9ab0;margin-top:3px">Ask a question below to get started</div>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        for _msg in _chat_history:
            if _msg["role"] == "user":
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-end;margin:8px 0">'
                    f'<div style="background:linear-gradient(135deg,#1565C0,#1976D2);'
                    f'color:#fff;border-radius:16px 16px 4px 16px;'
                    f'padding:10px 16px;max-width:80%;font-size:0.85rem;font-weight:600;'
                    f'box-shadow:0 2px 8px rgba(21,101,192,0.25)">'
                    f'<span style="opacity:0.75;font-size:0.68rem;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.5px">You</span><br>'
                    f'{_msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                _rendered = _md_to_html(_msg["content"])
                st.markdown(
                    f'<div style="display:flex;gap:10px;margin:8px 0 14px">'
                    f'<div style="min-width:32px;height:32px;background:linear-gradient(135deg,#1565C0,#42A5F5);'
                    f'border-radius:8px;display:flex;align-items:center;justify-content:center;'
                    f'font-size:1rem;margin-top:2px">🤖</div>'
                    f'<div style="background:#fff;border:1px solid rgba(21,101,192,0.18);'
                    f'border-radius:4px 16px 16px 16px;padding:14px 18px;'
                    f'font-size:0.84rem;line-height:1.65;color:#1a2a3a;flex:1;'
                    f'box-shadow:0 2px 8px rgba(21,101,192,0.08)">'
                    f'<span style="font-size:0.68rem;font-weight:700;color:#1565C0;'
                    f'text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:6px">AI Assistant</span>'
                    f'{_rendered}</div></div>',
                    unsafe_allow_html=True,
                )

    # ── input row ─────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:rgba(21,101,192,0.04);border:1.5px solid rgba(21,101,192,0.2);'
        'border-radius:12px;padding:10px 12px;margin-top:4px">',
        unsafe_allow_html=True
    )
    user_q = st.text_input(
        "Your question",
        placeholder="💬  Ask anything about your pharmacy data…",
        key="ai_chat_input",
        label_visibility="collapsed",
    )
    _btn_col1, _btn_col2 = st.columns([1, 3])
    with _btn_col1:
        ask_btn = st.button("➤ Ask", key="btn_ask_ai", type="primary", use_container_width=True)
    with _btn_col2:
        if st.button("🗑️ Clear Chat", key="btn_clear_chat", type="secondary", use_container_width=True):
            st.session_state["ai_chat_history"] = []
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if ask_btn and user_q.strip():
        _flat_history = tuple(m["content"] for m in _chat_history)
        with st.spinner("🤔 Thinking…"):
            chat_result = generate_ai_insights(
                _ai_context,
                user_question=user_q.strip(),
                chat_history=_flat_history,
            )
        st.session_state["ai_chat_history"].append({"role": "user",      "content": user_q.strip()})
        st.session_state["ai_chat_history"].append({"role": "assistant", "content": chat_result})
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.markdown('<div class="page-title">ℹ️ About PharmaDash</div>', unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(13,61,110,0.90),rgba(21,101,192,0.85));border:1px solid rgba(255,255,255,0.30);
                border-radius:20px;padding:32px 36px;margin-bottom:24px;
                box-shadow:0 8px 32px rgba(13,45,84,0.30);text-align:center">
      <div style="font-size:3.5rem;margin-bottom:12px">💊</div>
      <div style="font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;
                  color:#fff;letter-spacing:2px;margin-bottom:10px">
        PHARMA<span style="color:#64B5F6">DASH</span>
      </div>
      <div style="font-size:0.92rem;color:rgba(255,255,255,0.88);line-height:1.8;max-width:680px;margin:0 auto">
        A full-stack pharmacy analytics platform built to help multi-store pharmacy chains
        in Hyderabad make smarter, faster, data-driven decisions across inventory,
        sales, supplier management, and business strategy.
      </div>
      <div style="margin-top:16px;font-size:0.78rem;color:rgba(255,255,255,0.50);letter-spacing:1px">
        Version 2.0 &nbsp;|&nbsp; © 2026 PharmaDash Analytics
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="insight-box">
          <h4>🎯 Purpose & Goals</h4>
          <ul>
            <li>Centralise inventory and sales data across <b>5 Hyderabad stores</b>.</li>
            <li>Reduce medicine wastage from <b>expiry and overstocking</b>.</li>
            <li>Enable proactive <b>reorder and procurement</b> decisions.</li>
            <li>Surface revenue trends and <b>top-performing SKUs</b> instantly.</li>
            <li>Provide supplier scorecards to <b>improve lead times</b> and quality.</li>
          </ul>
        </div>

        <div class="insight-box">
          <h4>🗃️ Data Sources</h4>
          <ul>
            <li><b>Fact_Sales_20k.csv</b> — 20,000 sales transactions with date, store, category, customer type.</li>
            <li><b>Fact_Stock_20k.csv</b> — 20,000 stock records with expiry status, batch, quantity, cost.</li>
            <li><b>Dim_Products_20k.csv</b> — Product master with retail price and reorder points.</li>
            <li><b>Dim_Store.csv</b> — Store names, IDs, and Hyderabad location coordinates.</li>
            <li><b>Dim_Suppliers_20k.csv</b> — Supplier quality ratings and lead-time data.</li>
          </ul>
        </div>

        <div class="insight-box">
          <h4>🏥 Stores Covered</h4>
          <ul>
            <li>💊 <b>MedPlus</b> — Ameerpet</li>
            <li>🏥 <b>Apollo</b> — Kukatpally</li>
            <li>🩺 <b>WellCare</b> — Banjara Hills</li>
            <li>💉 <b>HealthFirst</b> — Madhapur</li>
            <li>🧬 <b>CityMed</b> — Dilsukhnagar</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="insight-box">
          <h4>⚙️ Technology Stack</h4>
          <ul>
            <li>🐍 <b>Python 3.11</b> — Core application language</li>
            <li>🎈 <b>Streamlit</b> — Interactive web dashboard framework</li>
            <li>🐼 <b>Pandas + NumPy</b> — Data processing and vectorised operations</li>
            <li>📊 <b>Plotly Express & Graph Objects</b> — Interactive charts and maps</li>
            <li>🗺️ <b>OpenStreetMap via Plotly</b> — Store geo-visualisation</li>
            <li>💾 <b>openpyxl</b> — Excel report export</li>
            <li>🤖 <b>Groq + Llama-3.3-70B</b> — AI-powered pharmacy assistant</li>
          </ul>
        </div>

        <div class="insight-box">
          <h4>✨ Key Features</h4>
          <ul>
            <li>🏠 <b>Home Page</b> — Platform overview with live KPI snapshot.</li>
            <li>📊 <b>Overview</b> — Revenue, units, net profit, GST estimate, expiry and health KPIs.</li>
            <li>🧪 <b>Inventory & Expiry</b> — Batch-level expiry table with status colour-coding.</li>
            <li>📈 <b>Sales & Demand</b> — Monthly trends, top-10 medicines, category splits, Rx vs OTC proxy.</li>
            <li>🏪 <b>Supplier & Store</b> — Map view, supplier ratings, store performance.</li>
            <li>💡 <b>Business Insights</b> — Store underperformance alerts, below-cost category flags, consolidated recommendations.</li>
            <li>🤖 <b>AI Assistant</b> — Groq-powered Llama chat for instant analytics Q&A.</li>
            <li>💰 <b>Net Profit View</b> — Revenue vs COGS vs Wastage per store.</li>
            <li>🔽 <b>Smart Filters</b> — Instant filter + one-click clear, Excel export.</li>
            <li>⚡ <b>Cached Aggregations</b> — Fast reruns with <code>@st.cache_data</code>.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # Performance Optimisations — full width so it fills the gap
    st.markdown("""
        <div class="insight-box">
          <h4>🔧 Performance Optimisations</h4>
          <ul style="column-count:2;column-gap:2rem">
            <li>Category dtypes for low-memory groupby operations.</li>
            <li>NumPy vectorised dot-products for KPI computation.</li>
            <li>Cached filter + aggregation functions keyed by filter params.</li>
            <li>Lazy data loading — CSVs loaded once per server process.</li>
            <li>Static Plotly rendering — mode-bar disabled for faster paint.</li>
          </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box" style="border-left:5px solid #1976D2;
         background:rgba(227,242,253,0.88);text-align:center;margin-top:4px">
      <h4>📬 Contact & Feedback</h4>
      <p style="font-size:0.88rem;color:#1a3a55;margin:0">
        For questions, feature requests, or data updates, please reach out to the
        <b>PharmaDash Analytics Team</b>.<br>
        Use the <b>Download Filtered Report</b> button in the sidebar to export current data as Excel.
      </p>
    </div>
    """, unsafe_allow_html=True)

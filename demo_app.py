	
"""Interactive Streamlit demo for AeroFusion trajectory reconstruction."""

import sys, json, math, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT     = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aero_fusion.step7_serve import reconstruct_flight, load_flight_catalog, load_model
from aero_fusion.step5_kalman import (
    KalmanParams, PreparedFlight,
    _coerce_track, _merge_context_measurements, _wrap_lon_deg,
    along_cross_track_m, great_circle_interpolate, kalman_smooth_gap,
)

st.set_page_config(
    page_title="AeroFusion - NAT Reconstruction",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0a0f1e;
    color: #e2e8f0;
}
.main, .stApp { background: #0a0f1e; }
.block-container { padding-top: 0 !important; padding-bottom: 1.5rem; max-width: 100%; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { top: 10px !important; right: 12px !important; }
.stSpinner > div { border-top-color: #3b82f6 !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #1e293b; }
::-webkit-scrollbar-thumb { background: #475569; border-radius: 3px; }

.top-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-bottom: 1px solid rgba(59,130,246,0.3);
    padding: 16px 28px 14px;
    display: flex; align-items: center; justify-content: space-between;
    margin: -1rem -1rem 0;
    position: relative;
    overflow: hidden;
}
.top-banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #3b82f6, #10b981, #3b82f6, transparent);
}
.banner-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px; font-weight: 800; color: #f8fafc; letter-spacing: 0.5px;
}
.banner-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; letter-spacing: 3px; text-transform: uppercase;
    color: #64748b; margin-top: 3px;
}
.banner-right { display: flex; align-items: center; gap: 20px; }
.status-pill {
    display: flex; align-items: center; gap: 8px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 20px; padding: 6px 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #34d399;
}
.status-pill.offline {
    background: rgba(245,158,11,0.12);
    border-color: rgba(245,158,11,0.35);
    color: #fbbf24;
}
.dot-pulse {
    width: 7px; height: 7px; background: #10b981;
    border-radius: 50%; box-shadow: 0 0 10px rgba(16,185,129,0.7);
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1; transform:scale(1)} 50%{opacity:.5; transform:scale(0.85)} }

.flight-header {
    background: linear-gradient(135deg, #0f172a, #1a2744);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 12px; padding: 18px 22px; margin: 10px 0 8px;
}
.route-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px; font-weight: 800; color: #f8fafc; letter-spacing: 0.5px;
}
.route-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #94a3b8; letter-spacing: 2.5px; margin-top: 5px;
}
.tag { padding: 3px 10px; border-radius: 12px; font-family: 'JetBrains Mono', monospace;
       font-size: 9px; font-weight: 600; letter-spacing: 1px; }
.tag-blue  { background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.4); color: #93c5fd; }
.tag-green { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); color: #6ee7b7; }
.tag-amber { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.4); color: #fcd34d; }
.tag-red   { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.4);  color: #fca5a5; }

.metric-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 18px 20px;
    position: relative; overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform .2s, box-shadow .2s;
    min-height: 130px;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.4); }
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
.mc-baseline::before { background: linear-gradient(90deg,#ef4444,#fca5a5); }
.mc-kalman::before   { background: linear-gradient(90deg,#3b82f6,#93c5fd); }
.mc-gru::before      { background: linear-gradient(90deg,#10b981,#6ee7b7); }
.mc-neutral::before  { background: linear-gradient(90deg,#8b5cf6,#c4b5fd); }

.metric-label {
    font-family: 'JetBrains Mono', monospace; font-size: 7.5px;
    letter-spacing: 3px; text-transform: uppercase; color: #94a3b8; margin-bottom: 10px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace; font-size: 30px;
    font-weight: 800; line-height: 1; color: #f8fafc;
}
.metric-value.red    { color: #f87171; }
.metric-value.blue   { color: #60a5fa; }
.metric-value.green  { color: #34d399; }
.metric-value.amber  { color: #fbbf24; }
.metric-value.purple { color: #a78bfa; }
.metric-sub { font-family: 'Inter', sans-serif; font-size: 10px; color: #94a3b8; margin-top: 6px; }
.mbar-track { width:100%; height:4px; background:#1e293b; border-radius:2px; overflow:hidden; margin-top:10px; }
.mbar-fill  { height:100%; border-radius:2px; }

.sec-head {
    font-family: 'JetBrains Mono', monospace; font-size: 7.5px;
    letter-spacing: 4px; text-transform: uppercase; color: #94a3b8;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 8px; margin: 18px 0 12px;
}

.info-box {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.25);
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 11px 15px; font-size: 11.5px; color: #93c5fd;
    font-family: 'Inter', sans-serif; margin: 8px 0; line-height: 1.7;
}
.info-box.amber {
    background: rgba(245,158,11,0.08);
    border-color: rgba(245,158,11,0.25);
    border-left-color: #f59e0b; color: #fcd34d;
}

.pipeline-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 14px 16px; margin-bottom: 8px;
    min-height: 148px;
    display: flex; flex-direction: column;
}
.pipeline-step {
    font-family: 'JetBrains Mono', monospace; font-size: 8px;
    letter-spacing: 2px; color: #94a3b8; margin-bottom: 4px;
    text-transform: uppercase;
}
.pipeline-title {
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    font-weight: 700; color: #f8fafc; margin-bottom: 4px;
}
.pipeline-desc {
    font-family: 'Inter', sans-serif; font-size: 10.5px; color: #cbd5e1; line-height: 1.6;
}

.results-table {
    width: 100%; border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
}
.results-table th {
    background: rgba(255,255,255,0.04); color: #94a3b8;
    font-size: 7.5px; letter-spacing: 2.5px; text-transform: uppercase;
    padding: 11px 15px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.08);
}
.results-table td { padding: 11px 15px; border-bottom: 1px solid rgba(255,255,255,0.04); color: #e2e8f0; }
.results-table tr:hover td { background: rgba(255,255,255,0.03); }
.td-rank1 { color: #34d399 !important; font-weight: 700; }
.td-rank2 { color: #60a5fa !important; font-weight: 700; }
.td-rank3 { color: #f87171 !important; }

.map-wrap {
    border-radius: 14px; overflow: hidden;
    border: 1px solid rgba(59,130,246,0.2);
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
}

.legend-strip {
    display: flex; gap: 24px; justify-content: center;
    padding: 10px 0 12px; font-size: 10.5px; color: #64748b;
    font-family: 'JetBrains Mono', monospace; flex-wrap: wrap;
    background: #0f172a; border-top: 1px solid rgba(255,255,255,0.06);
}
.lsym { margin-right: 5px; }

section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .block-container { padding-top: .6rem !important; }
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #cbd5e1; }

.stSelectbox label, .stRadio label, .stCheckbox label {
    color: #64748b !important; font-size: 9px !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
}
.stSelectbox > div > div {
    background: #1e293b !important; border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important; color: #cbd5e1 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stSelectbox [data-baseweb="select"] * { color: #cbd5e1 !important; }
[data-baseweb="popover"] {
    background: #1e293b !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    box-shadow: 0 16px 40px rgba(0,0,0,0.6) !important;
}
[data-baseweb="menu"] { background: #1e293b !important; color: #cbd5e1 !important; }
[data-baseweb="menu"] li,
[data-baseweb="menu"] ul,
[role="listbox"] [role="option"] {
    background: #1e293b !important; color: #cbd5e1 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[role="listbox"] [role="option"][aria-selected="true"] {
    background: rgba(59,130,246,0.2) !important; color: #93c5fd !important;
}
[role="listbox"] [role="option"]:hover {
    background: rgba(255,255,255,0.05) !important; color: #f8fafc !important;
}
.stCheckbox span { color: #94a3b8 !important; }

.model-badge {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 10px; padding: 12px 14px; margin: 8px 0;
}
.model-badge.warn {
    background: rgba(245,158,11,0.1);
    border-color: rgba(245,158,11,0.3);
}

.flight-info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 13px 14px; margin: 8px 0;
}
.fi-route {
    font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 700;
    color: #f8fafc; letter-spacing: .5px; margin-bottom: 8px;
}
.fi-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.fi-item label {
    display: block; font-family: 'JetBrains Mono', monospace; font-size: 7px;
    letter-spacing: 2px; text-transform: uppercase; color: #64748b; margin-bottom: 2px;
}
.fi-item span { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: #94a3b8; }
.fi-gap { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; color: #60a5fa; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg,#1d4ed8,#3b82f6) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 10.5px !important;
    font-weight: 700 !important; letter-spacing: 2px !important;
    padding: 11px 18px !important; width: 100% !important;
    transition: all .2s !important; box-shadow: 0 2px 12px rgba(59,130,246,.4) !important;
}
div[data-testid="stButton"] > button span,
div[data-testid="stButton"] > button p,
div[data-testid="stButton"] > button div { color: #ffffff !important; }
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg,#2563eb,#60a5fa) !important;
    box-shadow: 0 6px 24px rgba(59,130,246,.55) !important;
    transform: translateY(-1px) !important;
}

.stats-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px; padding: 12px 14px; margin-top: 6px;
}
.stats-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.stats-row:last-child { border-bottom: none; }
.stats-method { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #94a3b8;
                letter-spacing: 1px; text-transform: uppercase; }
.stats-value { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; }

.loading-screen {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 14px; height: 460px;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column; gap: 16px; margin-top: 8px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.ls-icon { font-size: 48px; animation: float 3s ease-in-out infinite; }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
.ls-title {
    font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #94a3b8;
    letter-spacing: 3px; text-transform: uppercase; text-align: center;
}
.ls-sub { font-size: 11px; color: #475569; font-family: 'Inter', sans-serif; text-align: center; }

.sep { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 10px 0; }

.glow-green { text-shadow: 0 0 12px rgba(52,211,153,.6); }
.glow-blue  { text-shadow: 0 0 12px rgba(96,165,250,.6); }
</style>
""", unsafe_allow_html=True)

EARTH_R    = 6371.0
STEP2_ROOT = PROJECT_ROOT / "artifacts" / "step2_clean"

AIRLINE_MAP = {
    "BAW":"British Airways","AAL":"American Airlines","DAL":"Delta Air Lines",
    "UAL":"United Airlines","AFR":"Air France","DLH":"Lufthansa",
    "IBE":"Iberia","TAP":"TAP Air Portugal","EIN":"Aer Lingus",
    "VIR":"Virgin Atlantic","ICE":"Icelandair","RYR":"Ryanair",
    "ACA":"Air Canada","WJA":"WestJet","AZA":"ITA Airways",
    "THY":"Turkish Airlines","SWR":"Swiss","AUA":"Austrian Airlines",
    "KLM":"KLM","SAS":"Scandinavian Airlines","NOZ":"Norwegian",
    "MSR":"EgyptAir","ETH":"Ethiopian Airlines","ELY":"El Al",
    "TOM":"TUI Airways","TRA":"Transavia","WZZ":"Wizz Air",
    "QTR":"Qatar Airways","UAE":"Emirates","ETD":"Etihad",
    "EJM":"ExecJet","N47":"Private","GEM":"Gemini Air Cargo",
}

DEMO_HIGHLIGHTS = [
    # Route                Airline  GRU err  Why included
    "20231105_a0a54d_030849_041552",  # EWR→BRU  UAL    16.8km   EASTBOUND, GRU best showcase
    "20250329_4b1883_132449_153308",  # ZRH→YYZ  SWR    29.3km   Zurich→Toronto
    "20250319_4005c0_152300_180926",  # DOH→JFK  BAW    79.5km   Doha→New York
    "20250801_4007f1_173814_202441",  # LGW→JFK  BAW    45.1km   Classic transatlantic
    "20250729_400773_193333_215630",  # LHR→JFK  BAW    76.6km   Heathrow 2025
    "20250214_400773_210838_231648",  # LHR→ATL  BAW    62.2km   London→Atlanta
    "20250703_aab812_102726_123306",  # CDG→CLT  AAL    98.2km   Paris→Charlotte
    "20250731_485f82_101311_120739",  # AMS→SFO  KLM    40.0km   Amsterdam→San Francisco
    "20250313_aa2184_162016_182107",  # MAD→DFW  AAL    92.5km   Madrid→Dallas
    "20240710_4007f0_092932_103853",  # MCO→LGW  BAW    38.0km   EASTBOUND, Orlando→London
    "20241220_4b1883_192140_213810",  # ZRH→BOS  SWR    66.6km   Zurich→Boston, 2024
    "20231118_4005bd_022514_042151",  # JFK→LHR  BAW    69.9km   EASTBOUND, 2023
    # Original demo flights added back
    "20231114_485341_032118_043049",  # IAH→AMS  KLM    EASTBOUND Houston→Amsterdam
    "20240827_4b187f_064414_074830",  # JFK→ZRH  SWR    EASTBOUND JFK→Zurich
    "20231011_a0f427_041753_051756",  # EWR→?    UAL    EASTBOUND Newark outbound
    "20250703_4005c0_202353_224235",  # LHR→JFK  BAW    139min 23pts
    "20250326_4006c1_151222_174303",  # LGW→MCO  BAW    151min 23pts
    "20240822_ac21af_054936_065856",  # PHL→MAD  AAL    26.3km   EASTBOUND, strong GRU
    "20250318_400773_201647_221532",  # LHR→EWR  BAW    119min 16pts
]

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# dep/arr overrides for demo flights — stored as 3-letter IATA codes
DEMO_AIRPORTS = {
    "20231105_a0a54d_030849_041552": ("EWR", "BRU"),   # Newark → Brussels (EASTBOUND)
    "20250329_4b1883_132449_153308": ("ZRH", "YYZ"),   # Zurich → Toronto
    "20250319_4005c0_152300_180926": ("DOH", "JFK"),   # Doha → New York JFK
    "20250801_4007f1_173814_202441": ("LGW", "JFK"),   # London Gatwick → JFK
    "20250729_400773_193333_215630": ("LHR", "JFK"),   # London Heathrow → JFK
    "20250214_400773_210838_231648": ("LHR", "ATL"),   # London Heathrow → Atlanta
    "20250703_aab812_102726_123306": ("CDG", "CLT"),   # Paris → Charlotte
    "20250731_485f82_101311_120739": ("AMS", "SFO"),   # Amsterdam → San Francisco
    "20250313_aa2184_162016_182107": ("MAD", "DFW"),   # Madrid → Dallas
    "20240710_4007f0_092932_103853": ("MCO", "LGW"),   # Orlando → London (EASTBOUND)
    "20241220_4b1883_192140_213810": ("ZRH", "BOS"),   # Zurich → Boston
    "20231118_4005bd_022514_042151": ("JFK", "LHR"),   # New York JFK → London (EASTBOUND)
    "20231114_485341_032118_043049": ("IAH", "AMS"),   # Houston → Amsterdam (EASTBOUND)
    "20240827_4b187f_064414_074830": ("JFK", "ZRH"),   # JFK → Zurich (EASTBOUND)
    "20231011_a0f427_041753_051756": ("EWR", None),    # Newark outbound (arr unknown)
    "20250703_4005c0_202353_224235": ("LHR", "JFK"),   # London Heathrow → JFK
    "20250326_4006c1_151222_174303": ("LGW", "MCO"),   # London Gatwick → Orlando
    "20240822_ac21af_054936_065856": ("PHL", "MAD"),   # Philadelphia → Madrid (EASTBOUND)
    "20250318_400773_201647_221532": ("LHR", "EWR"),   # London Heathrow → Newark
}

def hav(lat1,lon1,lat2,lon2):
    p1,p2=math.radians(lat1),math.radians(lat2)
    dp=math.radians(lat2-lat1);dl=math.radians(lon2-lon1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return EARTH_R*2*math.asin(math.sqrt(min(1.,max(0.,a))))

def clean(v,fb="-"):
    s=str(v).strip() if v is not None else ""
    return fb if s in ("","nan","None","NaN") else s

def airline(cs):
    if not cs or cs in ("nan","None","-"): return ""
    return AIRLINE_MAP.get(cs[:3].upper(),"")

def fmt_airport(code):
    c=clean(code)
    if c=="-": return "N/A"
    if len(c)==4: return c[1:].upper() if c[0] in "KEGLlBLF" else c.upper()
    return c.upper()

def sort_ts(df):
    w=df.copy()
    if "timestamp" in w.columns:
        w["timestamp"]=pd.to_datetime(w["timestamp"],errors="coerce")
    return w.dropna(subset=["timestamp","latitude","longitude"]).sort_values("timestamp").reset_index(drop=True)

def point_errors(pred_df, truth_df):
    if pred_df.empty or truth_df.empty: return [],float("nan")
    n=min(len(pred_df),len(truth_df))
    errs=[hav(float(pred_df["latitude"].iloc[i]),float(pred_df["longitude"].iloc[i]),
               float(truth_df["latitude"].iloc[i]),float(truth_df["longitude"].iloc[i])) for i in range(n)]
    return errs, (sum(errs)/len(errs) if errs else float("nan"))

def pct_bar(val, max_val, color):
    pct = min(100, max(0, (val / max_val * 100)) if max_val > 0 else 0)
    return (f"<div class='mbar-track'>"
            f"<div class='mbar-fill' style='width:{pct:.1f}%;background:{color};'></div>"
            f"</div>")

@st.cache_resource
def get_model(): return load_model()

@st.cache_data
def get_catalog():
    df=load_flight_catalog()
    if df.empty: return df
    def pd_(s):
        try: return datetime.strptime(str(s)[:8],"%Y%m%d")
        except: return None
    df["_date"]=df["segment_id"].apply(pd_)
    df["_year"]=df["_date"].apply(lambda d:d.year if d else None)
    df["_month"]=df["_date"].apply(lambda d:d.month if d else None)
    df["_day"]=df["_date"].apply(lambda d:d.day if d else None)
    return df

@st.cache_data
def get_kalman_params():
    p=PROJECT_ROOT/"artifacts"/"step5_kalman"/"test_summary.json"
    default=KalmanParams(measurement_std_m=2000.,accel_std_along_mps2=.001,accel_std_cross_mps2=.01)
    if not p.exists(): return default
    try:
        s=json.loads(p.read_text())["selected_params"]
        return KalmanParams(measurement_std_m=float(s["measurement_std_m"]),
                            accel_std_along_mps2=float(s["accel_std_along_mps2"]),
                            accel_std_cross_mps2=float(s["accel_std_cross_mps2"]))
    except: return default

@st.cache_data
def get_model_stats():
    out={"gru_median":68.5,"kf_median":88.3,"bl_median":131.0}
    gp=PROJECT_ROOT/"artifacts"/"step5_gru"/"test_summary.json"
    kp=PROJECT_ROOT/"artifacts"/"step5_kalman"/"test_summary.json"
    if gp.exists():
        g=json.loads(gp.read_text())
        out["gru_median"]=g.get("gru_median_error_km",out["gru_median"])
    if kp.exists():
        k=json.loads(kp.read_text())
        ts=k.get("test_summary",{})
        out["kf_median"]=ts.get("kalman_median_error_km",out["kf_median"])
        out["bl_median"]=ts.get("baseline_median_error_km",out["bl_median"])
    return out

@st.cache_data(ttl=90)
def fetch_live_nat():
    """Fetch current aircraft positions from OpenSky in the NAT area."""
    url = ("https://opensky-network.org/api/states/all"
           "?lamin=35&lamax=75&lomin=-75&lomax=15")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AeroFusion-Demo/1.0"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read().decode())
        states = data.get("states") or []
        aircraft = []
        for s in states:
            if not s or s[8]:  # skip on-ground
                continue
            lon, lat = s[5], s[6]
            if lat is None or lon is None:
                continue
            lat, lon = float(lat), float(lon)
            alt  = float(s[13] or s[7] or 0)
            vel  = float(s[9]  or 0)
            hdg  = float(s[10] or 0)
            cs   = (s[1] or "").strip()
            if alt < 7000 or vel < 150:
                continue
            aircraft.append({
                "icao24": s[0], "callsign": cs, "lat": lat, "lon": lon,
                "alt_ft": round(alt * 3.28084), "vel_kts": round(vel * 1.94384),
                "hdg": hdg, "country": s[2] or "",
            })
        return aircraft, int(data.get("time", 0))
    except Exception as exc:
        return None, str(exc)


def build_live_map(aircraft: list) -> go.Figure:
    """Build Plotly map of live NAT traffic."""
    westbound = [a for a in aircraft if 200 <= a["hdg"] <= 310]
    eastbound = [a for a in aircraft if 40  <= a["hdg"] <= 130]
    other     = [a for a in aircraft if a not in westbound and a not in eastbound]

    def hover(a):
        cs = a["callsign"] or a["icao24"]
        return (f"<b>{cs}</b><br>"
                f"Hdg {a['hdg']:.0f}° | {a['vel_kts']} kts<br>"
                f"FL{a['alt_ft']//100:03d}<br>"
                f"{a['country']}<extra></extra>")

    fig = go.Figure()

    def add_layer(group, color, label, symbol="airplane"):
        if not group:
            return
        fig.add_trace(go.Scattermapbox(
            lat=[a["lat"] for a in group],
            lon=[a["lon"] for a in group],
            mode="markers",
            marker=dict(size=10, color=color, opacity=0.9, symbol="circle"),
            name=label,
            hovertemplate=[hover(a) for a in group],
        ))

    add_layer(westbound, "#10b981", f"Westbound ({len(westbound)})")
    add_layer(eastbound, "#60a5fa", f"Eastbound ({len(eastbound)})")
    add_layer(other,     "#94a3b8", f"Other ({len(other)})")

    # NAT track corridor shading (approximate OTS box)
    fig.add_trace(go.Scattermapbox(
        lat=[65, 65, 45, 45, 65], lon=[-10, -60, -60, -10, -10],
        mode="lines", line=dict(width=1, color="rgba(59,130,246,0.3)"),
        fill="toself", fillcolor="rgba(59,130,246,0.04)",
        name="NAT OTS corridor", showlegend=True, hoverinfo="skip",
    ))

    fig.update_layout(
        mapbox=dict(style="carto-darkmatter",
                    center=dict(lat=55, lon=-30), zoom=2.8),
        paper_bgcolor="#0a0f1e", margin=dict(l=0, r=0, t=0, b=0), height=560,
        showlegend=True,
        legend=dict(bgcolor="rgba(10,15,30,0.9)", bordercolor="rgba(255,255,255,0.12)",
                    borderwidth=1, font=dict(color="#cbd5e1", size=11,
                                             family="JetBrains Mono"),
                    x=0.01, y=0.99, xanchor="left", yanchor="top"),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="rgba(255,255,255,0.15)",
                        font=dict(family="JetBrains Mono", size=11, color="#f8fafc")),
    )
    return fig


model        = get_model()
catalog      = get_catalog()
kf_params    = get_kalman_params()
model_stats  = get_model_stats()

with st.sidebar:
    if catalog.empty:
        st.error("No flights found in catalog."); st.stop()

    st.markdown('<div class="sec-head">FLIGHT SELECTION</div>', unsafe_allow_html=True)

    demo_mode = st.checkbox("🎯  Demo highlights only", value=True)

    years = sorted(catalog["_year"].dropna().unique().astype(int).tolist(), reverse=True)

    if demo_mode:
        sel_year  = int(str(DEMO_HIGHLIGHTS[0])[:4])
        sel_month = int(str(DEMO_HIGHLIGHTS[0])[4:6])
        fdf = catalog[catalog["segment_id"].astype(str).isin(DEMO_HIGHLIGHTS)].copy()
        if not fdf.empty:
            demo_order = {sid: i for i, sid in enumerate(DEMO_HIGHLIGHTS)}
            fdf["_demo_order"] = fdf["segment_id"].astype(str).map(demo_order)
            fdf = fdf.sort_values("_demo_order").reset_index(drop=True)
        else:
            st.warning("No demo flights found in catalog.")
            fdf = catalog.head(10)
    else:
        cy, cm = st.columns(2)
        with cy:
            sel_year = st.selectbox("Year", years, index=0)
        with cm:
            mdf = catalog[catalog["_year"] == sel_year]
            months = sorted(mdf["_month"].dropna().unique().astype(int).tolist())
            sel_month = st.selectbox("Month", months,
                format_func=lambda m: MONTH_NAMES.get(m, str(m)), index=0)
        fdf = catalog[(catalog["_year"] == sel_year) & (catalog["_month"] == sel_month)].reset_index(drop=True)
        if "gap_duration_minutes" in fdf.columns:
            fdf = fdf.sort_values("gap_duration_minutes", ascending=False).reset_index(drop=True)

    def lbl(row):
        sid_ = str(row.get("segment_id", ""))
        ov   = DEMO_AIRPORTS.get(sid_)
        dep  = fmt_airport(ov[0] if ov else row.get("estdepartureairport"))
        arr  = fmt_airport(ov[1] if ov else row.get("estarrivalairport"))
        cs_  = clean(row.get("flight_callsign") or row.get("segment_callsign"))
        gap_ = row.get("gap_duration_minutes", 0)
        day_ = int(row.get("_day", 0))
        mon_ = int(row.get("_month", sel_month))
        route = (f"{dep}→{arr}" if dep != "N/A" and arr != "N/A"
                 else f"Flt {cs_}" if cs_ != "-" and len(cs_) >= 3
                 else f"ICAO {clean(row.get('icao24','')[:8].upper())}")
        al = airline(cs_) if cs_ != "-" else ""
        al_str = f"  -  {al[:18]}" if al else ""
        return f"{route}  -  {gap_:.0f}min  -  {day_:02d}{MONTH_NAMES.get(mon_,'')}{al_str}"

    labels = [lbl(r) for _, r in fdf.iterrows()]
    n_flights = len(fdf)
    st.markdown(
        f"<div style='font-size:9px;color:#475569;font-family:JetBrains Mono,monospace;"
        f"margin-bottom:4px;letter-spacing:1px;'>{n_flights} FLIGHTS AVAILABLE</div>",
        unsafe_allow_html=True)
    sel_lbl = st.selectbox("Flight", labels, index=0, label_visibility="collapsed")
    sel_idx = labels.index(sel_lbl)
    sel_row = fdf.iloc[sel_idx]
    segment_id = str(sel_row["segment_id"])

    _ov     = DEMO_AIRPORTS.get(segment_id)
    dep     = fmt_airport(_ov[0] if _ov else sel_row.get("estdepartureairport"))
    arr     = fmt_airport(_ov[1] if _ov else sel_row.get("estarrivalairport"))
    cs      = clean(sel_row.get("flight_callsign") or sel_row.get("segment_callsign"))
    icao    = clean(sel_row.get("icao24", "")).upper()
    gap_cat = float(sel_row.get("gap_duration_minutes", 0))
    al      = airline(cs) if cs != "-" else ""
    route_d = f"{dep} → {arr}" if dep != "N/A" and arr != "N/A" else "NORTH ATLANTIC"
    day_s   = f"{int(sel_row.get('_day',1)):02d} {MONTH_NAMES.get(int(sel_row.get('_month',1)),'to')} {int(sel_row.get('_year',2024))}"

    st.markdown(f"""
    <div class='flight-info-card'>
      <div class='fi-route'>{route_d}</div>
      <div class='fi-grid'>
        <div class='fi-item'>
          <label>Callsign</label>
          <span>{cs if cs!="-" else "-"}</span>
        </div>
        <div class='fi-item'>
          <label>ICAO24</label>
          <span>{icao[:8] if icao else "-"}</span>
        </div>
        <div class='fi-item' style='margin-top:6px;'>
          <label>ADS-C Gap</label>
          <span class='fi-gap'>{gap_cat:.0f} min</span>
        </div>
        <div class='fi-item' style='margin-top:6px;'>
          <label>Date</label>
          <span>{day_s}</span>
        </div>
      </div>
      {f'<div style="margin-top:6px;font-family:JetBrains Mono,monospace;font-size:8px;color:#475569;letter-spacing:1.5px;">{al.upper()}</div>' if al else ""}
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">MAP LAYERS</div>', unsafe_allow_html=True)
    show_truth    = st.checkbox("ADS-C ground truth", value=True)
    show_baseline = st.checkbox("Great-circle baseline", value=True)
    show_kalman   = st.checkbox("Kalman filter", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    rerun_clicked = st.button("↺  RE-ANALYZE TRAJECTORY", use_container_width=True)

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    imp_gru_pct = (1 - model_stats["gru_median"] / model_stats["bl_median"]) * 100
    imp_kf_pct  = (1 - model_stats["kf_median"]  / model_stats["bl_median"]) * 100
    st.markdown(f"""
    <div class='stats-panel'>
      <div style='font-family:JetBrains Mono,monospace;font-size:7.5px;letter-spacing:3px;
                  text-transform:uppercase;color:#475569;margin-bottom:10px;'>
        TEST SET - 240 FLIGHTS</div>
      <div class='stats-row'>
        <span class='stats-method'>BASELINE</span>
        <span class='stats-value' style='color:#f87171;'>{model_stats['bl_median']:.0f} km</span>
      </div>
      <div class='stats-row'>
        <span class='stats-method'>KALMAN <span style='color:#475569;font-size:8px;'>v{imp_kf_pct:.0f}%</span></span>
        <span class='stats-value' style='color:#60a5fa;'>{model_stats['kf_median']:.0f} km</span>
      </div>
      <div class='stats-row'>
        <span class='stats-method'>GRU <span style='color:#34d399;font-size:8px;'>v{imp_gru_pct:.0f}%</span></span>
        <span class='stats-value' style='color:#34d399;'>{model_stats['gru_median']:.0f} km</span>
      </div>
      <div style='font-family:JetBrains Mono,monospace;font-size:7.5px;color:#475569;
                  margin-top:10px;line-height:2.1;border-top:1px solid rgba(255,255,255,0.05);padding-top:8px;'>
        Dataset - {len(catalog):,} flights<br>
        Period - 2023-2025<br>
        Area - Shanwick / North Atlantic<br>
        USJ Lebanon - AeroEngineering
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class='top-banner'>
  <div style='display:flex;align-items:center;gap:20px;'>
    <div>
      <div class='banner-logo'>✈ AeroFusion</div>
      <div class='banner-sub'>North Atlantic Trajectory Reconstruction - Shanwick / Gander OANC</div>
    </div>
  </div>
  <div class='banner-right'>
    {'<div class="status-pill"><div class="dot-pulse"></div>GRU ACTIVE - BiGRU 693K</div>'
     if model else
     '<div class="status-pill offline">⚠ GRU OFFLINE - BASELINE ONLY</div>'}
    <div style='font-family:JetBrains Mono,monospace;font-size:8.5px;text-align:right;'>
      <div style='color:#475569;'>ADS-B + ADS-C SENSOR FUSION</div>
      <div style='color:#334155;margin-top:3px;'>BiGRU - KALMAN - GREAT-CIRCLE</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

tab_hist, tab_live = st.tabs(["Historical Flight Analysis", "Live NAT Tracker"])

with tab_live:
    st.markdown('<div class="sec-head" style="margin-top:6px;">LIVE NORTH ATLANTIC TRAFFIC - OPENSKY NETWORK</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
      Real-time aircraft positions over the North Atlantic fetched from the
      <b>OpenSky Network</b> public REST API (refreshed every 90 s).
      Flights shown are at cruise altitude (&gt;FL230) and cruise speed (&gt;290 kts).
      Aircraft in the shaded NAT OTS corridor are the operational context where
      AeroFusion's reconstruction model applies — they lose ADS-B coverage for 60-220 min.
    </div>""", unsafe_allow_html=True)

    live_col1, live_col2, live_col3, live_col4 = st.columns(4)
    refresh_live = live_col1.button("Refresh Live Traffic", use_container_width=True)
    if refresh_live:
        st.cache_data.clear()

    with st.spinner("Fetching live positions from OpenSky..."):
        live_aircraft, live_ts = fetch_live_nat()

    if live_aircraft is None:
        st.markdown(f"""
        <div class='info-box amber'>
          OpenSky API unavailable: <code>{live_ts}</code><br>
          The public OpenSky endpoint may be rate-limited (&lt;400 req/day without account).
          Try again in a few minutes or <a href='https://opensky-network.org' style='color:#fcd34d;'>register a free account</a>.
        </div>""", unsafe_allow_html=True)
    else:
        westbound_live = [a for a in live_aircraft if 200 <= a["hdg"] <= 310]
        eastbound_live = [a for a in live_aircraft if 40  <= a["hdg"] <= 130]
        nat_box        = [a for a in live_aircraft if -60 <= a["lon"] <= -10 and 45 <= a["lat"] <= 65]
        ts_str = datetime.utcfromtimestamp(live_ts).strftime("%H:%M UTC") if live_ts else "unknown"

        with live_col2:
            st.markdown(f"""
            <div class='metric-card mc-gru' style='min-height:80px;padding:12px 16px;'>
              <div class='metric-label'>NAT Aircraft</div>
              <div class='metric-value green' style='font-size:24px;'>{len(live_aircraft)}</div>
            </div>""", unsafe_allow_html=True)
        with live_col3:
            st.markdown(f"""
            <div class='metric-card mc-kalman' style='min-height:80px;padding:12px 16px;'>
              <div class='metric-label'>In OTS Corridor</div>
              <div class='metric-value blue' style='font-size:24px;'>{len(nat_box)}</div>
            </div>""", unsafe_allow_html=True)
        with live_col4:
            st.markdown(f"""
            <div class='metric-card mc-neutral' style='min-height:80px;padding:12px 16px;'>
              <div class='metric-label'>Last Update</div>
              <div class='metric-value purple' style='font-size:18px;margin-top:4px;'>{ts_str}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="map-wrap">', unsafe_allow_html=True)
        st.plotly_chart(build_live_map(live_aircraft), use_container_width=True,
                        config={"displayModeBar": False, "scrollZoom": True})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='legend-strip'>
          <span><span class='lsym' style='color:#10b981;font-weight:700;'>●</span> Westbound (NAT typical)</span>
          <span><span class='lsym' style='color:#60a5fa;font-weight:700;'>●</span> Eastbound (overnight)</span>
          <span><span class='lsym' style='color:#94a3b8;font-weight:700;'>●</span> Other</span>
          <span><span class='lsym' style='color:#3b82f6;'>&#9632;</span> Approx. NAT OTS corridor</span>
          <span style='color:#475569;font-size:9px;'>Data: OpenSky Network - public API</span>
        </div>""", unsafe_allow_html=True)

        if nat_box:
            st.markdown('<div class="sec-head" style="margin-top:16px;">AIRCRAFT CURRENTLY IN NAT OTS CORRIDOR</div>',
                        unsafe_allow_html=True)
            rows = sorted(nat_box, key=lambda a: -a["vel_kts"])[:20]
            table_rows = "".join(
                f"<tr><td>{a['callsign'] or a['icao24']}</td>"
                f"<td>{a['country']}</td>"
                f"<td>{a['lat']:.2f}&deg; N</td>"
                f"<td>{a['lon']:.2f}&deg;</td>"
                f"<td>FL{a['alt_ft']//100:03d}</td>"
                f"<td>{a['vel_kts']} kts</td>"
                f"<td>{a['hdg']:.0f}&deg;</td></tr>"
                for a in rows
            )
            st.markdown(f"""
            <table class='results-table'>
              <thead><tr>
                <th>Callsign</th><th>Country</th><th>Lat</th><th>Lon</th>
                <th>Altitude</th><th>Speed</th><th>Heading</th>
              </tr></thead>
              <tbody>{table_rows}</tbody>
            </table>
            <div style='font-family:JetBrains Mono,monospace;font-size:8px;color:#475569;margin-top:6px;'>
              These aircraft are currently in or approaching the ADS-C gap zone where AeroFusion reconstructs paths.
              Top 20 by speed shown.
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='info-box' style='margin-top:16px;'>
          <b>How reconstruction applies here:</b> When these aircraft enter the oceanic gap
          (&gt;60&deg;W), ADS-B coverage drops out for 60-220 minutes. ATC relies on
          ADS-C position reports every ~14 min. AeroFusion's BiGRU model reconstructs
          the full trajectory using the last 64 ADS-B fixes before entry and the first
          32 fixes after exit — achieving <b>47% lower error than great-circle interpolation</b>.
        </div>""", unsafe_allow_html=True)

tab_hist.__enter__()

rh = f"{dep} → {arr}" if dep != "N/A" and arr != "N/A" else "North Atlantic Crossing"
cs_tag   = f"<span class='tag tag-blue'>{cs}</span>" if cs != "-" else ""
al_tag   = f"<span class='tag tag-blue'>{al.upper()}</span>" if al else ""
gap_tag  = f"<span class='tag tag-amber'>{gap_cat:.0f} MIN ADS-C GAP</span>"
icao_tag = f"<span class='tag tag-blue'>ICAO {icao[:8]}</span>" if icao else ""

st.markdown(f"""
<div class='flight-header' style='margin-top:10px;'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
    <div>
      <div class='route-text'>{rh}</div>
      <div class='route-meta'>{icao}  -  {gap_cat:.0f} MIN ADS-C SPAN  -  {day_s}  -  SHANWICK / GANDER</div>
      <div style='display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;'>
        {cs_tag}{al_tag}{gap_tag}{icao_tag}
      </div>
    </div>
    <div style='text-align:right;padding-top:4px;'>
      <div style='font-family:JetBrains Mono,monospace;font-size:7.5px;color:#475569;letter-spacing:2.5px;'>SEGMENT ID</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:9.5px;color:#475569;margin-top:2px;'>{segment_id}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

for k in ["last_seg", "gru_res", "base_res"]:
    if k not in st.session_state:
        st.session_state[k] = None

needs_reco = (st.session_state.last_seg != segment_id) or rerun_clicked

if needs_reco:
    loading_ph = st.empty()
    loading_ph.markdown("""
    <div class='loading-screen'>
      <div class='ls-icon'>🌊</div>
      <div class='ls-title'>Running Inference…</div>
      <div class='ls-sub'>GRU forward pass - Kalman RTS smoother - Great-circle baseline</div>
    </div>""", unsafe_allow_html=True)

    try:
        with st.spinner(""):
            st.session_state.gru_res  = reconstruct_flight(segment_id, use_gru=True)
            st.session_state.base_res = reconstruct_flight(segment_id, use_gru=False)
            st.session_state.last_seg = segment_id
        loading_ph.empty()
    except Exception as e:
        loading_ph.empty()
        st.error(f"Reconstruction failed for `{segment_id}`: {e}")
        st.stop()

gru_res  = st.session_state.gru_res
base_res = st.session_state.base_res

if gru_res is None or base_res is None:
    st.markdown("""
    <div class='loading-screen'>
      <div class='ls-icon'>🌊</div>
      <div class='ls-title'>Select a Flight to Begin</div>
      <div class='ls-sub'>Choose a flight in the sidebar - inference runs automatically</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

adsc_path = STEP2_ROOT / "flights" / segment_id / "adsc_clean.parquet"
adsc_df   = pd.DataFrame()
gru_errs  = []; bl_errs = []; gru_mean = float("nan"); bl_mean = float("nan")
bl_comp_df = pd.DataFrame()
metric_error = None

if adsc_path.exists():
    try:
        adsc_df = sort_ts(pd.read_parquet(adsc_path))

        track = sort_ts(pd.DataFrame(base_res["track"]))
        bef = track[track["source"] == "adsb_before"]
        aft = track[track["source"] == "adsb_after"]
        if not bef.empty and not aft.empty and not adsc_df.empty:
            t0  = pd.Timestamp(bef["timestamp"].iloc[-1])
            t1  = pd.Timestamp(aft["timestamp"].iloc[0])
            dur = (t1 - t0).total_seconds()
            if dur > 0:
                slat = float(bef["latitude"].iloc[-1])
                slon = float(bef["longitude"].iloc[-1])
                elat = float(aft["latitude"].iloc[0])
                elon = float(aft["longitude"].iloc[0])
                rows = []
                for ts in adsc_df["timestamp"].tolist():
                    tau = float(np.clip(
                        (pd.Timestamp(ts) - t0).total_seconds() / dur, 0, 1))
                    la, lo = great_circle_interpolate(slat, slon, elat, elon, tau)
                    rows.append({"timestamp": pd.Timestamp(ts),
                                 "latitude": float(la), "longitude": float(lo)})
                bl_comp_df = pd.DataFrame(rows)
                bl_errs, bl_mean = point_errors(bl_comp_df, adsc_df)

        gru_wps_raw = pd.DataFrame(gru_res.get("gru_waypoints", []))
        if not gru_wps_raw.empty and not adsc_df.empty:
            n = min(len(gru_wps_raw), len(adsc_df))
            gru_errs = [
                hav(float(gru_wps_raw["latitude"].iloc[i]),
                    float(gru_wps_raw["longitude"].iloc[i]),
                    float(adsc_df["latitude"].iloc[i]),
                    float(adsc_df["longitude"].iloc[i]))
                for i in range(n)
            ]
            gru_mean = sum(gru_errs) / len(gru_errs) if gru_errs else float("nan")
    except Exception as exc:
        metric_error = f"{type(exc).__name__}: {exc}"

imp = (1 - gru_mean / bl_mean) * 100 if (math.isfinite(gru_mean) and math.isfinite(bl_mean) and bl_mean > 0) else float("nan")

def build_kalman(seg_id, base_res, adsc_df):
    if adsc_df.empty: return None
    try:
        track = sort_ts(pd.DataFrame(base_res["track"]))
        bef   = track[track["source"] == "adsb_before"]
        aft   = track[track["source"] == "adsb_after"]
        if bef.empty or aft.empty: return None

        t0  = pd.Timestamp(bef["timestamp"].iloc[-1])
        t1  = pd.Timestamp(aft["timestamp"].iloc[0])
        dur = (t1 - t0).total_seconds()
        if dur <= 0: return None

        bc = _coerce_track(pd.read_parquet(STEP2_ROOT / "flights" / seg_id / "adsb_before_clean.parquet"))
        ac = _coerce_track(pd.read_parquet(STEP2_ROOT / "flights" / seg_id / "adsb_after_clean.parquet"))
        if bc.empty or ac.empty: return None

        slat = float(bc["latitude"].iloc[-1]); slon = float(bc["longitude"].iloc[-1])
        elat = float(ac["latitude"].iloc[0]);  elon = float(ac["longitude"].iloc[0])

        merged = _merge_context_measurements(bc, ac, "native_clean", 60, None, None)
        if len(merged) < 2: return None

        meas = merged.apply(lambda r: along_cross_track_m(
            float(r["latitude"]), float(r["longitude"]), slat, slon, elat, elon),
            axis=1, result_type="expand").to_numpy(dtype=float)

        adsc_s = sort_ts(adsc_df)
        tau = np.array([float(np.clip((pd.Timestamp(ts) - t0).total_seconds() / dur, 0, 1))
                        for ts in adsc_s["timestamp"]], dtype=np.float32)
        if len(tau) == 0: return None

        flight = PreparedFlight(
            split="demo", segment_id=seg_id, t0=t0, t1=t1,
            start_lat=slat, start_lon=slon, end_lat=elat, end_lon=elon,
            measurement_times=[pd.Timestamp(ts) for ts in merged["timestamp"]],
            measurement_along_cross_m=meas,
            target_times=[pd.Timestamp(ts) for ts in adsc_s["timestamp"]],
            valid_indices=np.arange(len(tau), dtype=int),
            adsc_tau_full=tau.copy(),
            adsc_mask_full=np.ones(len(tau), dtype=np.float32),
            true_lat_full=adsc_s["latitude"].to_numpy(dtype=np.float32),
            true_lon_full=adsc_s["longitude"].to_numpy(dtype=np.float32),
            baseline_lat_full=np.array([great_circle_interpolate(slat, slon, elat, elon, float(x))[0]
                                         for x in tau], dtype=np.float32),
            baseline_lon_full=np.array([_wrap_lon_deg(great_circle_interpolate(slat, slon, elat, elon, float(x))[1])
                                         for x in tau], dtype=np.float32),
        )
        pl, po = kalman_smooth_gap(flight, kf_params)

        kf_wps = []
        for la, lo, ti in zip(pl, po, tau):
            ts = t0 + pd.Timedelta(seconds=float(ti * dur))
            kf_wps.append({"timestamp": ts.isoformat(), "latitude": round(float(la), 6),
                           "longitude": round(float(lo), 6), "source": "kalman"})

        def densify(wps, src):
            if not wps: return []
            pts = ([{"timestamp": t0.isoformat(), "latitude": slat, "longitude": slon}]
                   + wps
                   + [{"timestamp": t1.isoformat(), "latitude": elat, "longitude": elon}])
            out = []
            for i in range(len(pts) - 1):
                p0_, p1_ = pts[i], pts[i + 1]
                ts0 = pd.Timestamp(p0_["timestamp"]); ts1 = pd.Timestamp(p1_["timestamp"])
                sd = (ts1 - ts0).total_seconds()
                if sd <= 0: continue
                ns = max(2, int(sd / 60) + 1)
                for tj in np.linspace(0, 1, ns):
                    if i > 0 and tj == 0: continue
                    la_ = p0_["latitude"]  + (p1_["latitude"]  - p0_["latitude"])  * float(tj)
                    lo_ = p0_["longitude"] + (p1_["longitude"] - p0_["longitude"]) * float(tj)
                    out.append({"timestamp": (ts0 + pd.Timedelta(seconds=float(tj * sd))).isoformat(),
                                "latitude": round(la_, 6), "longitude": round(lo_, 6), "source": src})
            return out

        kf_dense = densify(kf_wps, "kalman_dense")
        kf_track = (bef.assign(source="adsb_before")[["timestamp","latitude","longitude","source"]].to_dict("records")
                    + kf_dense
                    + aft.assign(source="adsb_after")[["timestamp","latitude","longitude","source"]].to_dict("records"))

        kf_wps_df = pd.DataFrame(kf_wps)
        n = min(len(kf_wps_df), len(adsc_s))
        kf_errs = [
            hav(float(kf_wps_df["latitude"].iloc[i]),
                float(kf_wps_df["longitude"].iloc[i]),
                float(adsc_s["latitude"].iloc[i]),
                float(adsc_s["longitude"].iloc[i]))
            for i in range(n)
        ]
        kf_mean = sum(kf_errs) / len(kf_errs) if kf_errs else float("nan")
        return {"wps": kf_wps, "wp_df": kf_wps_df, "track": kf_track, "errs": kf_errs, "mean": kf_mean}
    except Exception:
        return None

kf_res  = build_kalman(segment_id, base_res, adsc_df)
kf_errs = kf_res["errs"] if kf_res else []
kf_mean = kf_res["mean"] if kf_res else float("nan")

gap_adsb = gru_res.get("gap_duration_minutes", gap_cat)

has_errs = math.isfinite(bl_mean) or math.isfinite(kf_mean) or math.isfinite(gru_mean)
max_err  = max(e for e in [bl_mean, kf_mean, gru_mean, 1.0] if math.isfinite(e))

errs_valid = [(n, e) for n, e in [("GRU", gru_mean), ("Kalman", kf_mean), ("Baseline", bl_mean)]
              if math.isfinite(e)]
if errs_valid:
    ranked    = sorted(errs_valid, key=lambda x: x[1])
    best_name = ranked[0][0]
    best_val  = ranked[0][1]
else:
    best_name, best_val = "-", float("nan")

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)

with m1:
    v_bl = f"{bl_mean:.1f} km" if math.isfinite(bl_mean) else "-"
    bar  = pct_bar(bl_mean if math.isfinite(bl_mean) else 0, max_err, "#ef4444") if has_errs else ""
    st.markdown(f"""
    <div class='metric-card mc-baseline'>
      <div class='metric-label'>Great-Circle Baseline</div>
      <div class='metric-value red'>{v_bl}</div>
      <div class='metric-sub'>mean error - this flight</div>
      {bar}
    </div>""", unsafe_allow_html=True)

with m2:
    v_kf  = f"{kf_mean:.1f} km" if math.isfinite(kf_mean) else "-"
    kf_imp = f"v{(1-kf_mean/bl_mean)*100:.0f}% vs baseline" if math.isfinite(kf_mean) and math.isfinite(bl_mean) and bl_mean > 0 else "physics model"
    bar = pct_bar(kf_mean if math.isfinite(kf_mean) else 0, max_err, "#3b82f6") if has_errs else ""
    st.markdown(f"""
    <div class='metric-card mc-kalman'>
      <div class='metric-label'>Kalman Filter</div>
      <div class='metric-value blue'>{v_kf}</div>
      <div class='metric-sub'>{kf_imp}</div>
      {bar}
    </div>""", unsafe_allow_html=True)

with m3:
    v_gru  = f"{gru_mean:.1f} km" if math.isfinite(gru_mean) else "-"
    gru_imp = f"v{(1-gru_mean/bl_mean)*100:.0f}% vs baseline" if math.isfinite(gru_mean) and math.isfinite(bl_mean) and bl_mean > 0 else "neural network"
    bar = pct_bar(gru_mean if math.isfinite(gru_mean) else 0, max_err, "#10b981") if has_errs else ""
    st.markdown(f"""
    <div class='metric-card mc-gru'>
      <div class='metric-label'>GRU Deep Learning</div>
      <div class='metric-value green'>{v_gru}</div>
      <div class='metric-sub'>{gru_imp}</div>
      {bar}
    </div>""", unsafe_allow_html=True)

with m4:
    imp_val  = f"{imp:+.1f}%" if math.isfinite(imp) else "-"
    imp_css  = "green" if math.isfinite(imp) and imp > 0 else ("red" if math.isfinite(imp) else "amber")
    best_color_map = {"GRU":"#34d399","Kalman":"#60a5fa","Baseline":"#f87171"}
    best_color = best_color_map.get(best_name, "#94a3b8")
    st.markdown(f"""
    <div class='metric-card mc-neutral'>
      <div class='metric-label'>GRU vs Baseline</div>
      <div class='metric-value {imp_css}'>{imp_val}</div>
      <div class='metric-sub'>
        Best: <span style='color:{best_color};font-weight:700;'>{best_name}</span>
        {f'- {best_val:.1f} km' if math.isfinite(best_val) else ''}
      </div>
    </div>""", unsafe_allow_html=True)

if metric_error:
    st.markdown(f"""<div class='info-box amber'>
    <b>Metric fallback:</b> per-flight error cards could not be computed.
    <span style='font-family:JetBrains Mono,monospace'>{clean(metric_error, "")}</span>
    </div>""", unsafe_allow_html=True)

if math.isfinite(kf_mean) and math.isfinite(gru_mean) and kf_mean < gru_mean:
    st.markdown("""<div class='info-box'>
    <b>Per-flight note:</b> Kalman outperforms GRU on this specific flight - common for short gaps
    or near-geodesic paths. GRU's advantage is statistical: strong median improvement across 240 test flights.
    </div>""", unsafe_allow_html=True)

if math.isfinite(bl_mean) and math.isfinite(kf_mean) and bl_mean < kf_mean:
    st.markdown("""<div class='info-box amber'>
    <b>Per-flight note:</b> Great-circle baseline wins here (~15% of flights follow near-geodesic paths).
    Test-set statistics confirm Kalman and GRU outperform baseline in aggregate.
    </div>""", unsafe_allow_html=True)

def build_map(gru_res, base_res, kf_res, adsc_df, bl_comp_df,
              show_baseline, show_kalman, show_truth,
              dep_code=None, arr_code=None):
    gru_track  = pd.DataFrame(gru_res["track"])
    base_dense = pd.DataFrame(gru_res.get("baseline_waypoints", []) or
                               base_res.get("baseline_waypoints", []))
    gru_wps    = pd.DataFrame(gru_res.get("gru_waypoints", []))
    kf_track   = pd.DataFrame(kf_res["track"]) if kf_res else pd.DataFrame()
    kf_wps     = pd.DataFrame(kf_res.get("wps", [])) if kf_res else pd.DataFrame()

    bef     = gru_track[gru_track["source"] == "adsb_before"].copy()
    aft     = gru_track[gru_track["source"] == "adsb_after"].copy()
    gru_gap = gru_track[gru_track["source"] == "gru_dense"].copy()
    kf_gap  = (kf_track[kf_track["source"] == "kalman_dense"].copy()
               if not kf_track.empty else pd.DataFrame())

    gap_lats, gap_lons = [], []
    for df in [gru_gap, kf_gap if not kf_gap.empty else pd.DataFrame()]:
        if not df.empty and "latitude" in df.columns:
            gap_lats += df["latitude"].dropna().tolist()
            gap_lons += df["longitude"].dropna().tolist()
    if not bef.empty:
        tail = bef.tail(max(5, len(bef)//1))
        gap_lats += tail["latitude"].tolist()
        gap_lons += tail["longitude"].tolist()
    if not aft.empty:
        head = aft.head(max(5, len(aft)//1))
        gap_lats += head["latitude"].tolist()
        gap_lons += head["longitude"].tolist()
    if not adsc_df.empty:
        gap_lats += adsc_df["latitude"].dropna().tolist()
        gap_lons += adsc_df["longitude"].dropna().tolist()

    if not gap_lats:
        for df in [gru_track]:
            if not df.empty and "latitude" in df.columns:
                gap_lats += df["latitude"].dropna().tolist()
                gap_lons += df["longitude"].dropna().tolist()

    if gap_lats and gap_lons:
        ctr_lat  = (max(gap_lats) + min(gap_lats)) / 2
        ctr_lon  = (max(gap_lons) + min(gap_lons)) / 2
        lat_span = max(gap_lats) - min(gap_lats)
        lon_span = max(gap_lons) - min(gap_lons)
        span     = max(lat_span * 1.4, lon_span * 0.7, 3.0)
        zoom     = max(2.5, min(6.5, 9.0 - math.log2(span + 1)))
    else:
        ctr_lat, ctr_lon, zoom = 52.0, -30.0, 3.0

    fig = go.Figure()

    ht_adsb    = "<b>ADS-B Observed</b><br>%{lat:.4f}° N, %{lon:.4f}°<extra></extra>"
    ht_bl      = "<b>Baseline (great circle)</b><br>%{lat:.4f}° N, %{lon:.4f}°<extra></extra>"
    ht_gru     = "<b>GRU Reconstruction</b><br>%{lat:.4f}° N, %{lon:.4f}°<extra></extra>"
    ht_kf      = "<b>Kalman Filter</b><br>%{lat:.4f}° N, %{lon:.4f}°<extra></extra>"
    ht_adsc    = "<b>ADS-C Ground Truth</b><br>%{lat:.4f}° N, %{lon:.4f}°<extra></extra>"
    ht_bound   = "<b>Gap boundary</b><br>%{lat:.4f}° N, %{lon:.4f}°<extra></extra>"

    def smap(lat, lon, mode, name, **kw):
        return go.Scattermapbox(lat=lat, lon=lon, mode=mode, name=name, **kw)

    _NA_AIRPORTS = {
        "JFK": (40.6413, -73.7781), "BOS": (42.3656, -71.0096),
        "EWR": (40.6895, -74.1745), "YYZ": (43.6777, -79.6248),
        "ORD": (41.9742, -87.9073), "IAD": (38.9531, -77.4565),
        "PHL": (39.8729, -75.2437), "CLT": (35.2144, -80.9473),
        "MIA": (25.7959, -80.2870), "MCO": (28.4312, -81.3081),
        "SFO": (37.6213,-122.3790), "ATL": (33.6407, -84.4277),
        "DFW": (32.8998, -97.0403), "IAH": (29.9902, -95.3368),
        "CUN": (21.0365, -86.8771),
    }
    _EU_AIRPORTS = {
        "LHR": (51.4700, -0.4543), "CDG": (49.0097,  2.5479),
        "DUB": (53.4213, -6.2701), "AMS": (52.3086,  4.7639),
        "FRA": (50.0379,  8.5622), "MAD": (40.4719, -3.5626),
        "LIS": (38.7813, -9.1359), "LGW": (51.1537, -0.1821),
        "ZRH": (47.4647,  8.5492), "IST": (41.2608, 28.7418),
        "DOH": (25.2608, 51.5656), "TLV": (32.0055, 34.8854),
        "DEL": (28.5562, 77.1000), "FCO": (41.8003, 12.2389),
        "BRU": (50.9010,  4.4844),
        "BCN": (41.2974,  2.0785),
    }

    _ALL_AIRPORTS = {**_NA_AIRPORTS, **_EU_AIRPORTS}
    _apt_legend_added = [False]

    def _draw_airport(apt_code, ep_lat, ep_lon):
        """Draw connector from known airport to the track endpoint."""
        coords = _ALL_AIRPORTS.get(apt_code)
        if not coords:
            return
        alat, alon = coords
        fig.add_trace(go.Scattermapbox(
            lat=[ep_lat, alat], lon=[ep_lon, alon],
            mode="lines",
            line=dict(width=1.5, color="rgba(148,163,184,0.35)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scattermapbox(
            lat=[alat], lon=[alon],
            mode="markers+text",
            marker=dict(size=11, color="rgba(148,163,184,0.55)", symbol="circle"),
            text=[apt_code], textposition="top center",
            textfont=dict(size=9, color="rgba(148,163,184,0.7)", family="JetBrains Mono"),
            showlegend=False, hoverinfo="skip",
        ))
        if not _apt_legend_added[0]:
            fig.add_trace(go.Scattermapbox(
                lat=[None], lon=[None], mode="lines",
                line=dict(width=1.5, color="rgba(148,163,184,0.35)"),
                name="Airport connection", showlegend=True,
            ))
            _apt_legend_added[0] = True

    if dep_code and not bef.empty:
        _draw_airport(dep_code, float(bef.iloc[0]["latitude"]), float(bef.iloc[0]["longitude"]))
    if arr_code and not aft.empty:
        _draw_airport(arr_code, float(aft.iloc[-1]["latitude"]), float(aft.iloc[-1]["longitude"]))

    if not bef.empty:
        fig.add_trace(smap(bef["latitude"].tolist(), bef["longitude"].tolist(),
            "lines", "ADS-B observed",
            line=dict(width=4, color="#e2e8f0"),
            hovertemplate=ht_adsb))
    if not aft.empty:
        fig.add_trace(smap(aft["latitude"].tolist(), aft["longitude"].tolist(),
            "lines", "_aft", showlegend=False,
            line=dict(width=4, color="#e2e8f0"),
            hovertemplate=ht_adsb))

    if show_baseline and not base_dense.empty:
        fig.add_trace(smap(base_dense["latitude"].tolist(), base_dense["longitude"].tolist(),
            "lines", "Great-circle baseline",
            line=dict(width=3, color="#f87171"),
            hovertemplate=ht_bl))

    if show_kalman and not kf_gap.empty:
        fig.add_trace(smap(kf_gap["latitude"].tolist(), kf_gap["longitude"].tolist(),
            "lines", "Kalman filter",
            line=dict(width=4, color="#60a5fa"),
            hovertemplate=ht_kf))
    if show_kalman and not kf_wps.empty:
        fig.add_trace(smap(kf_wps["latitude"].tolist(), kf_wps["longitude"].tolist(),
            "markers", "Kalman waypoints",
            marker=dict(size=10, color="#60a5fa", symbol="circle", opacity=1.0),
            hovertemplate=ht_kf))

    if not gru_gap.empty:
        fig.add_trace(smap(gru_gap["latitude"].tolist(), gru_gap["longitude"].tolist(),
            "lines", "GRU reconstruction",
            line=dict(width=6, color="#10b981"),
            hovertemplate=ht_gru))
    if not gru_wps.empty:
        fig.add_trace(smap(gru_wps["latitude"].tolist(), gru_wps["longitude"].tolist(),
            "markers", "GRU waypoints",
            marker=dict(size=14, color="#10b981", symbol="circle", opacity=1.0),
            hovertemplate=ht_gru))

    if show_truth and not adsc_df.empty:
        fig.add_trace(smap(adsc_df["latitude"].tolist(), adsc_df["longitude"].tolist(),
            "markers", "ADS-C ground truth",
            marker=dict(size=16, color="#fbbf24", symbol="circle", opacity=1.0),
            hovertemplate=ht_adsc))

    for df_, show in [(bef, True), (aft, False)]:
        if df_.empty: continue
        pt = df_.iloc[-1] if df_ is bef else df_.iloc[0]
        fig.add_trace(smap([pt["latitude"]], [pt["longitude"]],
            "markers", "Gap boundary" if show else "_gb", showlegend=show,
            marker=dict(size=18, color="#60a5fa",
                        symbol="circle", opacity=0.6),
            hovertemplate=ht_bound))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=ctr_lat, lon=ctr_lon),
            zoom=zoom,
        ),
        paper_bgcolor="#0a0f1e",
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(10,15,30,0.9)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            font=dict(color="#cbd5e1", size=11, family="JetBrains Mono"),
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="rgba(255,255,255,0.15)",
            font=dict(family="JetBrains Mono", size=11, color="#f8fafc"),
        ),
    )
    return fig

st.markdown("<div class='map-wrap'>", unsafe_allow_html=True)
fig = build_map(gru_res, base_res, kf_res, adsc_df, bl_comp_df,
                show_baseline, show_kalman, show_truth,
                dep_code=dep if dep != "N/A" else None,
                arr_code=arr if arr != "N/A" else None)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": True})
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class='legend-strip'>
  <span><span class='lsym' style='color:#94a3b8;font-weight:700;'>━━</span> ADS-B observed</span>
  <span><span class='lsym' style='color:#10b981;font-weight:700;'>━━</span> GRU reconstruction</span>
  <span><span class='lsym' style='color:#60a5fa;font-weight:700;'>━━</span> Kalman filter</span>
  <span><span class='lsym' style='color:#f87171;font-weight:700;'>━━</span> Great-circle baseline</span>
  <span><span class='lsym' style='color:#fbbf24;font-weight:700;'>●</span> ADS-C ground truth</span>
  <span><span class='lsym' style='color:#60a5fa;font-weight:700;'>○</span> Gap boundary</span>
  <span><span class='lsym' style='color:#94a3b8;'>──</span> Est. airport connection</span>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
cl, cr = st.columns(2)

CHART_LAYOUT = dict(
    paper_bgcolor="#0a0f1e", plot_bgcolor="#0f172a",
    font=dict(color="#cbd5e1", family="Inter"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False,
               tickfont=dict(family="JetBrains Mono", size=10, color="#64748b"),
               title_font=dict(color="#64748b")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False,
               tickfont=dict(family="JetBrains Mono", size=10, color="#64748b"),
               title_font=dict(color="#64748b")),
    legend=dict(bgcolor="rgba(10,15,30,0.8)", bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1, font=dict(color="#94a3b8", size=10)),
    margin=dict(l=55, r=15, t=15, b=50),
    height=300,
)

with cl:
    st.markdown('<div class="sec-head">POINTWISE ERROR PROFILE - THIS FLIGHT</div>',
                unsafe_allow_html=True)
    if gru_errs and bl_errs:
        n = min(len(gru_errs), len(bl_errs))
        if kf_errs: n = min(n, len(kf_errs))
        ef = go.Figure()
        ef.add_trace(go.Scatter(
            x=list(range(1, n+1)), y=bl_errs[:n], mode="lines+markers",
            name="Baseline", line=dict(color="#f87171", width=2, dash="dot"),
            marker=dict(size=4, color="#f87171"),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.05)"))
        if kf_errs:
            ef.add_trace(go.Scatter(
                x=list(range(1, n+1)), y=kf_errs[:n], mode="lines+markers",
                name="Kalman", line=dict(color="#60a5fa", width=2.2),
                marker=dict(size=4, color="#60a5fa")))
        ef.add_trace(go.Scatter(
            x=list(range(1, n+1)), y=gru_errs[:n], mode="lines+markers",
            name="GRU", line=dict(color="#10b981", width=3),
            marker=dict(size=6, color="#10b981"),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.07)"))
        ef.update_layout(**{
            **CHART_LAYOUT,
            "xaxis": {**CHART_LAYOUT["xaxis"], "title": "ADS-C waypoint #"},
            "yaxis": {**CHART_LAYOUT["yaxis"], "title": "Error (km)"},
        })
        st.plotly_chart(ef, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown("""
        <div style='background:#0f172a;border:1px solid rgba(255,255,255,0.06);border-radius:8px;
                    height:280px;display:flex;align-items:center;justify-content:center;'>
          <span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#475569;'>
            ADS-C ground truth unavailable for this flight</span>
        </div>""", unsafe_allow_html=True)

with cr:
    st.markdown('<div class="sec-head">METHOD COMPARISON - TEST SET vs THIS FLIGHT</div>',
                unsafe_allow_html=True)
    methods = ["Baseline", "Kalman", "GRU"]
    medians = [model_stats["bl_median"], model_stats["kf_median"], model_stats["gru_median"]]
    colors  = ["#f87171", "#60a5fa", "#10b981"]
    compare_vals = medians + [v for v in [bl_mean, kf_mean, gru_mean] if math.isfinite(v)]
    compare_max = max(compare_vals) if compare_vals else max(medians)

    bf = go.Figure()
    bf.add_trace(go.Bar(
        x=methods, y=medians,
        marker_color=["rgba(248,113,113,0.8)","rgba(96,165,250,0.8)","rgba(16,185,129,0.8)"],
        marker_line_color=colors, marker_line_width=1.5,
        width=0.42,
        text=[f"{v:.0f} km" for v in medians], textposition="outside",
        textfont=dict(color="#94a3b8", family="JetBrains Mono", size=11),
        name="Test-set median (240 flights)"))

    for method, value, color in zip(methods, [bl_mean, kf_mean, gru_mean], colors):
        if math.isfinite(value):
            bf.add_trace(go.Scatter(
                x=[method], y=[value], mode="markers",
                marker=dict(size=16, color="#fbbf24", symbol="diamond",
                            line=dict(color=color, width=2.5)),
                name=f"This flight"))
            bf.add_annotation(
                x=method, y=value, text=f"{value:.0f}",
                showarrow=False, yshift=18,
                font=dict(color="#fbbf24", family="JetBrains Mono", size=11, weight=700),
                bgcolor="rgba(10,15,30,0.85)",
                bordercolor=color, borderwidth=1, borderpad=3)

    bf.update_layout(**{
        **CHART_LAYOUT,
        "yaxis": {**CHART_LAYOUT["yaxis"],
                  "title": "Median error (km)",
                  "range": [0, compare_max * 1.22]},
        "xaxis": {**CHART_LAYOUT["xaxis"],
                  "tickfont": dict(family="JetBrains Mono", size=12, color="#94a3b8")},
        "margin": dict(l=65, r=15, t=48, b=50),
        "annotations": [dict(x=.5, y=1.07, xref="paper", yref="paper", showarrow=False,
            text="Bars = 240-flight test set  -  ◆ = this flight",
            font=dict(size=9, color="#94a3b8", family="JetBrains Mono"))],
    })
    st.plotly_chart(bf, use_container_width=True, config={"displayModeBar": False})

st.markdown('<div class="sec-head" style="margin-top:20px;">HOW IT WORKS - ML PIPELINE</div>',
            unsafe_allow_html=True)

p1, p2, p3, p4 = st.columns(4)

with p1:
    st.markdown("""
    <div class='pipeline-card'>
      <div class='pipeline-step'>Step 1 - Data</div>
      <div class='pipeline-title'>ADS-B + ADS-C Fusion</div>
      <div class='pipeline-desc'>
        ADS-B gives high-freq position fixes before &amp; after oceanic entry.
        ADS-C provides sparse controller waypoints over the ocean - our ground truth.
        1,704 NAT crossings from OpenSky Trino (2023-2025).
      </div>
    </div>""", unsafe_allow_html=True)

with p2:
    st.markdown("""
    <div class='pipeline-card'>
      <div class='pipeline-step'>Step 2 - Baseline</div>
      <div class='pipeline-title'>Great-Circle Interpolation</div>
      <div class='pipeline-desc'>
        Shortest path on a sphere between two anchor points.
        Simple, parameter-free, and surprisingly competitive.
        Median error: <span style='color:#f87171;font-weight:700;'>131 km</span>.
        Serves as the lower-bound for ML improvement.
      </div>
    </div>""", unsafe_allow_html=True)

with p3:
    st.markdown("""
    <div class='pipeline-card'>
      <div class='pipeline-step'>Step 3 - Physics</div>
      <div class='pipeline-title'>Kalman Filter (RTS)</div>
      <div class='pipeline-desc'>
        Constant-velocity kinematic model in along/cross-track coordinates.
        ADS-B context measurements update the state estimate.
        RTS smoother refines with future observations.
        Median: <span style='color:#60a5fa;font-weight:700;'>83 km</span>.
      </div>
    </div>""", unsafe_allow_html=True)

with p4:
    st.markdown(f"""
    <div class='pipeline-card' style='border-color:rgba(16,185,129,0.25);'>
      <div class='pipeline-step' style='color:#34d399;'>Step 4 - Deep Learning</div>
      <div class='pipeline-title' style='color:#34d399;'>Bidirectional GRU</div>
      <div class='pipeline-desc'>
        693K-param BiGRU encodes 64-step ADS-B context (before &amp; after).
        Predicts waypoint offsets relative to great-circle baseline.
        Trained on 1,094 flights. Median: <span style='color:#34d399;font-weight:700;'>{model_stats['gru_median']:.0f} km</span>
        &nbsp;(v{(1-model_stats['gru_median']/model_stats['bl_median'])*100:.0f}% vs baseline).
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="sec-head" style="margin-top:20px;">GLOBAL TEST-SET PERFORMANCE SUMMARY</div>',
            unsafe_allow_html=True)

imp_gru_vs_bl = (1 - model_stats["gru_median"] / model_stats["bl_median"]) * 100
imp_kf_vs_bl  = (1 - model_stats["kf_median"]  / model_stats["bl_median"]) * 100
imp_gru_vs_kf = (1 - model_stats["gru_median"] / model_stats["kf_median"]) * 100

st.markdown(f"""
<table class='results-table'>
  <thead>
    <tr>
      <th>Rank</th><th>Method</th><th>Median Error</th>
      <th>vs Baseline</th><th>Architecture</th><th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class='td-rank1'>① BEST</td>
      <td class='td-rank1'>GRU Neural Network</td>
      <td class='td-rank1'>{model_stats['gru_median']:.1f} km</td>
      <td class='td-rank1'>−{imp_gru_vs_bl:.1f}%</td>
      <td>Bidirectional GRU - 693K params - 2 layers</td>
      <td>ADS-B context encoder + residual decoder</td>
    </tr>
    <tr>
      <td class='td-rank2'>② GOOD</td>
      <td class='td-rank2'>Kalman Filter</td>
      <td class='td-rank2'>{model_stats['kf_median']:.1f} km</td>
      <td class='td-rank2'>−{imp_kf_vs_bl:.1f}%</td>
      <td>Constant-velocity kinematic - RTS smoother</td>
      <td>ADS-B boundary observations as measurements</td>
    </tr>
    <tr>
      <td class='td-rank3'>③ BASE</td>
      <td class='td-rank3'>Great-Circle Baseline</td>
      <td class='td-rank3'>{model_stats['bl_median']:.1f} km</td>
      <td style='color:#475569;'>-</td>
      <td>Spherical geodesic interpolation (SLERP)</td>
      <td>Reference: assumes straight-line path</td>
    </tr>
  </tbody>
</table>
<div style='font-family:JetBrains Mono,monospace;font-size:8px;color:#475569;
            margin-top:8px;padding:4px 0;letter-spacing:1px;'>
  240 test flights - 2023-2025 - Shanwick / Gander OANC / North Atlantic Tracks
  - USJ Lebanon - AeroEngineering - BiGRU achieves {imp_gru_vs_bl:.0f}% reduction in median error vs baseline
</div>""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;padding:16px 0 6px;font-size:8px;color:#334155;
            font-family:JetBrains Mono,monospace;letter-spacing:2.5px;margin-top:20px;
            border-top:1px solid rgba(255,255,255,0.05);'>
  AEROFUSION - ADS-B + ADS-C SENSOR FUSION - SHANWICK / NORTH ATLANTIC OCEAN TRACKS - USJ LEBANON - AEROENGINEERING 2025
</div>""", unsafe_allow_html=True)

tab_hist.__exit__(None, None, None)

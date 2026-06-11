"""
dashboard.py — SmartWatch Health Analytics
Run: streamlit run dashboard.py
"""

import os, pickle, datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Auth ──────────────────────────────────────────────────────────────────────
from auth.db    import init_db
from auth.pages import show_auth_page

st.set_page_config(
    page_title="Circadian Insight System",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — light + dark mode
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

html,body,.stApp{font-family:'DM Sans',sans-serif !important;}

/* LIGHT */
@media (prefers-color-scheme:light){
  .stApp{
    background:#f8fafc !important;
    color:#0f172a !important;
  }

/* SIDEBAR — UNIVERSAL FIX */
/* SIDEBAR BACKGROUND */
section[data-testid="stSidebar"]{
  background: linear-gradient(160deg,#4f46e5,#7c3aed) !important;
}

/* TEXT */
section[data-testid="stSidebar"] *{
  color: #ffffff !important;
}

/* TITLE */
section[data-testid="stSidebar"] h2{
  color: #ffffff !important;
  font-weight: 700 !important;
}

/* BUTTONS */
section[data-testid="stSidebar"] .stButton > button{
  background: rgba(0,0,0,0.25) !important;   /* 🔥 FIXED */
  border: 1px solid rgba(255,255,255,0.25) !important;
  color: #ffffff !important;
  border-radius: 12px !important;
  padding: 12px 16px !important;
  font-weight: 500 !important;
}

/* HOVER */
section[data-testid="stSidebar"] .stButton > button:hover{
  background: rgba(0,0,0,0.40) !important;
}

/* ACTIVE BUTTON */


/* DATE INPUT */
section[data-testid="stSidebar"] input{
  background: #ffffff !important;
  color: #111827 !important;
}

  [data-testid="stMetric"]{
    background:#ffffff !important;
    border:1px solid #e2e8f0 !important;
    border-radius:14px !important;
    padding:18px !important;
    box-shadow:0 4px 12px rgba(0,0,0,0.06) !important;
  }

  h1,h2,h3{
    color:#0f172a !important;
    font-weight:600 !important;
  }

  .stMarkdown, .stText{
    color:#1e293b !important;
  }

  /* Fix Plotly text visibility */
  .js-plotly-plot .plotly .main-svg {
    background: transparent !important;
  }
}

/* DARK */
@media (prefers-color-scheme:dark){
  .stApp{background:#0b0f1a !important;color:#e2e8f0 !important}
  section[data-testid="stSidebar"]{
    background:#10172a !important;
    border-right:1px solid #1e2a42 !important}
  section[data-testid="stSidebar"] *{color:#e2e8f0 !important}
  section[data-testid="stSidebar"] .stButton>button{
    background:rgba(99,102,241,0.08) !important;
    border:1px solid rgba(99,102,241,0.25) !important;color:#c7d2fe !important}
  section[data-testid="stSidebar"] .stButton>button:hover{
    background:rgba(99,102,241,0.22) !important;color:#fff !important}
  [data-testid="stMetric"]{background:#141d32 !important;
    border:1px solid #1e2a42 !important;border-radius:16px !important;
    padding:20px 24px !important}
  [data-testid="stMetricLabel"]{color:#64748b !important}
  [data-testid="stMetricValue"]{color:#e2e8f0 !important}
  h1,h2,h3{color:#f1f5f9 !important}
  hr{border-color:#1e2a42 !important}
  .tag-normal{background:#052e16;border:1px solid #22c55e;color:#4ade80}
  .tag-abnormal{background:#450a0a;border:1px solid #ef4444;color:#fca5a5}
  .tag-note{background:#0c1a3a;border:1px solid #60a5fa;color:#93c5fd}
  .tag-warn{background:#422006;border:1px solid #f59e0b;color:#fcd34d}
  .result-row{border-bottom:1px solid #1e2a42}
  .sidebar-user{background:rgba(99,102,241,0.10);border:1px solid rgba(99,102,241,0.25)}
}

/* SHARED */
section[data-testid="stSidebar"] .stButton>button{
  width:100% !important;border-radius:10px !important;
  padding:12px 16px !important;margin-bottom:8px !important;
  font-size:14px !important;font-weight:500 !important;
  text-align:left !important;transition:all .2s !important}
[data-testid="stMetricLabel"]{
  font-size:11px !important;font-weight:600 !important;
  letter-spacing:.08em !important;text-transform:uppercase !important}
[data-testid="stMetricValue"]{font-size:26px !important;font-weight:700 !important}
.block-container{padding:2rem 2.5rem !important;max-width:1400px !important}
h1{font-size:1.8rem !important;font-weight:700 !important}
.stAlert{border-radius:12px !important;font-weight:500 !important}

.tag-normal,.tag-abnormal,.tag-note,.tag-warn{
  display:inline-block;border-radius:6px;
  padding:2px 10px;font-size:12px;font-weight:600;margin:2px}

.result-row{
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 0}
/* Calendar popup fix */
div[data-baseweb="calendar"] * {
  color:#0f172a !important;
}

div[role="dialog"] * {
  color:#0f172a !important;
}
            
.sidebar-user{
  border-radius:12px;padding:12px 14px;margin-bottom:12px}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# AUTH GATE
# ══════════════════════════════════════════════════════════════════════════════
init_db()

if not st.session_state.get("authenticated"):
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        show_auth_page()
    st.stop()

user = st.session_state["user"]

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR          = "data"
FINAL_DATA_PATH   = f"{DATA_DIR}/final_data.csv"
SLEEP_DATA_PATH   = f"{DATA_DIR}/sleep_data.csv"
MANUAL_LOG_PATH   = f"{DATA_DIR}/manual_log.csv"
PROFILE_PATH      = f"{DATA_DIR}/user_profile.csv"
MODEL_PATH        = f"{DATA_DIR}/autoencoder_model.keras"
SCALER_PATH       = f"{DATA_DIR}/scaler.pkl"
FEATURE_COLS_PATH = f"{DATA_DIR}/feature_cols.txt"
THRESHOLD_PATH    = f"{DATA_DIR}/anomaly_threshold.txt"

PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="DM Sans",
        color="#0f172a",   # 👈 CHANGE THIS (dark text for light mode)
        size=12
    ),
    # margin=dict(l=16,r=16,t=36,b=16), hovermode="x unified",
)

# Reference ranges for manual entry assessment
REFS = {
    "HeartRate":                (40,  60,  100, 140),
    "HeartRateVariabilitySDNN": (5,   20,  100, 200),
    "RestingHeartRate":         (30,  50,   90, 110),
    "StepCount":                (0,  4000,20000,40000),
    "ActiveEnergyBurned":       (0,   100, 800, 1500),
    "BasalEnergyBurned":        (800,1200,2200, 3000),
    "PhysicalEffort":           (0.0, 0.1,  0.8,  1.0),
    "WalkingSpeed":             (0.0, 2.0,  7.0, 10.0),
    "RespiratoryRate":          (6,   12,   20,   30),
    "sleep_hours":              (0.0, 6.0,  9.0, 12.0),
}

def assess(key, val):
    if key not in REFS: return "—", "tag-note"
    lw,nl,nh,hw = REFS[key]
    if   val < lw:  return "Very Low ⚠️",  "tag-abnormal"
    elif val < nl:  return "Below Range",   "tag-warn"
    elif val <= nh: return "Normal ✅",     "tag-normal"
    elif val <= hw: return "Above Range",   "tag-warn"
    else:           return "Very High ⚠️", "tag-abnormal"

def hex_to_rgba(h, alpha=0.12):
    h = h.strip().lstrip("#")
    if len(h)!=6: return f"rgba(99,102,241,{alpha})"
    try: return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"
    except: return f"rgba(99,102,241,{alpha})"

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_final_data():
    if not os.path.exists(FINAL_DATA_PATH):
        st.error("❌ `final_data.csv` not found. Run `python main.py` first.")
        st.stop()
    df = pd.read_csv(FINAL_DATA_PATH)
    df["startdate"] = pd.to_datetime(df["startdate"], errors="coerce")
    df.dropna(subset=["startdate"], inplace=True)
    df["date"] = df["startdate"].dt.date
    return df

@st.cache_data(ttl=300)
def load_sleep_data():
    if not os.path.exists(SLEEP_DATA_PATH):
        return pd.DataFrame(columns=["date","sleep_hours"])
    df = pd.read_csv(SLEEP_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df

@st.cache_resource
def load_artefacts():
    for p in [MODEL_PATH,SCALER_PATH,FEATURE_COLS_PATH,THRESHOLD_PATH]:
        if not os.path.exists(p): return None,None,None,None
    try:
        try: from tensorflow.keras.models import load_model
        except ImportError: from keras.models import load_model
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH,"rb") as f: scaler = pickle.load(f)
        with open(FEATURE_COLS_PATH) as f: fcols=[l.strip() for l in f if l.strip()]
        with open(THRESHOLD_PATH) as f: thresh=float(f.read().strip())
        return model,scaler,fcols,thresh
    except: return None,None,None,None

def load_manual_log():
    if not os.path.exists(MANUAL_LOG_PATH): return pd.DataFrame()
    df = pd.read_csv(MANUAL_LOG_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df

def save_manual_log(df): df.to_csv(MANUAL_LOG_PATH, index=False)

def load_profile():
    if os.path.exists(PROFILE_PATH):
        df = pd.read_csv(PROFILE_PATH)
        if not df.empty: return df.iloc[0].to_dict()
    return {"age":25,"occupation":"Other"}

def save_profile(age, occ):
    pd.DataFrame([{"age":age,"occupation":occ}]).to_csv(PROFILE_PATH, index=False)

def get_sleep_for_date(date, sleep_df):
    row = sleep_df[sleep_df["date"]==date]
    if not row.empty: return round(float(row["sleep_hours"].values[0]),2)
    log = load_manual_log()
    if not log.empty and "date" in log.columns and "sleep_hours" in log.columns:
        r2 = log[log["date"]==date]
        if not r2.empty:
            v = r2["sleep_hours"].values[0]
            if not pd.isna(v): return round(float(v),2)
    return 0.0

# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def line_chart(df, x, y, color="#2563eb", y_label=""):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x],
        y=df[y],
        mode="lines",
        line=dict(
            color=color,
            width=2.5   # 👈 thicker like anomaly
        ),
        fill="tozeroy",
        fillcolor=f"rgba(37,99,235,0.08)",  # soft fill
        hovertemplate=f"{y_label}: %{{y:.2f}}<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(
            title="Time",
            showgrid=False,
            color="#334155"
        ),
        yaxis=dict(
            title=y_label,
            showgrid=True,
            gridcolor="#e2e8f0",
            color="#334155"
        )
    )

    return fig

def anomaly_chart(df, threshold=None):
    fig = go.Figure()

    if df.empty or "reconstruction_error" not in df.columns:
        return fig

    x = df["startdate"]
    y = df["reconstruction_error"]

    # Continuous base line (always connected)
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(width=2),
        name="Reconstruction Error",
        hovertemplate="Error: %{y:.6f}<extra></extra>",
    ))

    # Overlay anomaly points (instead of breaking line)
    anomalies = df[df["anomaly"] == 1]

    fig.add_trace(go.Scatter(
        x=anomalies["startdate"],
        y=anomalies["reconstruction_error"],
        mode="markers",
        marker=dict(size=6, color="red"),
        name="Anomaly",
    ))

    if threshold:
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            annotation_text=f"Threshold ({threshold:.5f})",
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(title="Time"),
        yaxis=dict(title="Reconstruction Error"),
        title_font=dict(size=14, color="#0f172a"),
    )
    fig.update_layout(
    title=dict(
        text="Reconstruction Error — Anomaly Detection",
        font=dict(size=16, color="#0f172a"),
        x=0
    )
)

    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE + DATA LOAD
# ══════════════════════════════════════════════════════════════════════════════
if "page" not in st.session_state: st.session_state.page = "Dashboard"

df       = load_final_data()
sleep_df = load_sleep_data()

watch_dates = sorted(df["date"].unique())
min_date    = min(watch_dates)
max_date    = max(watch_dates)

if ("selected_date" not in st.session_state or
        st.session_state.selected_date not in watch_dates):
    st.session_state.selected_date = max_date

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # User card
    st.markdown(f"""
    <div class="sidebar-user">
        <div style="font-size:1.2rem;margin-bottom:2px">👤</div>
        <div style="font-weight:600;font-size:.95rem">
            {user['display_name']}
        </div>
        <div style="font-size:.75rem;opacity:.7">@{user['username']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h2 style="
        font-weight:700;
        font-size:20px;
        margin-bottom:10px;
        color:white;">
                Circadian Insight System
    </h2>
    """, unsafe_allow_html=True)
    for label, icon in [("Dashboard","🏠"),("Analysis","📊"),
                         ("Summary","📋"),("Manual Entry","✏️"),("Settings","⚙️")]:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label

    st.markdown("---")

    # Date picker — clamped to watch data range only (no date errors)
    if st.session_state.page in ("Dashboard","Analysis","Summary"):
        st.session_state.selected_date = st.date_input(
            "📅 Select Date",
            value=st.session_state.selected_date,
            min_value=min_date,
            max_value=max_date,
        )
        st.markdown("---")

    if st.button("🚪  Logout", use_container_width=True):
        st.session_state.clear(); st.rerun()

page          = st.session_state.page
selected_date = st.session_state.selected_date

# ══════════════════════════════════════════════════════════════════════════════
# WATCH DATA — shared filter used by Dashboard / Analysis / Summary
# ══════════════════════════════════════════════════════════════════════════════
if page in ("Dashboard","Analysis","Summary"):
    full_day_df = df[df["date"]==selected_date].copy().reset_index(drop=True)
    filtered_df = (full_day_df.iloc[::len(full_day_df)//1000].reset_index(drop=True)
                   if len(full_day_df)>1000 else full_day_df)

    total_sleep   = get_sleep_for_date(selected_date, sleep_df)
    anomaly_count = int(full_day_df["anomaly"].sum())             if "anomaly"       in full_day_df.columns else 0
    avg_stress    = round(full_day_df["stress_score"].mean(), 2)  if "stress_score"  in full_day_df.columns else 0.0
    total_steps   = int(full_day_df["StepCount"].sum())           if "StepCount"     in full_day_df.columns else 0
    avg_hr        = round(full_day_df["HeartRate"].mean(), 1)     if "HeartRate"     in full_day_df.columns else 0.0

    # Load threshold for anomaly chart
    thresh = None
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f: thresh = float(f.read().strip())

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title(f"Welcome, {user['display_name'].split()[0]} 👋")
    st.caption(f"Showing watch data for **{selected_date}**")

    # Sleep gate
    if total_sleep == 0:
        st.info("💤 No sleep data for this date. Enter it manually below.")
        time_opts = ([f"{h:02d}:{m:02d}" for h in range(18,24) for m in (0,15,30,45)] +
                     [f"{h:02d}:{m:02d}" for h in range(0,13)  for m in (0,15,30,45)])
        end_opts  = [f"{h:02d}:{m:02d}" for h in range(4,13)  for m in (0,15,30,45)]
        ca, cb = st.columns(2)
        with ca: s_start = st.selectbox("Sleep Start Time", time_opts, index=8)
        with cb: s_end   = st.selectbox("Wake-up Time",     end_opts,  index=12)
        if st.button("✅ Confirm Sleep", type="primary"):
            hrs = (pd.to_datetime(s_end)-pd.to_datetime(s_start)).total_seconds()/3600
            if hrs<=0: hrs+=24
            log = load_manual_log()
            if not log.empty and "date" in log.columns:
                log = log[log["date"]!=selected_date]
            log = pd.concat([log, pd.DataFrame([{
                "date":selected_date,"sleep_hours":round(hrs,2),"is_manual":0}])],
                ignore_index=True)
            save_manual_log(log)
            st.success(f"Saved {round(hrs,2)} hrs."); st.rerun()
        st.stop()

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("💤 Total Sleep",  f"{total_sleep} hrs")
    c2.metric("🚶 Total Steps",  f"{total_steps:,}")
    c3.metric("🧠 Avg Stress",   f"{avg_stress}")
    c4.metric("⚠️ Anomalies",    f"{anomaly_count}")

    st.markdown("---")

    # Anomaly detection chart
    st.subheader("Anomaly Detection")
    if "reconstruction_error" in filtered_df.columns:
        st.plotly_chart(anomaly_chart(filtered_df, thresh),
                        use_container_width=True, config={"displayModeBar":False})

    # Stress score chart
    if "stress_score" in filtered_df.columns:
        st.subheader("Stress Score")
        st.plotly_chart(line_chart(filtered_df,"startdate","stress_score",
                                   "#f59e0b","Stress Score"),
                        use_container_width=True, config={"displayModeBar":False})

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analysis":
    st.title("Health Analysis")
    st.caption(f"Showing data for **{selected_date}**")

    # Reconstruction error at top
    if "reconstruction_error" in filtered_df.columns:
        st.subheader("⚠️ Reconstruction Error")
        st.plotly_chart(anomaly_chart(filtered_df, thresh),
                        use_container_width=True, config={"displayModeBar":False})

    for col, label, color, ylabel in [
        ("ActiveEnergyBurned",       "🔥 Active Energy Burned",  "#f97316", "kcal"),
        ("StepCount",                "🚶 Step Count",            "#22c55e", "Steps"),
        ("HeartRate",                "❤️ Heart Rate",            "#ef4444", "BPM"),
        ("HeartRateVariabilitySDNN", "📈 HRV (SDNN)",            "#a78bfa", "ms"),
        ("recovery_index",           "🔄 Recovery Index",        "#38bdf8", "Index [0–1]"),
    ]:
        if col in filtered_df.columns:
            st.subheader(label)
            st.plotly_chart(line_chart(filtered_df,"startdate",col,color,ylabel),
                            use_container_width=True, config={"displayModeBar":False})

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Summary":
    st.title("Daily Summary")
    st.caption(f"Health report for **{selected_date}**")

    total_energy  = round(full_day_df["ActiveEnergyBurned"].sum(),1)     if "ActiveEnergyBurned" in full_day_df.columns else 0.0
    sedentary_pct = round(100*full_day_df["sedentary_index"].mean(),1)   if "sedentary_index"    in full_day_df.columns else 0.0

    if   anomaly_count == 0: st.success("✅ No anomalies detected. Healthy day!")
    elif anomaly_count < 10: st.warning(f"⚠️ {anomaly_count} anomalies detected.")
    else:                    st.error(f"🚨 {anomaly_count} significant anomalies detected.")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📋 Daily Metrics")
        st.dataframe(pd.DataFrame({
            "Metric": ["💤 Sleep","❤️ Avg HR","🧠 Stress",
                       "🔥 Energy","🚶 Steps","🛋️ Sedentary","⚠️ Anomalies"],
            "Value":  [f"{total_sleep} hrs", f"{avg_hr} bpm", f"{avg_stress}",
                       f"{total_energy} kcal", f"{total_steps:,}",
                       f"{sedentary_pct}%", f"{anomaly_count}"],
        }), use_container_width=True, hide_index=True)

    with c2:
        st.subheader("💡 Insights")
        for ins in [
            "😴 Sleep below 6 hrs — aim for 7–9." if total_sleep < 6 else "✅ Good sleep duration.",
            "⚠️ High stress — consider relaxation." if avg_stress > 5 else "✅ Stress levels controlled.",
            "❤️ Elevated heart rate." if avg_hr > 100 else "✅ Heart rate within normal range.",
            "🛋️ High sedentary time." if sedentary_pct > 70 else "✅ Good activity distribution.",
            "🚶 Low steps — aim for 8,000+." if total_steps < 5000 else "✅ Excellent step count!",
        ]: st.markdown(f"- {ins}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MANUAL ENTRY  — no date needed, instant analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Manual Entry":
    st.title("✏️ Manual Health Check")
    st.caption("Enter your health values to get an instant normal/abnormal assessment.")
    st.info("💡 Enter approximate values — rough estimates still give useful insights.")

    st.markdown("---")
    st.subheader("❤️ Heart Metrics")
    c1,c2,c3 = st.columns(3)
    with c1: hr  = st.number_input("Heart Rate (bpm)",          30,220,72,  help="Normal resting: 60–100 bpm")
    with c2: hrv = st.number_input("HRV — SDNN (ms)",            1,300,45,  help="Higher = better recovery. Typical: 20–100 ms")
    with c3: rhr = st.number_input("Resting Heart Rate (bpm)",  30,120,65,  help="Normal: 50–90 bpm")

    st.subheader("🚶 Activity")
    c4,c5,c6 = st.columns(3)
    with c4: steps    = st.number_input("Step Count",               0,100000,5000,100, help="Target: 8,000+")
    with c5: active_e = st.number_input("Active Energy (kcal)",     0,3000,300,        help="Typical: 200–600 kcal")
    with c6: basal_e  = st.number_input("Basal Energy (kcal)",    500,4000,1800,       help="Typical: 1400–2000 kcal")

    st.subheader("🏃 Performance & Vitals")
    c7,c8,c9,c10 = st.columns(4)
    with c7:  effort   = st.slider("Physical Effort (0–1)",  0.0,1.0,0.3,0.05, help="0=rest, 1=max effort")
    with c8:  walk_spd = st.number_input("Walking Speed (km/h)",   0.0,15.0,4.5,0.1)
    with c9:  resp     = st.number_input("Respiratory Rate (br/min)", 6,50,15, help="Normal: 12–20 br/min")
    with c10: sleep_h  = st.number_input("Sleep (hrs)",             0.0,14.0,7.0,0.25, help="Recommended: 7–9 hrs")

    st.markdown("---")

    if st.button("🔍 Analyse My Values", type="primary", use_container_width=True):
        raw = {
            "HeartRate":                float(hr),
            "HeartRateVariabilitySDNN": float(hrv),
            "RestingHeartRate":         float(rhr),
            "StepCount":                float(steps),
            "ActiveEnergyBurned":       float(active_e),
            "BasalEnergyBurned":        float(basal_e),
            "PhysicalEffort":           float(effort),
            "WalkingSpeed":             float(walk_spd),
            "RespiratoryRate":          float(resp),
            "sleep_hours":              float(sleep_h),
        }
        _hr  = max(raw["HeartRate"],1)
        _hrv = max(raw["HeartRateVariabilitySDNN"],1)
        stress   = round(min(_hr/_hrv, 20), 2)
        recovery = round(min(raw["ActiveEnergyBurned"]/(_hr+1)/10, 1.0), 2)

        st.markdown("---")
        st.subheader("📊 Analysis Results")

        # Per-metric table
        check_keys = [
            ("HeartRate",                "❤️ Heart Rate",          f"{hr:.0f} bpm"),
            ("HeartRateVariabilitySDNN", "📈 HRV",                 f"{hrv:.0f} ms"),
            ("RestingHeartRate",         "💤 Resting HR",          f"{rhr:.0f} bpm"),
            ("StepCount",               "🚶 Steps",               f"{steps:,.0f}"),
            ("ActiveEnergyBurned",       "🔥 Active Energy",       f"{active_e:.0f} kcal"),
            ("BasalEnergyBurned",        "⚡ Basal Energy",        f"{basal_e:.0f} kcal"),
            ("PhysicalEffort",           "💪 Physical Effort",     f"{effort:.2f}"),
            ("WalkingSpeed",             "🏃 Walking Speed",       f"{walk_spd:.1f} km/h"),
            ("RespiratoryRate",          "🌬️ Respiratory Rate",   f"{resp:.0f} br/min"),
            ("sleep_hours",              "😴 Sleep",               f"{sleep_h:.1f} hrs"),
        ]

        normal_count, abnormal_list = 0, []
        rows_html = ""
        for key, label, val_str in check_keys:
            status, css = assess(key, raw[key])
            ref = REFS.get(key,("","","",""))
            lw,nl,nh,hw = ref
            if "Normal" in status: normal_count += 1
            else: abnormal_list.append(label)
            rows_html += f"""
            <div class="result-row">
              <span style="font-weight:500;width:180px">{label}</span>
              <span style="font-family:monospace;font-size:.9rem;width:120px">{val_str}</span>
              <span class="{css}">{status}</span>
              <span style="opacity:.5;font-size:.8rem;margin-left:auto">
                Normal: {nl}–{nh}</span>
            </div>"""

        # Derived scores
        s_status, s_css = ("High ⚠️","tag-abnormal") if stress>5 else ("Normal ✅","tag-normal")
        r_status, r_css = ("Low ⚠️","tag-warn") if recovery<0.3 else ("Normal ✅","tag-normal")
        rows_html += f"""
        <div class="result-row">
          <span style="font-weight:500;width:180px">🧠 Stress Score</span>
          <span style="font-family:monospace;font-size:.9rem;width:120px">{stress:.2f}</span>
          <span class="{s_css}">{s_status}</span>
          <span style="opacity:.5;font-size:.8rem;margin-left:auto">Normal: 0–5</span>
        </div>
        <div class="result-row">
          <span style="font-weight:500;width:180px">🔄 Recovery Index</span>
          <span style="font-family:monospace;font-size:.9rem;width:120px">{recovery:.2f}</span>
          <span class="{r_css}">{r_status}</span>
          <span style="opacity:.5;font-size:.8rem;margin-left:auto">Normal: 0.3–1.0</span>
        </div>"""

        st.markdown(rows_html, unsafe_allow_html=True)

        # ML anomaly score
        st.markdown("---")
        st.subheader("🤖 ML Anomaly Score")
        model, scaler, fcols, thresh_ml = load_artefacts()
        mse, is_anomaly = 0.0, False

        if model is not None:
            derived = {
                "stress_score":             stress,
                "sedentary_index":          int(raw["StepCount"]<10),
                "recovery_index":           recovery,
                "HeartRate_roll5":          raw["HeartRate"],
                "StepCount_roll5":          raw["StepCount"],
                "ActiveEnergyBurned_roll5": raw["ActiveEnergyBurned"],
            }
            row_dict = {**raw, **derived}
            row_arr  = np.array([float(row_dict.get(c,0.0)) for c in fcols],
                                dtype=np.float32).reshape(1,-1)
            row_sc   = np.clip(scaler.transform(row_arr), 0.0, 1.0)
            recon    = model.predict(row_sc, verbose=0)
            mse      = float(np.mean(np.square(row_sc - recon)))
            is_anomaly = mse > thresh_ml

            m1,m2,m3 = st.columns(3)
            m1.metric("Reconstruction Error", f"{mse:.5f}")
            m2.metric("Threshold",            f"{thresh_ml:.5f}")
            m3.metric("Pattern",              "⚠️ Anomalous" if is_anomaly else "✅ Normal")

            # fig_g = go.Figure(go.Bar(
            #     x=[min(mse/thresh_ml,3.0)], y=[""], orientation="h",
            #     marker_color="#ef4444" if is_anomaly else "#22c55e",
            #     text=[f"MSE={mse:.5f}"], textposition="outside",
            # ))
            # fig_g.add_vline(x=1.0, line_dash="dot", line_color="#f59e0b",
            #                 annotation_text="Threshold")
            # fig_g.update_layout(**PLOTLY_LAYOUT, height=90,
            #                     xaxis=dict(range=[0,3.2],showticklabels=False,showgrid=False),
            #                     yaxis=dict(showgrid=False,showticklabels=False),
            #                     margin=dict(l=0,r=120,t=8,b=8))
            # st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})

            if is_anomaly:
                st.error("🚨 Your values show an unusual pattern vs your historical data.")
            else:
                st.success("✅ Your values are consistent with your normal health pattern.")
        else:
            st.warning("⚠️ Run `python main.py` once to enable ML anomaly detection.")

        # Summary banner
        st.markdown("---")
        st.subheader("💡 Key Insights")
        if len(abnormal_list) == 0:
            st.success(f"✅ All metrics are within normal range!")
        elif len(abnormal_list) <= 2:
            st.warning(f"⚠️ Outside normal range: **{', '.join(abnormal_list)}**")
        else:
            st.error(f"🚨 Multiple concerns: **{', '.join(abnormal_list)}**")

        tips = []
        if hr       > 100: tips.append("❤️ High heart rate — rest, hydrate, avoid caffeine.")
        if hrv      <  20: tips.append("📈 Low HRV — your body needs recovery. Prioritise sleep.")
        if steps    < 4000:tips.append("🚶 Low steps — even a 20-min walk helps.")
        if sleep_h  <  6:  tips.append("😴 Under 6 hrs sleep — affects every other metric.")
        if stress   >  5:  tips.append("🧠 High stress score — try breathing exercises.")
        if resp     >  20: tips.append("🌬️ Elevated respiratory rate — check for stress/illness.")
        if not tips:       tips.append("🌟 No specific concerns. Keep it up!")
        for t in tips: st.markdown(f"- {t}")

        # Save to log
        log = load_manual_log()
        entry_date = datetime.date.today()
        if not log.empty and "date" in log.columns:
            log = log[log["date"]!=entry_date]
        log = pd.concat([log, pd.DataFrame([{
            "date": entry_date, "is_manual":1,
            "reconstruction_error": mse,
            "anomaly": int(is_anomaly),
            **raw,
        }])], ignore_index=True)
        save_manual_log(log)
        st.caption(f"✅ Entry saved for {entry_date}.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Settings":
    st.title("⚙️ Settings")
    st.subheader("👤 Personal Profile")

    profile = load_profile()
    OCCS = ["Student","Software Engineer","Healthcare Worker","Teacher",
            "Athlete","Office Worker","Freelancer","Retired","Other"]

    age = st.number_input("Age", 1, 120, int(profile.get("age",25)))
    occ = st.selectbox("Occupation", OCCS,
                       index=OCCS.index(profile.get("occupation","Other"))
                       if profile.get("occupation","Other") in OCCS else len(OCCS)-1)

    if st.button("💾 Save Profile", type="primary"):
        save_profile(int(age), occ)
        st.success("Profile saved!")

    st.markdown("---")
    st.subheader("📋 Manual Entry Log")
    log = load_manual_log()
    if not log.empty and len(log)>0:
        show_cols = [c for c in ["date","sleep_hours","HeartRate","StepCount",
                                  "stress_score","anomaly"] if c in log.columns]
        st.dataframe(log[show_cols].sort_values("date", ascending=False)
                     if "date" in log.columns else log[show_cols],
                     use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear All Manual Entries"):
            os.remove(MANUAL_LOG_PATH); st.success("Cleared."); st.rerun()
    else:
        st.info("No manual entries yet. Use ✏️ Manual Entry in the sidebar.")

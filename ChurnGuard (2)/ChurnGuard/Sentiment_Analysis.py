import os, io, re, time, warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from wordcloud import WordCloud, STOPWORDS
import streamlit as st
from streamlit_option_menu import option_menu

warnings.filterwarnings("ignore")

def run_sentiment_app():

# ─────────────────────────────────────────────────────────────────────────────
# 0. PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="ChurnGuard Pro | Sentiment Intelligence",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# ─────────────────────────────────────────────────────────────────────────────
# 1. DESIGN SYSTEM  (CSS)
# ─────────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* ── Google Fonts ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

    /* ── Base ─────────────────────────────────────────────────── */
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #F0F4FF; color: #0F172A; }

    /* ── Remove Streamlit chrome ──────────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 2rem 3rem 2rem; max-width: 1400px; }

    /* ── Scrollbar ────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #F0F4FF; }
    ::-webkit-scrollbar-thumb { background: #BFDBFE; border-radius: 3px; }

    /* ── Sidebar ──────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stTextArea textarea,
    [data-testid="stSidebar"] .stSelectbox select {
        background: #F8FAFC !important; color: #0F172A !important;
        border: 1px solid #CBD5E1 !important; border-radius: 8px;
    }

    /* ── Topbar / Brand ───────────────────────────────────────── */
    .brand-bar {
        display: flex; align-items: center; gap: 16px;
        padding: 22px 0 18px 0; margin-bottom: 4px;
        border-bottom: 1px solid #E2E8F0;
    }
    .brand-logo {
        width: 46px; height: 46px; border-radius: 12px;
        background: linear-gradient(135deg,#2563EB,#0EA5E9);
        display: flex; align-items: center; justify-content: center;
        font-size: 24px; flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(37,99,235,0.25);
    }
    .brand-name { font-family: 'Syne', sans-serif; font-size: 1.45rem; font-weight: 800;
        background: linear-gradient(90deg,#2563EB,#0EA5E9); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; line-height: 1.1; }
    .brand-sub { font-size: 0.72rem; color: #94A3B8; letter-spacing: .12em; text-transform: uppercase; }
    .nav-badge { 
        display: inline-block; font-size: 0.62rem; font-weight: 600;
        padding: 2px 8px; border-radius: 20px; letter-spacing: .06em;
        background: #EFF6FF; color: #2563EB;
        border: 1px solid #BFDBFE; margin-left: 6px;
    }

    /* ── Section heading ──────────────────────────────────────── */
    .section-heading {
        font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700;
        color: #64748B; letter-spacing: .08em; text-transform: uppercase;
        margin: 28px 0 14px 0; display: flex; align-items: center; gap: 8px;
    }
    .section-heading::after {
        content:''; flex: 1; height: 1px;
        background: linear-gradient(90deg,rgba(37,99,235,0.25),transparent);
    }

    /* ── KPI Cards ────────────────────────────────────────────── */
    .kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 28px; }
    .kpi-card {
        background: #FFFFFF; border: 1px solid #E2E8F0;
        border-radius: 16px; padding: 22px 20px; position: relative; overflow: hidden;
        transition: border-color .25s, transform .2s;
        box-shadow: 0 1px 6px rgba(37,99,235,0.06);
    }
    .kpi-card:hover { border-color: #2563EB; transform: translateY(-3px); }
    .kpi-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: var(--accent);
    }
    .kpi-label { font-size: 0.72rem; font-weight: 600; color: #94A3B8;
        letter-spacing: .1em; text-transform: uppercase; margin-bottom: 10px; }
    .kpi-value { font-family: 'DM Mono', monospace; font-size: 2rem; font-weight: 500;
        color: var(--accent); line-height: 1; margin-bottom: 8px; }
    .kpi-delta { font-size: 0.75rem; color: #94A3B8; display: flex; align-items: center; gap: 4px; }
    .kpi-icon { position: absolute; right: 18px; top: 18px; font-size: 1.8rem; opacity: 1; }

    /* ── Chart card ───────────────────────────────────────────── */
    .chart-title { font-family: 'DM Sans', sans-serif; font-size: 0.88rem; font-weight: 600;
        color: #64748B; letter-spacing: .05em; text-transform: uppercase; margin-bottom: 14px; }

    /* ── Alert Banner ─────────────────────────────────────────── */
    .alert-critical { 
        background: linear-gradient(90deg,rgba(239,68,68,.08),rgba(239,68,68,.03));
        border: 1px solid rgba(239,68,68,.25); border-left: 4px solid #EF4444;
        border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
        font-size: 0.85rem; color: #FCA5A5;
    }
    .alert-warning { 
        background: linear-gradient(90deg,rgba(245,158,11,.08),rgba(245,158,11,.03));
        border: 1px solid rgba(245,158,11,.25); border-left: 4px solid #F59E0B;
        border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
        font-size: 0.85rem; color: #FCD34D;
    }
    .alert-ok { 
        background: linear-gradient(90deg,rgba(16,185,129,.08),rgba(16,185,129,.03));
        border: 1px solid rgba(16,185,129,.25); border-left: 4px solid #10B981;
        border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
        font-size: 0.85rem; color: #6EE7B7;
    }

    /* ── Insight Pills ────────────────────────────────────────── */
    .pill-pos { display:inline-block; background:rgba(16,185,129,.12); color:#10B981;
        border:1px solid rgba(16,185,129,.3); border-radius:20px; 
        padding:3px 10px; font-size:.75rem; font-weight:600; margin:3px; }
    .pill-neut { display:inline-block; background:#F1F5F9; color:#64748B;
        border:1px solid #CBD5E1; border-radius:20px; 
        padding:3px 10px; font-size:.75rem; font-weight:600; margin:3px; }
    .pill-neg { display:inline-block; background:rgba(239,68,68,.12); color:#EF4444;
        border:1px solid rgba(239,68,68,.3); border-radius:20px; 
        padding:3px 10px; font-size:.75rem; font-weight:600; margin:3px; }

    /* ── Prediction Result ────────────────────────────────────── */
    .pred-result {
        padding: 16px 20px; border-radius: 12px; text-align: center;
        font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;
        letter-spacing: .04em; margin-top: 10px;
    }
    .pred-positive { background: rgba(16,185,129,.1); border: 1px solid rgba(16,185,129,.35); color: #10B981; }
    .pred-negative { background: rgba(239,68,68,.1); border: 1px solid rgba(239,68,68,.35); color: #EF4444; }
    .pred-neutral  { background: rgba(148,163,184,.1); border: 1px solid rgba(148,163,184,.3); color: #94A3B8; }

    /* ── Confidence bar ───────────────────────────────────────── */
    .conf-bar-wrap { margin-top: 12px; }
    .conf-label { font-size:.72rem; color:#64748B; margin-bottom:5px; 
        display:flex; justify-content:space-between; }
    .conf-track { height:6px; background:#E2E8F0; border-radius:3px; overflow:hidden; }
    .conf-fill-pos  { height:100%; background:linear-gradient(90deg,#059669,#10B981); border-radius:3px; }
    .conf-fill-neut { height:100%; background:linear-gradient(90deg,#475569,#94A3B8); border-radius:3px; }
    .conf-fill-neg  { height:100%; background:linear-gradient(90deg,#B91C1C,#EF4444); border-radius:3px; }

    /* ── Dataframe ────────────────────────────────────────────── */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

     /* ── Buttons ──────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg,#2563EB,#0EA5E9);
        color: white;
        border: none;
        border-radius: 6px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        padding: .5rem 1.5rem;
        transition: all .2s;
    }

    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* DOWNLOAD BUTTON */
    div.stDownloadButton > button {
        background: linear-gradient(135deg,#2563EB,#0EA5E9);
        color: white;
        border: none;
        border-radius: 6px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        padding: .5rem 1.5rem;
        transition: all .2s;
        width: 100%;
    }

    div.stDownloadButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* ── Tabs ─────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: #F8FAFC; border-radius: 12px 12px 0 0;
        border-bottom: 1px solid #E2E8F0; gap: 0; padding: 4px 8px 0 8px;
    }
    .stTabs [data-baseweb="tab"] { 
        color: #64748B; font-weight: 600; font-size: .85rem;
        padding: 10px 20px; border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] { 
        color: #60A5FA !important; 
        background: rgba(30,86,160,.12) !important;
        border-bottom: 2px solid #1E56A0 !important;
    }

    /* ── Selectbox / Multiselect / Slider — FIXED ─────────────── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
    }
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] div,
    .stSelectbox [data-baseweb="select"] input {
        color: #0F172A !important;
    }
    .stMultiSelect [data-baseweb="select"] input {
        color: #0F172A !important;
    }
    [data-baseweb="tag"] { color: #FFFFFF !important; }
    [data-baseweb="tag"] span { color: #FFFFFF !important; }
    [data-baseweb="tag"] div  { color: #FFFFFF !important; }
    [data-baseweb="tag"] p    { color: #FFFFFF !important; }
    [data-baseweb="tag"] button svg { fill: #FFFFFF !important; }
    [data-baseweb="menu"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08) !important;
    }
    [data-baseweb="menu"] li {
        color: #0F172A !important;
        background: #FFFFFF !important;
    }
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] [aria-selected="true"] {
        background: #EFF6FF !important;
        color: #2563EB !important;
    }
    [data-baseweb="select"] [data-testid="stWidgetLabel"],
    .stSelectbox input::placeholder,
    .stMultiSelect input::placeholder {
        color: #94A3B8 !important;
    }
    .stSelectbox svg,
    .stMultiSelect svg:not([data-baseweb="tag"] svg) {
        fill: #64748B !important;
    }
    .stSlider [data-baseweb="slider"] { color: #1E56A0; }

    /* ── Selectbox di Menu sidebar (atas) ─────────────────────── */
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div {
        color: #0F172A !important;
    }
                
    /* ── Nav link default ─────────────────────────────────────── */
    .nav-link {
        color: #64748B !important;
    }

    /* ── Nav link selected ────────────────────────────────────── */
    .nav-link-selected {
        background: linear-gradient(135deg,#2563EB,#0EA5E9) !important;
        color: white !important;
        border: 1px solid transparent !important;
        box-shadow: 0 4px 12px rgba(37,99,235,0.25) !important;
    }

    /* ── Icon default abu ─────────────────────────────────────── */
    .nav-link i,
    .nav-link svg {
        color: #6B7280 !important;
        fill: #6B7280 !important;
        opacity: 1 !important;
    }

    /* ── Icon warna: inherit dari parent <a> ─────────────────── */
    /* Icon <i> ada di DALAM <a class="nav-link">.               */
    /* Tidak ada inline color di <i> → cukup set color di <a>.   */
    /* Saat .active, color:white di <a> otomatis diwariskan.     */
    [data-testid="stSidebar"] .nav-link:not(.active) i.icon {
        color: #6B7280;
    }
    [data-testid="stSidebar"] .nav-link.active i.icon {
        color: inherit;
    }

    /* ── Teks putih pas dipilih ───────────────────────────────── */
    .nav-link-selected span,
    [data-testid="stSidebar"] .nav-link-selected span,
    [data-testid="stSidebar"] .nav-link.active {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────────────────────────────────────
    _LEGEND_BASE = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0)

    PLOTLY_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#94A3B8", size=12),
        margin=dict(l=10, r=10, t=36, b=10),
        legend=_LEGEND_BASE,
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#0F172A", bordercolor="#2563EB"),
    )

    def plotly_layout(**overrides):
        base = dict(PLOTLY_LAYOUT)
        if "legend" in overrides:
            base["legend"] = {**_LEGEND_BASE, **overrides.pop("legend")}
        base.update(overrides)
        return base

    COLOR_POS  = "#10B981"
    COLOR_NEG  = "#EF4444"
    COLOR_BLUE = "#3B82F6"
    COLOR_AMB  = "#F59E0B"

    STOPWORDS_ID = {
    "yang","dan","di","ke","dari","ini","itu","dengan","untuk","tidak","ada",
    "dalam","sudah","kami","kamu","saya","aja","ya","yg","ga","gak","juga",
    "tapi","kalau","bisa","kita","semua","apa","lagi","karena","atau","kalau",
    "app","aplikasi","sangat","banget","untuk","jadi","buat","pake","pakai",
    "udah","belum","masih","nya","si","the","of","to","and","in","is","it",
    "a","an","was","are","be","has","had","have","but","or","do","did","not",
    }

    def clean_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD ASSETS (cached)
# ─────────────────────────────────────────────────────────────────────────────
    @st.cache_resource(show_spinner=False)
    def load_model():
        BASE = "models"
        paths = {
            "model":    os.path.join(BASE, "sentiment_model.pkl"),
            "vec":      os.path.join(BASE, "tfidf_vectorizer.pkl"),
            "features": os.path.join(BASE, "forward_features.joblib"),
        }
        model    = joblib.load(paths["model"])    if os.path.exists(paths["model"])    else None
        vec      = joblib.load(paths["vec"])      if os.path.exists(paths["vec"])      else None
        features = joblib.load(paths["features"]) if os.path.exists(paths["features"]) else None
        return model, vec, features

    def apply_pipeline(texts: list[str], vectorizer, features, model) -> np.ndarray:
        X = vectorizer.transform(texts)
        if features is not None:
            if hasattr(features, "transform"):
                X = features.transform(X)
            elif isinstance(features, (np.ndarray, list)):
                X = X[:, features]
        return model.predict(X)

    def apply_pipeline_proba(texts: list[str], vectorizer, features, model) -> np.ndarray | None:
        if not hasattr(model, "predict_proba"):
            return None
        X = vectorizer.transform(texts)
        if features is not None:
            if hasattr(features, "transform"):
                X = features.transform(X)
            elif isinstance(features, (np.ndarray, list)):
                X = X[:, features]
        return model.predict_proba(X)

    LABEL_MAP = {
        0: "negative", 1: "neutral", 2: "positive",
        "0": "negative", "1": "neutral", "2": "positive",
        "negative": "negative", "neutral": "neutral", "positive": "positive",
    }

    @st.cache_data(show_spinner=False, ttl=300)
    def load_data():
        csv_path = "ulasan_aplikasi.csv"
        if not os.path.exists(csv_path):
            return None, None

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        POSSIBLE_TEXT = ["Review","review","content","text","ulasan","komentar","comment"]
        col_text = next((c for c in POSSIBLE_TEXT if c in df.columns), df.columns[0])

        POSSIBLE_LABEL = ["sentiment","label","Sentiment","Label","sentimen","Sentimen"]
        col_label = next((c for c in POSSIBLE_LABEL if c in df.columns), None)

        if col_label:
            df["sentiment_raw"] = df[col_label].astype(str).str.strip().str.lower()
            def norm_label(x):
                x = str(x).lower()
                if any(k in x for k in ["pos","good","great","bagu","pua","2"]): return "positive"
                if any(k in x for k in ["neg","bad","buruk","kecel","kece","0"]): return "negative"
                return "neutral"
            df["sentiment"] = df["sentiment_raw"].apply(norm_label)
        else:
            model_auto, vec_auto, feat_auto = load_model()
            if model_auto and vec_auto:
                df["text_for_pred"] = df[col_text].astype(str).apply(clean_text)
                preds = apply_pipeline(df["text_for_pred"].tolist(), vec_auto, feat_auto, model_auto)
                df["sentiment"] = [LABEL_MAP.get(p, "neutral") for p in preds]
            else:
                df["sentiment"] = "positive"

        POSSIBLE_DATE = ["date","Date","tanggal","Tanggal","created_at","at","time","timestamp"]
        col_date = next((c for c in POSSIBLE_DATE if c in df.columns), None)
        if col_date:
            df["date"] = pd.to_datetime(df[col_date], errors="coerce")
        else:
            np.random.seed(42)
            n = len(df)
            base = datetime.now() - timedelta(days=180)
            df["date"] = [base + timedelta(days=int(x)) for x in np.random.uniform(0,180,n)]

        POSSIBLE_RATING = ["rating","Rating","score","Score","bintang","star"]
        col_rating = next((c for c in POSSIBLE_RATING if c in df.columns), None)
        if col_rating:
            df["rating"] = pd.to_numeric(df[col_rating], errors="coerce")
        else:
            df["rating"] = df["sentiment"].map({"positive":5,"negative":2,"neutral":3})

        df = df.dropna(subset=[col_text]).copy()
        df["text_clean"]   = df[col_text].astype(str).apply(clean_text)
        df["char_length"]  = df[col_text].astype(str).apply(len)
        df["word_count"]   = df["text_clean"].str.split().apply(len)
        df["month"]        = df["date"].dt.to_period("M").astype(str)
        df["week"]         = df["date"].dt.isocalendar().week.astype(int)
        df["day_of_week"]  = df["date"].dt.day_name()
        df["hour"]         = df["date"].dt.hour if col_date else np.random.randint(7, 23, len(df))

        df["display_status"] = df["sentiment"].map({
            "positive": "🟢 Positive", "negative": "🔴 Negative", "neutral": "⚪ Neutral"
        })

        return df, col_text

# ─────────────────────────────────────────────────────────────────────────────
# 4. WORDCLOUD GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
    def wordcloud_fig(df: pd.DataFrame, col: str, sentiment: str, cmap: str, bg: str = "#FFFFFF"):
        subset = df[df["sentiment"] == sentiment]
        if subset.empty:
            return None
        text = " ".join(subset[col].astype(str))
        custom_stop = STOPWORDS.union(STOPWORDS_ID)
        try:
            wc = WordCloud(
                background_color=None, mode="RGBA",
                width=900, height=380, colormap=cmap,
                max_words=120, stopwords=custom_stop,
                collocations=False, prefer_horizontal=0.85,
                min_font_size=10,
            ).generate(text)
        except Exception:
            return None

        fig, ax = plt.subplots(figsize=(9, 3.8), facecolor="none")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        return fig

# ─────────────────────────────────────────────────────────────────────────────
# 5. INSIGHT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
    def top_n_words(texts: pd.Series, n: int = 8) -> list[tuple]:
        from collections import Counter
        words = []
        for t in texts.astype(str):
            words.extend([w for w in t.split() if len(w) > 3 and w not in STOPWORDS_ID])
        return Counter(words).most_common(n)

    def compute_insights(df: pd.DataFrame, col: str) -> dict:
        pos = df[df["sentiment"] == "positive"]
        neg = df[df["sentiment"] == "negative"]
        neut = df[df["sentiment"] == "neutral"]
        total = len(df)
        neg_pct = len(neg) / total * 100 if total > 0 else 0

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        now = df["date"].max()
        last_month = df[df["date"] >= now - timedelta(days=30)]
        prev_month  = df[(df["date"] >= now - timedelta(days=60)) & (df["date"] < now - timedelta(days=30))]

        lm_neg = (last_month["sentiment"] == "negative").sum()
        pm_neg = (prev_month["sentiment"] == "negative").sum()
        delta_neg = lm_neg - pm_neg

        top_pos_words = top_n_words(pos["text_clean"] if not pos.empty else pd.Series([], dtype=str))
        top_neg_words = top_n_words(neg["text_clean"] if not neg.empty else pd.Series([], dtype=str))
        top_neu_words = top_n_words(neut["text_clean"] if not neut.empty else pd.Series([], dtype=str))

        return dict(
            total=total, pos=len(pos), neg=len(neg), neut=len(neut),
            neg_pct=neg_pct, delta_neg=delta_neg,
            avg_rating=df["rating"].mean() if "rating" in df.columns else 0,
            top_pos_words=top_pos_words, top_neg_words=top_neg_words, top_neu_words=top_neu_words,
            lm_neg=lm_neg,
        )

# ─────────────────────────────────────────────────────────────────────────────
# 6. CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
    def chart_sentiment_donut(df):
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        color_map = {"positive": COLOR_POS, "negative": COLOR_NEG, "neutral": "#64748B"}
        fig = px.pie(
            counts, values="count", names="sentiment", hole=0.62,
            color="sentiment", color_discrete_map=color_map,
        )
        fig.update_traces(
            textinfo="percent", textfont_size=12, textfont_color="white",
            marker=dict(line=dict(color="#FFFFFF", width=3)),
            pull=[0.04 if s == "negative" else 0 for s in counts["sentiment"]],
        )
        fig.add_annotation(
            text=f"{len(df)}<br><span style='font-size:10px'>Reviews</span>",
            x=0.5, y=0.5, font=dict(size=20, color="#E2E8F0", family="DM Mono"),
            showarrow=False
        )
        fig.update_layout(**plotly_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.1, x=0.5, xanchor="center")
        ), title="Sentiment Distribution", showlegend=True)
        return fig

    def chart_trend_area(df):
        trend = (
            df.groupby(["month", "sentiment"])
            .size().reset_index(name="count")
        )
        color_map = {"positive": COLOR_POS, "negative": COLOR_NEG, "neutral": "#64748B"}
        fig = px.area(
            trend, x="month", y="count", color="sentiment",
            color_discrete_map=color_map, markers=True,
        )
        fig.update_traces(line_width=2)
        fig.update_layout(
            **plotly_layout(), title="Monthly Sentiment Trend",
            xaxis=dict(showgrid=False, tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
        )
        return fig

    def chart_rating_histogram(df):
        fig = px.histogram(
            df, x="rating", color="sentiment",
            color_discrete_map={"positive": COLOR_POS, "negative": COLOR_NEG, "neutral": "#64748B"},
            nbins=10, barmode="overlay", opacity=0.8,
        )
        fig.update_layout(**plotly_layout(), title="Rating Distribution by Sentiment",
                        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
                        bargap=0.1)
        return fig

    def chart_day_heatmap(df):
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        heat = df.groupby(["day_of_week","sentiment"]).size().unstack(fill_value=0)
        heat = heat.reindex([d for d in day_order if d in heat.index])
        heat = heat.get("negative", pd.Series(0, index=heat.index)).reset_index()
        heat.columns = ["day","neg_count"]
        fig = px.bar(
            heat, x="day", y="neg_count",
            color="neg_count", color_continuous_scale=["#FEE2E2","#EF4444","#991B1B"],
            text="neg_count",
        )
        fig.update_traces(textposition="outside", textfont_color="#E2E8F0")
        fig.update_layout(**plotly_layout(), title="Negative Reviews by Day of Week",
                        coloraxis_showscale=False,
                        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#E2E8F0"))
        return fig

    def chart_scatter_length_rating(df):
        sample = df.sample(min(800, len(df)), random_state=42)
        color_map = {"positive": COLOR_POS, "negative": COLOR_NEG, "neutral": "#64748B"}
        fig = px.scatter(
            sample, x="word_count", y="rating", color="sentiment",
            color_discrete_map=color_map, opacity=0.65,
            hover_data={"word_count": True, "rating": True},
            size="char_length", size_max=14,
        )
        fig.update_layout(**plotly_layout(), title="Word Count vs Rating",
                        xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
                        yaxis=dict(showgrid=True, gridcolor="#E2E8F0"))
        return fig

    def chart_top_words_bar(df, col, sentiment, color, n=10):
        subset = df[df["sentiment"] == sentiment]
        if subset.empty:
            return None
        words = top_n_words(subset["text_clean"], n)
        if not words:
            return None
        words_df = pd.DataFrame(words, columns=["word", "count"]).sort_values("count")
        fig = px.bar(
            words_df, x="count", y="word", orientation="h",
            color_discrete_sequence=[color],
        )
        fig.update_traces(marker_line_width=0, opacity=0.9)
        fig.update_layout(
            **plotly_layout(), title=f"Top Keywords — {'Positive' if sentiment=='positive' else 'Neutral' if sentiment=='neutral' else 'Negative'}",
            xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
            yaxis=dict(showgrid=False),
            bargap=0.3,
        )
        return fig

    def chart_hourly_volume(df):
        hour_data = df.groupby(["hour", "sentiment"]).size().reset_index(name="count")
        color_map = {"positive": COLOR_POS, "negative": COLOR_NEG, "neutral": "#64748B"}
        fig = px.bar(
            hour_data, x="hour", y="count", color="sentiment",
            color_discrete_map=color_map, barmode="stack",
        )
        fig.update_layout(
            **plotly_layout(), title="Review Volume by Hour",
            xaxis=dict(title="Hour of Day", showgrid=False, dtick=2),
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
        )
        return fig

    def chart_churn_gauge(neg_pct: float):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=neg_pct,
            number={"suffix": "%", "font": {"size": 32, "color": "#E2E8F0", "family": "DM Mono"}},
            delta={"reference": 30, "increasing": {"color": COLOR_NEG}, "decreasing": {"color": COLOR_POS}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748B", "tickwidth": 1},
                "bar": {"color": COLOR_NEG if neg_pct > 50 else COLOR_AMB if neg_pct > 30 else COLOR_POS,
                        "thickness": 0.28},
                "bgcolor": "#FFFFFF",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(16,185,129,.08)"},
                    {"range": [30, 60], "color": "rgba(245,158,11,.06)"},
                    {"range": [60, 100], "color": "rgba(239,68,68,.06)"},
                ],
                "threshold": {"line": {"color": COLOR_NEG, "width": 2}, "thickness": 0.8, "value": 50},
            },
            title={"text": "Churn Risk Index", "font": {"color": "#94A3B8", "size": 13, "family": "DM Sans"}},
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        fig.update_layout(**plotly_layout(), height=260)
        return fig

# ─────────────────────────────────────────────────────────────────────────────
# 7. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
    def render_sidebar(model, vectorizer, features, df, col_text):
        with st.sidebar:
            st.markdown("""
            <div class="brand-bar">
                <div class="brand-logo">🛡️</div>
                <div>
                    <div class="brand-name">BY.U</div>
                    <div class="brand-sub">Sentiment Intelligence</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-heading">Navigation</div>', unsafe_allow_html=True)
            page = option_menu(
                menu_title=None,
                options=[
                    "Executive Dashboard",
                    "Deep Analysis",
                    "Data Explorer",
                    "AI Predictor"
                ],
                icons=[
                    "bar-chart-fill",
                    "graph-up-arrow",
                    "table",
                    "robot"
                ],
                menu_icon="none",
                default_index=0,
                styles={
                    "container": {
                        "padding": "0!important",
                        "background-color": "transparent",
                    },
                    "nav-link": {
                        "font-size": "14px",
                        "font-weight": "600",
                        "text-align": "left",
                        "margin": "6px 0",
                        "padding": "12px 14px",
                        "border-radius": "10px",
                        "background-color": "#FFFFFF",
                        "color": "#64748B",
                        "border": "1px solid #E2E8F0",
                        "transition": "all .2s ease",
                    },
                    "nav-link:hover": {
                        "background-color": "#EFF6FF",
                        "color": "#2563EB",
                        "border": "1px solid #BFDBFE",
                    },
                    "icon": {
                        "font-size": "16px",
                    },
                    "nav-link-selected": {
                        "background": "linear-gradient(135deg,#2563EB,#0EA5E9)",
                        "color": "white",
                        "border": "1px solid transparent",
                        "box-shadow": "0 4px 12px rgba(37,99,235,0.25)",
                    },
                }
            )

            # ── FIX icon: karena tidak ada inline color di <i>, icon akan
            # inherit color dari parent <a>. Cukup set warna abu untuk
            # state non-active via CSS, dan active akan otomatis putih.
            st.markdown("""
            <style>
            /* Icon abu untuk nav-link non-active */
            [data-testid="stSidebar"] .nav-link:not(.active) i.icon {
                color: #6B7280 !important;
            }
            /* Icon inherit putih dari parent .active — tidak perlu override */
            [data-testid="stSidebar"] .nav-link.active i.icon {
                color: inherit !important;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-heading">Filters</div>', unsafe_allow_html=True)

            months = ["All"] + sorted(df["month"].unique().tolist(), reverse=True)
            sel_month = st.selectbox("Period", months)

            sentiments = st.multiselect(
                "Sentiments", ["positive","negative","neutral"],
                default=["positive","negative","neutral"],
            )

            min_wc, max_wc = int(df["word_count"].min()), int(df["word_count"].max())
            wc_range = st.slider("Min Word Count", min_wc, max_wc, min_wc)

            st.markdown('<div class="section-heading">Quick Predict</div>', unsafe_allow_html=True)

            user_text = st.text_area("", placeholder="Ketik ulasan pelanggan di sini…", height=120, key="sidebar_input")

            if st.button("⚡ Analisis Sentimen"):
                if not user_text.strip():
                    st.warning("Masukkan teks terlebih dahulu.")
                elif model is None or vectorizer is None:
                    st.error("Model belum dimuat. Periksa folder models/")
                else:
                    cleaned   = clean_text(user_text)
                    preds     = apply_pipeline([cleaned], vectorizer, features, model)
                    sentiment = LABEL_MAP.get(preds[0], "neutral")
                    proba     = apply_pipeline_proba([cleaned], vectorizer, features, model)

                    icons = {"positive": "🟢 POSITIVE", "negative": "🔴 NEGATIVE", "neutral": "⚪ NEUTRAL"}
                    css   = {"positive": "pred-positive", "negative": "pred-negative", "neutral": "pred-neutral"}
                    st.markdown(f'<div class="pred-result {css.get(sentiment,"pred-neutral")}">{icons.get(sentiment, sentiment)}</div>', unsafe_allow_html=True)

                    if proba is not None:
                        row = proba[0]
                        for cls, prob in zip(["negative","neutral","positive"][:len(row)], row):
                            pct = prob * 100
                            if cls == "positive":
                                bar_cls = "conf-fill-pos"
                            elif cls == "neutral":
                                bar_cls = "conf-fill-neut"
                            else:
                                bar_cls = "conf-fill-neg"
                            st.markdown(f"""
                            <div class="conf-bar-wrap">
                            <div class="conf-label"><span>{cls.capitalize()}</span><span>{pct:.1f}%</span></div>
                            <div class="conf-track"><div class="{bar_cls}" style="width:{pct}%"></div></div>
                            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-heading">System Status</div>', unsafe_allow_html=True)

            if model is not None:
                st.markdown('<div class="alert-ok">✅ sentiment_model.pkl · Active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-critical">❌ sentiment_model.pkl · Not Found</div>', unsafe_allow_html=True)

            if vectorizer is not None:
                st.markdown('<div class="alert-ok">✅ tfidf_vectorizer.pkl · Active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-critical">❌ tfidf_vectorizer.pkl · Not Found</div>', unsafe_allow_html=True)

            if features is not None:
                feat_info = f"{len(features)} features" if hasattr(features, "__len__") else "selector active"
                st.markdown(f'<div class="alert-ok">✅ forward_features.joblib · {feat_info}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-warning">⚠️ forward_features.joblib · Not Found (pakai semua fitur)</div>', unsafe_allow_html=True)

            total_rec = len(df)
            st.markdown(f'<div class="alert-ok">📊 ulasan_aplikasi.csv · {total_rec:,} rows</div>', unsafe_allow_html=True)

            st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

            return page, sel_month, sentiments, wc_range

# ─────────────────────────────────────────────────────────────────────────────
# 8. PAGE — EXECUTIVE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
    def page_dashboard(df, col_text, insights):

        st.markdown(
            '<div class="section-heading">📈 Executive Dashboard</div>',
            unsafe_allow_html=True
        )

        total = insights.get('total', 0)
        pos   = insights.get('pos', 0)
        neut  = insights.get('neut', 0)
        neg   = insights.get('neg', 0)

        pos_pct = (pos / total * 100) if total > 0 else 0
        neu_pct = (neut / total * 100) if total > 0 else 0
        neg_pct = (neg / total * 100) if total > 0 else 0

        delta_neg  = insights.get("delta_neg", 0)
        delta_icon = "🔺" if delta_neg > 0 else "🔻"

        kpis = [
            ("#3B82F6", "Total Reviews",    f"{total:,}", "All-time data ingested"),
            ("#10B981", "Positive Reviews", f"{pos:,}",   f"{pos_pct:.1f}% of total"),
            ("#64748B", "Neutral Reviews",  f"{neut:,}",  f"{neu_pct:.1f}% of total"),
            ("#EF4444", "Negative Reviews", f"{neg:,}",   f"{delta_icon} {abs(delta_neg)} vs prev month"),
        ]

        cols = st.columns(4)
        for col, (accent, label, value, sub) in zip(cols, kpis):
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="--accent:{accent}">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-delta">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if neg_pct >= 50:
            st.markdown(f'<div class="alert-critical">🚨 <strong>CRITICAL:</strong> Sentimen negatif mendominasi ({neg_pct:.1f}%). Segera tinjau keluhan utama pelanggan di bawah.</div>', unsafe_allow_html=True)
        elif neg_pct >= 30:
            st.markdown(f'<div class="alert-warning">⚠️ <strong>WARNING:</strong> Sentimen negatif cukup tinggi ({neg_pct:.1f}%). Disarankan untuk menindaklanjuti ulasan negatif segera.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-ok">✅ <strong>SEHAT:</strong> Profil sentimen dalam batas normal ({neg_pct:.1f}% negatif). Pertahankan kualitas layanan.</div>', unsafe_allow_html=True)

        if insights.get("top_pos_words") or insights.get("top_neu_words") or insights.get("top_neg_words"):
            pos_pills = " ".join(f'<span class="pill-pos">{w}</span>' for w, _ in insights.get("top_pos_words", []))
            neu_pills = " ".join(f'<span class="pill-neut">{w}</span>' for w, _ in insights.get("top_neu_words", []))
            neg_pills = " ".join(f'<span class="pill-neg">{w}</span>' for w, _ in insights.get("top_neg_words", []))

            st.markdown(f"<div style='margin-bottom:8px'>🟢 <b>Trending Positive:</b> {pos_pills}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-bottom:8px'>🟡 <b>Trending Neutral:</b> {neu_pills}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-bottom:20px'>🔴 <b>Trending Negative:</b> {neg_pills}</div>", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(chart_sentiment_donut(df), use_container_width=True)
        with c2:
            st.plotly_chart(chart_trend_area(df), use_container_width=True)

        st.plotly_chart(chart_day_heatmap(df), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 9. PAGE — DEEP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
    def page_deep_analysis(df, col_text):
        st.markdown('<div class="section-heading">🔬 Deep Analysis</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["  📊 Statistical  ", "  ☁️ Word Intelligence  ", "  🕐 Time Analysis  "])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(chart_rating_histogram(df), use_container_width=True)
            with c2:
                st.plotly_chart(chart_scatter_length_rating(df), use_container_width=True)

            c3, c4, c5 = st.columns(3)
            fig_pos  = chart_top_words_bar(df, col_text, "positive", COLOR_POS)
            fig_neut = chart_top_words_bar(df, col_text, "neutral", "#64748B")
            fig_neg  = chart_top_words_bar(df, col_text, "negative", COLOR_NEG)
            with c3:
                if fig_pos:  st.plotly_chart(fig_pos,  use_container_width=True)
            with c4:
                if fig_neut: st.plotly_chart(fig_neut, use_container_width=True)
            with c5:
                if fig_neg:  st.plotly_chart(fig_neg,  use_container_width=True)

        with tab2:
            st.markdown('<div class="chart-title">☁️ Positive Sentiment — Strength Keywords</div>', unsafe_allow_html=True)
            fig_wc_pos = wordcloud_fig(df, "text_clean", "positive", "YlGn")
            if fig_wc_pos: st.pyplot(fig_wc_pos, use_container_width=True)
            else: st.info("No positive reviews found.")

            st.markdown('<div class="chart-title">☁️ Neutral Sentiment — Balanced Keywords</div>', unsafe_allow_html=True)
            fig_wc_neut = wordcloud_fig(df, "text_clean", "neutral", "Spectral")
            if fig_wc_neut: st.pyplot(fig_wc_neut, use_container_width=True)
            else: st.info("No neutral reviews found.")

            st.markdown('<div class="chart-title">☁️ Negative Sentiment — Pain Point Keywords</div>', unsafe_allow_html=True)
            fig_wc_neg = wordcloud_fig(df, "text_clean", "negative", "OrRd")
            if fig_wc_neg: st.pyplot(fig_wc_neg, use_container_width=True)
            else: st.info("No negative reviews found.")

        with tab3:
            st.plotly_chart(chart_hourly_volume(df), use_container_width=True)

            monthly = (
                df.groupby("month").apply(
                    lambda x: pd.Series({
                        "total": len(x),
                        "positive": (x["sentiment"] == "positive").sum(),
                        "neutral": (x["sentiment"] == "neutral").sum(),
                        "negative": (x["sentiment"] == "negative").sum(),
                        "neg_rate": f"{(x['sentiment']=='negative').sum()/len(x)*100:.1f}%",
                        "avg_rating": f"{x['rating'].mean():.2f} ⭐",
                    })
                ).reset_index()
            )
            st.markdown('<div class="section-heading">Monthly Breakdown</div>', unsafe_allow_html=True)
            st.dataframe(monthly, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 10. PAGE — DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
    def page_data_explorer(df, col_text):
        st.markdown('<div class="section-heading">📝 Data Explorer</div>', unsafe_allow_html=True)

        col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
        with col_s1:
            search_q = st.text_input("🔍 Keyword Search", placeholder="e.g. login, slow, crash, great…")
        with col_s2:
            sent_filter = st.selectbox("Filter by Sentiment", ["All", "positive", "negative", "neutral"])
        with col_s3:
            sort_by = st.selectbox("Sort by", ["Latest", "Oldest", "Longest", "Shortest"])

        fdf = df.copy()
        if search_q:
            fdf = fdf[fdf[col_text].str.contains(search_q, case=False, na=False)]
        if sent_filter != "All":
            fdf = fdf[fdf["sentiment"] == sent_filter]

        sort_map = {
            "Latest": ("date", False), "Oldest": ("date", True),
            "Longest": ("char_length", False), "Shortest": ("char_length", True),
        }
        sort_col, asc = sort_map[sort_by]
        fdf = fdf.sort_values(sort_col, ascending=asc)

        st.caption(f"Showing {len(fdf):,} of {len(df):,} records")

        display_cols = [col_text, "display_status", "rating", "char_length", "word_count", "date"]
        display_cols = [c for c in display_cols if c in fdf.columns]

        st.dataframe(
            fdf[display_cols].head(500),
            column_config={
                col_text: st.column_config.TextColumn("Review", width="large"),
                "display_status": st.column_config.TextColumn("Sentiment", width="small"),
                "rating": st.column_config.NumberColumn("Rating ⭐", format="%.1f"),
                "char_length": st.column_config.ProgressColumn(
                    "Chars", format="%d", min_value=0, max_value=int(df["char_length"].max())),
                "word_count": st.column_config.NumberColumn("Words", format="%d"),
                "date": st.column_config.DateColumn("Date"),
            },
            use_container_width=True,
            hide_index=True,
            height=480,
        )

        csv_buf = fdf.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Export Filtered Data (CSV)",
            data=csv_buf,
            file_name=f"churnguard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────────────────────────────────────
# 11. PAGE — AI PREDICTOR (BATCH)
# ─────────────────────────────────────────────────────────────────────────────
    def page_ai_predictor(model, vectorizer, features):
        st.markdown('<div class="section-heading">🤖 AI Predictor — Batch & Single</div>', unsafe_allow_html=True)

        feat_info = ""
        if features is not None:
            n_feat = len(features) if hasattr(features, "__len__") else "?"
            feat_info = f" → forward_features ({n_feat} features selected)"
        st.markdown(
            f'<div class="alert-ok" style="margin-bottom:16px">⚙️ <b>Pipeline:</b> '
            f'tfidf_vectorizer.pkl{feat_info} → sentiment_model.pkl</div>',
            unsafe_allow_html=True
        )

        tab_single, tab_batch = st.tabs(["  ✏️ Single Review  ", "  📦 Batch Upload  "])

        with tab_single:
            text_in = st.text_area("Teks Ulasan Pelanggan", placeholder="Tulis atau tempel ulasan di sini…", height=180)
            if st.button("⚡ Prediksi Sentimen"):
                if not text_in.strip():
                    st.warning("Masukkan teks terlebih dahulu.")
                elif model is None or vectorizer is None:
                    st.error("Model belum dimuat. Periksa folder models/")
                else:
                    cleaned   = clean_text(text_in)
                    preds     = apply_pipeline([cleaned], vectorizer, features, model)
                    sentiment = LABEL_MAP.get(preds[0], "neutral")

                    icons = {"positive":"🟢 POSITIVE","negative":"🔴 NEGATIVE","neutral":"⚪ NEUTRAL"}
                    css   = {"positive":"pred-positive","negative":"pred-negative","neutral":"pred-neutral"}
                    st.markdown(f'<div class="pred-result {css.get(sentiment,"pred-neutral")}">{icons.get(sentiment,sentiment)}</div>', unsafe_allow_html=True)

                    proba = apply_pipeline_proba([cleaned], vectorizer, features, model)
                    if proba is not None:
                        row    = proba[0]
                        labels = ["negative","neutral","positive"][:len(row)]
                        cols   = st.columns(len(row))
                        for c, cls, p in zip(cols, labels, row):
                            with c:
                                st.metric(cls.capitalize(), f"{p*100:.1f}%")

        with tab_batch:
            st.markdown("Upload CSV dengan kolom bernama **`Review`**, **`text`**, **`content`**, atau **`ulasan`**.")
            uploaded = st.file_uploader("Pilih file CSV", type=["csv"])
            if uploaded:
                if model is None or vectorizer is None:
                    st.error("Model/vectorizer tidak ditemukan. Pastikan sentiment_model.pkl & tfidf_vectorizer.pkl ada di models/")
                else:
                    batch_df  = pd.read_csv(uploaded)
                    possible  = ["Review","review","content","text","ulasan","Content","Text"]
                    bt_col    = next((c for c in possible if c in batch_df.columns), batch_df.columns[0])

                    with st.spinner(f"Memproses {len(batch_df):,} ulasan…"):
                        batch_df["text_clean"] = batch_df[bt_col].astype(str).apply(clean_text)
                        preds = apply_pipeline(batch_df["text_clean"].tolist(), vectorizer, features, model)
                        batch_df["predicted_sentiment"] = [LABEL_MAP.get(p, str(p)) for p in preds]

                        proba_all = apply_pipeline_proba(batch_df["text_clean"].tolist(), vectorizer, features, model)
                        if proba_all is not None:
                            labels = ["negative","neutral","positive"][:proba_all.shape[1]]
                            for i, cls in enumerate(labels):
                                batch_df[f"prob_{cls}"] = (proba_all[:, i] * 100).round(2)

                    st.success(f"✅ Berhasil memprediksi {len(batch_df):,} baris.")

                    sent_counts = batch_df["predicted_sentiment"].value_counts().reset_index()
                    sent_counts.columns = ["sentiment","count"]
                    color_map = {"positive":COLOR_POS,"negative":COLOR_NEG,"neutral":"#64748B"}
                    fig_sum = px.bar(sent_counts, x="sentiment", y="count", color="sentiment",
                                    color_discrete_map=color_map, text="count")
                    fig_sum.update_layout(**plotly_layout(), height=220, showlegend=False)
                    fig_sum.update_traces(textposition="outside")
                    st.plotly_chart(fig_sum, use_container_width=True)

                    cols_show = [bt_col, "predicted_sentiment"] + [c for c in batch_df.columns if c.startswith("prob_")]
                    st.dataframe(batch_df[cols_show].head(300), use_container_width=True, hide_index=True)

                    out = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️  Download Hasil Prediksi (CSV)", data=out,
                                    file_name=f"prediksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# 12. MAIN
# ─────────────────────────────────────────────────────────────────────────────
    def main_logic():
        model, vectorizer, features = load_model()
        df_raw, col_text  = load_data()

        if df_raw is None:
            st.error("⚠️  **ulasan_aplikasi.csv** tidak ditemukan. Pastikan file ada di root folder CHURNGUARD/")
            st.info("Kolom yang diharapkan: kolom teks ulasan (nama apapun) + opsional: sentiment, date, rating")
            st.stop()

        page, sel_month, sentiments, wc_range = render_sidebar(model, vectorizer, features, df_raw, col_text)

        df = df_raw.copy()
        if sel_month != "All":
            df = df[df["month"] == sel_month]
        if sentiments:
            df = df[df["sentiment"].isin(sentiments)]
        df = df[df["word_count"] >= wc_range]

        if df.empty:
            st.warning("Tidak ada data yang cocok dengan filter. Silakan sesuaikan filter di sidebar.")
            st.stop()

        insights = compute_insights(df, col_text)

        if page == "Executive Dashboard":
            page_dashboard(df, col_text, insights)
        elif page == "Deep Analysis":
            page_deep_analysis(df, col_text)
        elif page == "Data Explorer":
            page_data_explorer(df, col_text)
        elif page == "AI Predictor":
            page_ai_predictor(model, vectorizer, features)

    main_logic()
import io
import hashlib
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# ─────────────────────────────────────────
# PAGE CONFIG  (harus paling atas)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def run_churnguard_app():

    # ─────────────────────────────────────────
    # CUSTOM CSS (full app)
    # ─────────────────────────────────────────
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .main { background-color: #F0F4FF; }

        .stMetric {
            background: #FFFFFF; border: 1px solid #E2E8F0;
            border-radius: 10px; padding: 16px 20px;
            box-shadow: 0 1px 4px rgba(37,99,235,0.06);
        }
        .stMetric label { color: #64748B !important; font-size: .78rem !important;
            letter-spacing: .08em; text-transform: uppercase; }
        .stMetric [data-testid="stMetricValue"] {
            color: #0F172A !important; font-family: 'DM Sans', sans-serif; font-size: 1.8rem !important; }
        .stMetric [data-testid="stMetricDelta"] { font-size: .8rem !important; }

        .risk-card-high   { background:#FEF2F2; border-left:4px solid #EF4444; padding:14px 18px; border-radius:8px; margin:6px 0; color:#0F172A; }
        .risk-card-medium { background:#FFFBEB; border-left:4px solid #F59E0B; padding:14px 18px; border-radius:8px; margin:6px 0; color:#0F172A; }
        .risk-card-low    { background:#F0FDF4; border-left:4px solid #22C55E; padding:14px 18px; border-radius:8px; margin:6px 0; color:#0F172A; }

        .section-header {
            font-family: 'DM Sans', sans-serif; color: #2563EB; font-size: .75rem;
            letter-spacing: .15em; text-transform: uppercase;
            border-bottom: 1px solid #E2E8F0; padding-bottom: 8px; margin-bottom: 20px;
        }
        div[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
        .stButton > button {
            background: linear-gradient(135deg,#2563EB,#0EA5E9); color: white; border: none; border-radius: 6px;
            font-family: 'DM Sans', sans-serif; font-weight: 600; padding: .5rem 1.5rem; transition: all .2s;
        }
        .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
        .customer-detail-box {
            background: #FFFFFF; border: 1px solid #E2E8F0;
            border-radius: 10px; padding: 18px 22px; margin-bottom: 12px;
            box-shadow: 0 1px 4px rgba(37,99,235,0.06);
        }
        .accuracy-badge {
            background: #EFF6FF; border: 1px solid #BFDBFE;
            border-radius: 8px; padding: 10px 14px; text-align: center; margin-bottom: 8px;
        }
        .model-metric-card {
            background: #FFFFFF; border: 1px solid #E2E8F0;
            border-radius: 10px; padding: 18px; text-align: center;
            box-shadow: 0 1px 4px rgba(37,99,235,0.06);
        }
                
        /* menu aktif */
        .nav-link-selected {
            background: linear-gradient(135deg,#2563EB,#0EA5E9) !important;
            color: white !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(37,99,235,.25);
        }

        /* icon menu aktif jadi putih */
        .nav-link-selected i {
            color: white !important;
        }

        /* hover effect */
        .nav-link:hover {
            transform: translateX(3px);
            transition: 0.2s ease;
        }
        
        /* ── Sidebar Section Heading ───────────────────────── */
        .section-heading {
            font-family: 'Space Mono', monospace;
            font-size: .74rem;
            font-weight: 700;
            color: #64748B;
            letter-spacing: .12em;
            text-transform: uppercase;

            margin: 14px 0 8px 0;   /* ← lebih rapet */

            display: flex;
            align-items: center;
            gap: 8px;
        }

        .section-heading::after {
            content: '';
            flex: 1;
            height: 1px;
            background: linear-gradient(
                90deg,
                rgba(37,99,235,0.25),
                transparent
            );
        }
                
        /* ── Sidebar spacing balanced ───────────────────────── */
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 1.2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        /* jarak antar widget/sidebar item */
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            gap: 0.65rem !important;
        }

        /* divider */
        section[data-testid="stSidebar"] hr {
            margin: 1rem 0 !important;
        }

        /* text markdown */
        section[data-testid="stSidebar"] p {
            margin-bottom: 0.45rem !important;
        }
    </style>
    """, unsafe_allow_html=True)


    # ─────────────────────────────────────────
    # LOAD ARTIFACTS
    # ─────────────────────────────────────────
    MODEL_PATH = "models"
    @st.cache_resource
    def load_model():
        model         = joblib.load(f"{MODEL_PATH}/churnguard_model.joblib")
        forward_feats = joblib.load(f"{MODEL_PATH}/forward_features.joblib")
        oe_plan       = joblib.load(f"{MODEL_PATH}/encoder_plan.joblib")
        oe_nps        = joblib.load(f"{MODEL_PATH}/encoder_nps.joblib")
        return model, forward_feats, oe_plan, oe_nps

    @st.cache_data
    def load_data():
        return pd.read_csv("churnguard_predictions.csv")

    @st.cache_data
    def load_saved_metrics():
        """Baca model_metrics.joblib yang di-export dari notebook."""
        try:
            return joblib.load(f"{MODEL_PATH}/model_metrics.joblib")
        except FileNotFoundError:
            return None

    try:
        model, forward_features, oe_plan, oe_nps = load_model()
        MODEL_LOADED = True
    except FileNotFoundError:
        MODEL_LOADED = False

    try:
        df = load_data()
        DATA_LOADED = True
    except FileNotFoundError:
        DATA_LOADED = False
        df = pd.DataFrame()

    # Prioritaskan metrics dari joblib; fallback hitung dari CSV
    saved_metrics = load_saved_metrics()

    # ─────────────────────────────────────────
    # HELPER: ENCODING
    # ─────────────────────────────────────────
    CONTRACT_CATEGORIES = ['annual', 'monthly', 'quarterly']

    def encode_input(raw: dict) -> pd.DataFrame:
        df_in = pd.DataFrame([raw])
        df_in['plan_type_enc']    = oe_plan.transform(df_in[['plan_type']])[0]
        df_in['nps_category_enc'] = oe_nps.transform(df_in[['nps_category']])[0]
        for cat in CONTRACT_CATEGORIES:
            df_in[f'contract_type_{cat}'] = (df_in['contract_type'] == cat).astype(int)
        _pm = {'starter': 0, 'professional': 1, 'enterprise': 2}
        df_in['_plan_num']            = df_in['plan_type'].map(_pm)
        df_in['nps_x_adoption']       = df_in['avg_nps_score'] * df_in['avg_feature_adoption']
        df_in['usage_x_tech_tickets'] = df_in['avg_monthly_usage_hrs'] * df_in['technical_tickets']
        df_in['plan_x_usage']         = df_in['_plan_num'] * df_in['avg_monthly_usage_hrs']
        df_in['tenure_x_engagement']  = df_in['tenure_days'] * df_in['engagement_score']
        df_in['usage_per_tenure']     = df_in['avg_monthly_usage_hrs'] / (df_in['tenure_days'] / 30 + 0.01)
        df_in['tickets_per_month']    = df_in['total_tickets'] / (df_in['tenure_days'] / 30 + 0.01)
        df_in['late_x_dunning']       = df_in['payment_delay_rate'] * df_in['dunning_rate']
        df_in['inactive_level']       = (df_in['days_since_last_login'] > 30).astype(int)
        df_in['usage_consistency']    = 1 / (df_in.get('std_monthly_usage_hrs', pd.Series([0])) + 1)
        df_in['usage_drop_ratio']     = 1 - (df_in['min_monthly_usage_hrs'] / (df_in['max_monthly_usage_hrs'] + 0.001))
        df_in['engagement_score']     = (
            df_in['avg_feature_adoption'] * 0.4 +
            df_in['avg_monthly_usage_hrs'] * 0.4 +
            (1 / (df_in['days_since_last_login'] + 1)) * 100 * 0.2
        )
        df_in['risk_score'] = (
            df_in['usage_drop_ratio'] * 0.5 +
            df_in['inactive_level'] * 0.3 +
            (1 - df_in['avg_feature_adoption'] / 100) * 0.2
        )
        df_in['payment_cv'] = df_in.get('std_payment_value', pd.Series([0])) / (df_in['avg_payment_value'] + 1)
        for col in [f for f in forward_features if f not in df_in.columns]:
            df_in[col] = 0
        return df_in[forward_features]


    # ─────────────────────────────────────────
    # HELPER: GAUGE + RISK
    # ─────────────────────────────────────────
    def make_gauge(proba, risk_color, height=260):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Probability", 'font': {'color': '#64748B', 'size': 13}},
            number={'suffix': '%', 'font': {'color': '#0F172A', 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#94A3B8'},
                'bar': {'color': risk_color}, 'bgcolor': '#E2E8F0',
                'steps': [
                    {'range': [0, 40],   'color': '#F0FDF4'},
                    {'range': [40, 70],  'color': '#FFFBEB'},
                    {'range': [70, 100], 'color': '#FEF2F2'},
                ],
                'threshold': {'line': {'color': risk_color, 'width': 3}, 'thickness': .8, 'value': proba * 100}
            }
        ))
        fig.update_layout(paper_bgcolor='#FFFFFF', font_color='#0F172A',
                        margin=dict(l=20, r=20, t=40, b=20), height=height)
        return fig

    def risk_info(proba):
        if proba >= 0.7: return "🔴 HIGH RISK",  "#EF4444", "risk-card-high"
        if proba >= 0.4: return "🟡 MEDIUM RISK", "#F59E0B", "risk-card-medium"
        return            "🟢 LOW RISK",   "#22C55E", "risk-card-low"

    def fmt(v, fmt_str=None, suffix='', prefix='', default='—'):
        if v is None or (isinstance(v, float) and np.isnan(v)): return default
        if fmt_str: return f"{prefix}{v:{fmt_str}}{suffix}"
        return f"{prefix}{v}{suffix}"


    # ─────────────────────────────────────────
    # PDF REPORT GENERATOR
    # ─────────────────────────────────────────
    def generate_pdf_report(df_data: pd.DataFrame, saved_m: dict) -> bytes:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm
        )

        # ── Warna ──
        C_DARK    = colors.HexColor("#F8FAFC")
        C_PANEL   = colors.HexColor("#EFF6FF")
        C_BORDER  = colors.HexColor("#CBD5E1")
        C_BLUE    = colors.HexColor("#2563EB")
        C_GREEN   = colors.HexColor("#3fb950")
        C_YELLOW  = colors.HexColor("#d29922")
        C_RED     = colors.HexColor("#f85149")
        C_TEXT    = colors.HexColor("#0F172A")
        C_MUTED   = colors.HexColor("#64748B")
        C_WHITE   = colors.white

        styles = getSampleStyleSheet()
        def S(name, **kw):
            return ParagraphStyle(name, parent=styles['Normal'], **kw)

        sTitle    = S('T', fontSize=22, textColor=C_BLUE, fontName='Helvetica-Bold', spaceAfter=4, alignment=TA_CENTER)
        sSub      = S('S', fontSize=10, textColor=C_MUTED, spaceAfter=2, alignment=TA_CENTER)
        sH1       = S('H1', fontSize=13, textColor=C_BLUE, fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=6)
        sH2       = S('H2', fontSize=10, textColor=C_TEXT, fontName='Helvetica-Bold', spaceBefore=8, spaceAfter=4)
        sBody     = S('B',  fontSize=9,  textColor=C_TEXT, leading=14)
        sCaption  = S('C',  fontSize=8,  textColor=C_MUTED, spaceAfter=6)
        sRight    = S('R',  fontSize=8,  textColor=C_MUTED, alignment=TA_RIGHT)

        story = []
        now   = datetime.datetime.now().strftime("%d %B %Y, %H:%M")

        # ════════════════════════════════
        # COVER
        # ════════════════════════════════
        story.append(Spacer(1, 1.5*cm))
        story.append(Paragraph("🛡️ ChurnGuard", sTitle))
        story.append(Paragraph("Customer Churn Analysis Report", sSub))
        story.append(Paragraph(f"Generated: {now}  |  User: {st.session_state.get('username','—')}", sCaption))
        story.append(HRFlowable(width="100%", thickness=1, color=C_BORDER, spaceAfter=16))

        # ════════════════════════════════
        # 1. RINGKASAN EKSEKUTIF
        # ════════════════════════════════
        story.append(Paragraph("1. Ringkasan Eksekutif", sH1))

        total = len(df_data)
        if 'churn_proba' in df_data.columns:
            high_n   = int((df_data['churn_proba'] >= 0.7).sum())
            med_n    = int(((df_data['churn_proba'] >= 0.4) & (df_data['churn_proba'] < 0.7)).sum())
            low_n    = int((df_data['churn_proba'] < 0.4).sum())
            avg_prob = df_data['churn_proba'].mean()
        else:
            high_n = med_n = low_n = 0; avg_prob = 0.0

        churned = int(df_data['churn'].sum()) if 'churn' in df_data.columns else 0

        exec_data = [
            ['Metrik', 'Nilai', 'Keterangan'],
            ['Total Customer',        f"{total:,}",              'Seluruh data dalam sistem'],
            ['Churn Aktual',          f"{churned:,} ({churned/total*100:.1f}%)", 'Label ground truth'],
            ['Avg Churn Probability', f"{avg_prob:.1%}",         'Rata-rata probabilitas model'],
            ['High Risk (≥70%)',      f"{high_n:,} ({high_n/total*100:.1f}%)", 'Perlu tindakan segera'],
            ['Medium Risk (40-70%)',  f"{med_n:,} ({med_n/total*100:.1f}%)",  'Perlu monitoring'],
            ['Low Risk (<40%)',       f"{low_n:,} ({low_n/total*100:.1f}%)",   'Aman'],
        ]
        t_exec = Table(exec_data, colWidths=[5*cm, 3.5*cm, 8.5*cm])
        t_exec.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), C_PANEL),
            ('TEXTCOLOR',  (0,0), (-1,0), C_BLUE),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 8.5),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_DARK, C_PANEL]),
            ('TEXTCOLOR',  (0,1), (-1,-1), C_TEXT),
            ('GRID',       (0,0), (-1,-1), 0.4, C_BORDER),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING',(0,0),(-1,-1), 8),
            ('TOPPADDING', (0,0),(-1,-1), 6),
            ('BOTTOMPADDING',(0,0),(-1,-1),6),
            # Warnai baris High Risk merah
            ('TEXTCOLOR',  (1,4), (1,4), C_RED),
            ('TEXTCOLOR',  (1,5), (1,5), C_YELLOW),
            ('TEXTCOLOR',  (1,6), (1,6), C_GREEN),
        ]))
        story.append(t_exec)
        story.append(Spacer(1, 0.4*cm))

        # ════════════════════════════════
        # 2. PERFORMA MODEL
        # ════════════════════════════════
        story.append(Paragraph("2. Performa Model XGBoost", sH1))
        story.append(Paragraph(
            "Model dilatih menggunakan XGBoost dengan hyperparameter tuning via Optuna dan "
            "feature selection menggunakan Forward Selection. Metrik di bawah dihitung pada data test set.",
            sBody
        ))
        story.append(Spacer(1, 0.3*cm))

        if saved_m:
            perf_data = [
                ['Metrik', 'Nilai', 'Interpretasi'],
                ['Accuracy',  f"{saved_m.get('Accuracy',0):.1%}",  'Persentase prediksi yang benar secara keseluruhan'],
                ['F1-Score',  f"{saved_m.get('F1-Score',0):.1%}",  'Harmonic mean precision & recall'],
                ['Precision', f"{saved_m.get('Precision',0):.1%}", 'Dari prediksi churn, berapa yang benar-benar churn'],
                ['Recall',    f"{saved_m.get('Recall',0):.1%}",    'Dari yang churn, berapa berhasil terdeteksi'],
                ['Train-Test Gap', f"{saved_m.get('Gap',0):.4f}",  f"Status: {saved_m.get('Status','—')} — model tidak overfit"],
            ]
        else:
            perf_data = [['Metrik', 'Nilai', 'Keterangan'],
                        ['—', '—', 'model_metrics.joblib tidak ditemukan']]

        t_perf = Table(perf_data, colWidths=[3.5*cm, 3*cm, 10.5*cm])
        t_perf.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), C_PANEL),
            ('TEXTCOLOR',  (0,0), (-1,0), C_BLUE),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 8.5),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_DARK, C_PANEL]),
            ('TEXTCOLOR',  (0,1), (-1,-1), C_TEXT),
            ('TEXTCOLOR',  (1,1), (1,-1), C_GREEN),
            ('FONTNAME',   (1,1), (1,-1), 'Helvetica-Bold'),
            ('GRID',       (0,0), (-1,-1), 0.4, C_BORDER),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING',(0,0),(-1,-1), 8),
            ('TOPPADDING', (0,0),(-1,-1), 6),
            ('BOTTOMPADDING',(0,0),(-1,-1), 6),
        ]))
        story.append(t_perf)
        story.append(Spacer(1, 0.4*cm))

        # ════════════════════════════════
        # 3. DISTRIBUSI RISIKO PER PLAN
        # ════════════════════════════════
        story.append(Paragraph("3. Distribusi Risiko per Plan Type", sH1))

        if 'plan_type' in df_data.columns and 'churn_proba' in df_data.columns:
            plan_rows = [['Plan Type', 'Total', 'High Risk', 'Medium Risk', 'Low Risk', 'Avg Prob']]
            for plan in sorted(df_data['plan_type'].unique()):
                sub = df_data[df_data['plan_type'] == plan]
                n   = len(sub)
                plan_rows.append([
                    plan.title(),
                    str(n),
                    f"{(sub['churn_proba']>=0.7).sum()} ({(sub['churn_proba']>=0.7).mean():.0%})",
                    f"{((sub['churn_proba']>=0.4)&(sub['churn_proba']<0.7)).sum()} ({((sub['churn_proba']>=0.4)&(sub['churn_proba']<0.7)).mean():.0%})",
                    f"{(sub['churn_proba']<0.4).sum()} ({(sub['churn_proba']<0.4).mean():.0%})",
                    f"{sub['churn_proba'].mean():.1%}",
                ])
            t_plan = Table(plan_rows, colWidths=[2.8*cm, 1.8*cm, 3.2*cm, 3.2*cm, 3.2*cm, 2.8*cm])
            t_plan.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), C_PANEL),
                ('TEXTCOLOR',  (0,0), (-1,0), C_BLUE),
                ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE',   (0,0), (-1,-1), 8.5),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_DARK, C_PANEL]),
                ('TEXTCOLOR',  (0,1), (-1,-1), C_TEXT),
                ('GRID',       (0,0), (-1,-1), 0.4, C_BORDER),
                ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
                ('LEFTPADDING',(0,0),(-1,-1), 8),
                ('TOPPADDING', (0,0),(-1,-1), 6),
                ('BOTTOMPADDING',(0,0),(-1,-1), 6),
            ]))
            story.append(t_plan)
        story.append(Spacer(1, 0.4*cm))

        # ════════════════════════════════
        # 4. TOP 20 HIGH-RISK CUSTOMERS
        # ════════════════════════════════
        story.append(PageBreak())
        story.append(Paragraph("4. Top 20 Customer Berisiko Tertinggi", sH1))
        story.append(Paragraph(
            "Daftar customer dengan probabilitas churn tertinggi. Prioritaskan untuk segera dihubungi oleh tim Customer Success.",
            sBody
        ))
        story.append(Spacer(1, 0.3*cm))

        if 'churn_proba' in df_data.columns:
            top20 = df_data.sort_values('churn_proba', ascending=False).head(20)
            cols_show = ['customer_id', 'churn_proba', 'plan_type', 'contract_type', 'tenure_days']
            cols_show = [c for c in cols_show if c in top20.columns]
            headers   = {'customer_id':'Customer ID','churn_proba':'Churn Prob','plan_type':'Plan',
                        'contract_type':'Contract','tenure_days':'Tenure (hari)'}

            rows = [[headers.get(c, c) for c in cols_show]]
            for _, r in top20.iterrows():
                row_data = []
                for c in cols_show:
                    v = r[c]
                    if c == 'churn_proba':   row_data.append(f"{v:.1%}")
                    elif c == 'tenure_days': row_data.append(f"{int(v)} hari")
                    else:                    row_data.append(str(v))
                rows.append(row_data)

            col_w = [17/len(cols_show)*cm] * len(cols_show)
            t_top = Table(rows, colWidths=col_w)
            t_top.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), C_PANEL),
                ('TEXTCOLOR',  (0,0), (-1,0), C_BLUE),
                ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE',   (0,0), (-1,-1), 8),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_DARK, C_PANEL]),
                ('TEXTCOLOR',  (0,1), (-1,-1), C_TEXT),
                ('TEXTCOLOR',  (1,1), (1,-1), C_RED),
                ('FONTNAME',   (1,1), (1,-1), 'Helvetica-Bold'),
                ('GRID',       (0,0), (-1,-1), 0.4, C_BORDER),
                ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
                ('LEFTPADDING',(0,0),(-1,-1), 7),
                ('TOPPADDING', (0,0),(-1,-1), 5),
                ('BOTTOMPADDING',(0,0),(-1,-1), 5),
            ]))
            story.append(t_top)

        # ════════════════════════════════
        # 5. REKOMENDASI
        # ════════════════════════════════
        story.append(Spacer(1, 0.6*cm))
        story.append(Paragraph("5. Rekomendasi Tindakan", sH1))

        reco_data = [
            ['Segment', 'Tindakan yang Disarankan', 'Prioritas'],
            ['High Risk\n(≥70%)',   'Hubungi langsung dalam 48 jam. Tawarkan diskon retention,\nreview kontrak, atau dedicated support.', '🔴 Segera'],
            ['Medium Risk\n(40-70%)','Kirim email nurturing & survei kepuasan. Monitor usage\nmingguan dan jadwalkan check-in bulanan.',  '🟡 Minggu ini'],
            ['Low Risk\n(<40%)',    'Lanjutkan program loyalty & upsell. Libatkan dalam\nbeta feature untuk meningkatkan engagement.',  '🟢 Rutin'],
            ['Detractor NPS',       'Eskalasi ke Customer Success Manager. Lakukan root\ncause analysis tiket yang belum terselesaikan.', '🔴 Segera'],
            ['Feature Adoption <30%','Jadwalkan sesi onboarding ulang. Kirim tutorial\ntargeted sesuai plan yang digunakan.',            '🟡 Minggu ini'],
        ]
        t_reco = Table(reco_data, colWidths=[3*cm, 10.5*cm, 3.5*cm])
        t_reco.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), C_PANEL),
            ('TEXTCOLOR',  (0,0), (-1,0), C_BLUE),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 8.5),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_DARK, C_PANEL]),
            ('TEXTCOLOR',  (0,1), (-1,-1), C_TEXT),
            ('GRID',       (0,0), (-1,-1), 0.4, C_BORDER),
            ('VALIGN',     (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1), 8),
            ('TOPPADDING', (0,0),(-1,-1), 7),
            ('BOTTOMPADDING',(0,0),(-1,-1), 7),
        ]))
        story.append(t_reco)

        # ── Footer ──
        story.append(Spacer(1, 1*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
        story.append(Paragraph(
            f"ChurnGuard Report  ·  {now}  ·  Confidential",
            S('foot', fontSize=7, textColor=C_MUTED, alignment=TA_CENTER, spaceBefore=6)
        ))

        doc.build(story)
        return buf.getvalue()


    # ─────────────────────────────────────────
    # SIDEBAR
    # ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='padding:10px 0 20px 0'>
            <h1 style='margin:0;color:#2563EB;'>🛡️ ChurnGuard</h1>
            <p style='color:#64748B;font-size:13px;margin-top:4px'>
                Customer Retention Intelligence
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="
            background:#EFF6FF;
            padding:12px;
            border-radius:10px;
            border:1px solid #DBEAFE;
            margin-bottom:20px;
        ">
            <div style='font-size:12px;color:#64748B'>Logged in as</div>
            <div style='font-weight:700;color:#2563EB'>
                {st.session_state['username']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="section-heading">Navigation</div>',
            unsafe_allow_html=True
        )

        page = option_menu(
            menu_title=None,
            options=[
                "Overview",
                "Prediksi Customer",
                "Semua Data",
                "Detail Customer",
                "Feature Importance"
            ],
            icons=[
                "bar-chart-fill",
                "search",
                "table",
                "person-fill",
                "graph-up-arrow"
            ],
            menu_icon="cast",
            default_index=0,

            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "#FFFFFF",
                },

                "icon": {
                    "font-size": "18px"
                },

                "nav-link": {
                    "font-size": "14px",
                    "font-weight": "500",
                    "text-align": "left",
                    "margin": "8px 0",
                    "padding": "13px 16px",
                    "border-radius": "12px",
                    "border": "1px solid #DBEAFE",
                    "background-color": "#FFFFFF",
                    "color": "#0F172A",
                    "--hover-color": "#EFF6FF",
                    "transition": "all .2s ease-in-out",
                },

                "nav-link-selected": {
                    "background": "linear-gradient(135deg,#2563EB,#0EA5E9)",
                    "color": "white",
                    "font-weight": "600",
                    "border": "1px solid transparent",
                    "box-shadow": "0 4px 12px rgba(37,99,235,.25)",
                },
            }
        )

        st.markdown("---")

        if DATA_LOADED:
            total   = len(df)
            churned = int(df['churn'].sum()) if 'churn' in df.columns else 0
            st.markdown(f"**Total Customer:** `{total:,}`")
            st.markdown(f"**Churn Aktual:** `{churned:,}` ({churned/total*100:.1f}%)")

        # ── Model metrics dari joblib ──
        if saved_metrics:
            st.markdown("---")
            st.markdown(
                '<div class="section-heading">Performa Model</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"""
            <div class='accuracy-badge'>
                <div style='font-family:Space Mono, monospace;font-size:1.5rem;color:#2563EB;font-weight:700'>
                    {saved_metrics.get('Accuracy',0):.1%}
                </div>
                <div style='font-size:.7rem;color:#64748B;text-transform:uppercase;letter-spacing:.08em'>Accuracy</div>
            </div>
            <div class='accuracy-badge'>
                <div style='font-family:Space Mono, monospace;font-size:1.5rem;color:#22C55E;font-weight:700'>
                    {saved_metrics.get('F1-Score',0):.3f}
                </div>
                <div style='font-size:.7rem;color:#64748B;text-transform:uppercase;letter-spacing:.08em'>F1-Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Tombol Generate PDF ──
        if DATA_LOADED:
            st.markdown(
                '<div class="section-heading">Laporan PDF</div>',
                unsafe_allow_html=True
            )
            if st.button("📄 Generate & Download PDF", use_container_width=True, type="primary"):
                with st.spinner("Membuat laporan PDF..."):
                    pdf_bytes = generate_pdf_report(df, saved_metrics)
                fname = f"churnguard_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )

        st.markdown("---")
        # if st.button("🚪 Logout", use_container_width=True):
        #     logout()
        #     st.rerun()
        # st.caption("v1.0 · ChurnGuard Project")


    # ═══════════════════════════════════════════════════════
    # PAGE 1: OVERVIEW
    # ═══════════════════════════════════════════════════════
    if page == "Overview":
        st.markdown("## 📊 Overview")
        st.markdown("<p class='section-header'>ringkasan performa model & distribusi data</p>", unsafe_allow_html=True)

        if not DATA_LOADED:
            st.info("Load `churnguard_predictions.csv` untuk melihat halaman ini.")
            st.stop()

        # ── KPI row ──
        col1, col2, col3, col4 = st.columns(4)
        if 'churn_proba' in df.columns:
            high_n   = (df['churn_proba'] >= 0.7).sum()
            med_n    = ((df['churn_proba'] >= 0.4) & (df['churn_proba'] < 0.7)).sum()
            avg_prob = df['churn_proba'].mean()
            with col1: st.metric("Total Customer", f"{len(df):,}")
            with col2: st.metric("🔴 High Risk (≥70%)", f"{high_n:,}", f"{high_n/len(df)*100:.1f}% dari total", delta_color="inverse")
            with col3: st.metric("🟡 Medium Risk", f"{med_n:,}")
            with col4: st.metric("Avg Churn Probability", f"{avg_prob:.1%}")

        # ── Model Performance dari joblib ──
        if saved_metrics:
            st.markdown("---")
            st.markdown("#### 📐 Performa Model (dari Training)")

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(f"""
                <div class='model-metric-card'>
                    <div style='font-size:.72rem;color:#64748B;text-transform:uppercase;letter-spacing:.08em'>Accuracy</div>
                    <div style='font-family:Space Mono, monospace;font-size:1.6rem;color:#2563EB;font-weight:700;margin-top:6px'>
                        {saved_metrics.get('Accuracy',0):.1%}
                    </div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class='model-metric-card'>
                    <div style='font-size:.72rem;color:#64748B;text-transform:uppercase;letter-spacing:.08em'>F1-Score</div>
                    <div style='font-family:Space Mono, monospace;font-size:1.6rem;color:#22C55E;font-weight:700;margin-top:6px'>
                        {saved_metrics.get('F1-Score',0):.3f}
                    </div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class='model-metric-card'>
                    <div style='font-size:.72rem;color:#64748B;text-transform:uppercase;letter-spacing:.08em'>Precision</div>
                    <div style='font-family:Space Mono, monospace;font-size:1.6rem;color:#F59E0B;font-weight:700;margin-top:6px'>
                        {saved_metrics.get('Precision',0):.3f}
                    </div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class='model-metric-card'>
                    <div style='font-size:.72rem;color:#64748B;text-transform:uppercase;letter-spacing:.08em'>Recall</div>
                    <div style='font-family:Space Mono, monospace;font-size:1.6rem;color:#F59E0B;font-weight:700;margin-top:6px'>
                        {saved_metrics.get('Recall',0):.3f}
                    </div>
                </div>""", unsafe_allow_html=True)
            with m5:
                gap    = saved_metrics.get('Gap', 0)
                status = saved_metrics.get('Status', '—')
                gap_color = '#3fb950' if status == 'OK' else '#f85149'
                st.markdown(f"""
                <div class='model-metric-card'>
                    <div style='font-size:.72rem;color:#64748B;text-transform:uppercase;letter-spacing:.08em'>Train-Test Gap</div>
                    <div style='font-family:Space Mono, monospace;font-size:1.4rem;color:{gap_color};font-weight:700;margin-top:6px'>
                        {gap:.4f}
                    </div>
                    <div style='font-size:.72rem;color:{gap_color};margin-top:2px'>{status}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Charts bawah ──
        col_hist, col_bar = st.columns([1.5, 1])
        with col_hist:
            st.markdown("#### Distribusi Churn Probability")
            if 'churn_proba' in df.columns:
                fig = px.histogram(df, x='churn_proba', nbins=40,
                    color_discrete_sequence=['#2563EB'], template='plotly_white',
                    labels={'churn_proba': 'Churn Probability'})
                fig.add_vline(x=0.4, line_dash='dot', line_color='#d29922', annotation_text="Medium")
                fig.add_vline(x=0.7, line_dash='dot', line_color='#f85149', annotation_text="High")
                fig.update_layout(plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
                    margin=dict(l=0,r=0,t=10,b=0), height=300,
                    xaxis=dict(gridcolor='#E2E8F0'), yaxis=dict(gridcolor='#E2E8F0'))
                st.plotly_chart(fig, use_container_width=True)
        with col_bar:
            st.markdown("#### Churn Rate by Plan")
            if 'plan_type' in df.columns and 'churn' in df.columns:
                pc = df.groupby('plan_type')['churn'].mean().reset_index()
                pc.columns = ['Plan','Rate']
                pc['Rate %'] = pc['Rate'] * 100
                fig2 = px.bar(pc, x='Plan', y='Rate %',
                    color='Rate %', color_continuous_scale=['#3fb950','#d29922','#f85149'],
                    template='plotly_white')
                fig2.update_layout(plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
                    margin=dict(l=0,r=0,t=10,b=0), height=300, coloraxis_showscale=False,
                    xaxis=dict(gridcolor='#E2E8F0'), yaxis=dict(gridcolor='#E2E8F0'))
                st.plotly_chart(fig2, use_container_width=True)

        if 'tenure_days' in df.columns and 'churn_proba' in df.columns:
            st.markdown("#### Avg Churn Probability vs Tenure")
            df['tenure_bucket'] = pd.cut(df['tenure_days'],
                bins=[0,90,180,365,730,9999],
                labels=['< 3 bln','3-6 bln','6-12 bln','1-2 thn','> 2 thn'])
            tc = df.groupby('tenure_bucket', observed=True)['churn_proba'].mean().reset_index()
            fig3 = px.line(tc, x='tenure_bucket', y='churn_proba', markers=True,
                template='plotly_white', color_discrete_sequence=['#2563EB'],
                labels={'churn_proba':'Avg Churn Prob','tenure_bucket':'Tenure'})
            fig3.update_layout(plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
                margin=dict(l=0,r=0,t=10,b=0), height=260,
                xaxis=dict(gridcolor='#E2E8F0'), yaxis=dict(gridcolor='#E2E8F0', tickformat='.0%'))
            st.plotly_chart(fig3, use_container_width=True)


    # ═══════════════════════════════════════════════════════
    # PAGE 2: PREDIKSI
    # ═══════════════════════════════════════════════════════
    elif page == "Prediksi Customer":
        st.markdown("## 🔍 Prediksi Churn Customer")
        st.markdown("<p class='section-header'>masukkan data customer untuk prediksi real-time</p>", unsafe_allow_html=True)

        if not MODEL_LOADED:
            st.error("Model belum di-load. Pastikan file `.joblib` ada di folder yang sama.")
            st.stop()

        with st.form("predict_form"):
            st.markdown("#### 📋 Informasi Akun")
            c1,c2,c3 = st.columns(3)
            with c1: plan_type     = st.selectbox("Plan Type", ['starter','professional','enterprise'])
            with c2: contract_type = st.selectbox("Contract Type", CONTRACT_CATEGORIES)
            with c3: total_users   = st.number_input("Total Users", min_value=1, value=5)
            tenure_days = st.slider("Tenure (hari)", 0, 2000, 365)

            st.markdown("#### 💳 Billing")
            c1,c2,c3 = st.columns(3)
            with c1:
                avg_payment_value  = st.number_input("Avg Payment Value", value=500.0)
                payment_delay_rate = st.slider("Payment Delay Rate", 0.0, 1.0, 0.1)
            with c2:
                total_payments = st.number_input("Total Payments", min_value=0, value=12)
                max_delay_days = st.number_input("Max Delay Days", min_value=0, value=5)
            with c3:
                dunning_rate   = st.slider("Dunning Rate", 0.0, 1.0, 0.05)
                avg_delay_days = st.number_input("Avg Delay Days", min_value=0.0, value=2.0)

            st.markdown("#### 📱 Usage")
            c1,c2,c3 = st.columns(3)
            with c1:
                avg_monthly_usage_hrs = st.number_input("Avg Monthly Usage (hrs)", value=20.0)
                min_monthly_usage_hrs = st.number_input("Min Monthly Usage (hrs)", value=5.0)
            with c2:
                max_monthly_usage_hrs = st.number_input("Max Monthly Usage (hrs)", value=40.0)
                avg_feature_adoption  = st.slider("Avg Feature Adoption (%)", 0.0, 100.0, 50.0)
            with c3:
                days_since_last_login = st.number_input("Days Since Last Login", min_value=0, value=7)

            st.markdown("#### 💬 NPS & Support")
            c1,c2,c3 = st.columns(3)
            with c1:
                avg_nps_score    = st.slider("Avg NPS Score", 0.0, 10.0, 7.0)
                latest_nps_score = st.slider("Latest NPS Score", 0.0, 10.0, 7.0)
            with c2:
                nps_category  = st.selectbox("NPS Category", ['Detractor','Passive','Promoter'])
                total_tickets = st.number_input("Total Support Tickets", min_value=0, value=2)
            with c3:
                technical_tickets = st.number_input("Technical Tickets", min_value=0, value=1)
                unresolved_ratio  = st.slider("Unresolved Ticket Ratio", 0.0, 1.0, 0.2)

            submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True)

        if submitted:
            raw_input = {
                'plan_type': plan_type, 'contract_type': contract_type, 'total_users': total_users,
                'tenure_days': tenure_days, 'total_payments': total_payments,
                'total_revenue': avg_payment_value * total_payments,
                'avg_payment_value': avg_payment_value, 'std_payment_value': avg_payment_value * 0.1,
                'payment_delay_rate': payment_delay_rate, 'max_delay_days': max_delay_days,
                'avg_delay_days': avg_delay_days, 'dunning_count': int(dunning_rate * total_payments),
                'dunning_rate': dunning_rate, 'payment_cv': 0.1,
                'late_x_dunning': payment_delay_rate * dunning_rate,
                'avg_monthly_usage_hrs': avg_monthly_usage_hrs, 'max_monthly_usage_hrs': max_monthly_usage_hrs,
                'min_monthly_usage_hrs': min_monthly_usage_hrs,
                'std_monthly_usage_hrs': (max_monthly_usage_hrs - min_monthly_usage_hrs) / 4,
                'avg_feature_adoption': avg_feature_adoption, 'min_feature_adoption': avg_feature_adoption * 0.7,
                'days_since_last_login': days_since_last_login,
                'engagement_score': avg_feature_adoption*0.4 + avg_monthly_usage_hrs*0.4 + (1/(days_since_last_login+1))*100*0.2,
                'usage_drop_ratio': 1 - (min_monthly_usage_hrs / (max_monthly_usage_hrs + 0.001)),
                'inactive_level': int(days_since_last_login > 30),
                'usage_consistency': 1 / ((max_monthly_usage_hrs - min_monthly_usage_hrs) / 4 + 1),
                'risk_score': (1-min_monthly_usage_hrs/(max_monthly_usage_hrs+0.001))*0.5 + int(days_since_last_login>30)*0.3 + (1-avg_feature_adoption/100)*0.2,
                'avg_nps_score': avg_nps_score, 'min_nps_score': avg_nps_score * 0.7,
                'latest_nps_score': latest_nps_score, 'nps_response_count': 3, 'nps_std': 1.0,
                'nps_trend': latest_nps_score - avg_nps_score, 'nps_category': nps_category,
                'total_tickets': total_tickets, 'high_priority_tickets': 0,
                'open_tickets': int(total_tickets * unresolved_ratio),
                'billing_tickets': 0, 'technical_tickets': technical_tickets,
                'unresolved_ratio': unresolved_ratio, 'high_priority_ratio': 0.0, 'billing_ticket_ratio': 0.0,
                'nps_x_adoption': avg_nps_score * avg_feature_adoption,
                'usage_x_tech_tickets': avg_monthly_usage_hrs * technical_tickets,
                'plan_x_usage': {'starter':0,'professional':1,'enterprise':2}[plan_type] * avg_monthly_usage_hrs,
                'tenure_x_engagement': tenure_days * (avg_feature_adoption*0.4 + avg_monthly_usage_hrs*0.4),
                'usage_per_tenure': avg_monthly_usage_hrs / (tenure_days / 30 + 0.01),
                'tickets_per_month': total_tickets / (tenure_days / 30 + 0.01),
            }
            try:
                X_input = encode_input(raw_input)
                proba   = model.predict_proba(X_input)[0][1]
                rl, rc, rclass = risk_info(proba)
                st.markdown("---")
                st.markdown("### Hasil Prediksi")
                cg, cd = st.columns([1, 1.5])
                with cg: st.plotly_chart(make_gauge(proba, rc), use_container_width=True)
                with cd:
                    st.markdown(f"<div class='{rclass}'><h4 style='margin:0;color:{rc}'>{rl}</h4>"
                        f"<p style='margin:4px 0 0;color:#64748B;font-size:.85rem'>Probabilitas churn: "
                        f"<strong style='color:#0F172A'>{proba:.1%}</strong></p></div>", unsafe_allow_html=True)
                    st.markdown("**Signal yang terdeteksi:**")
                    sigs = []
                    if days_since_last_login > 30: sigs.append(f"⚠️ Tidak login selama {days_since_last_login} hari")
                    if payment_delay_rate > 0.3:   sigs.append(f"⚠️ Keterlambatan bayar {payment_delay_rate:.0%}")
                    if avg_feature_adoption < 30:  sigs.append(f"⚠️ Feature adoption rendah ({avg_feature_adoption:.0f}%)")
                    if nps_category == 'Detractor': sigs.append("⚠️ Customer adalah Detractor")
                    if unresolved_ratio > 0.5:     sigs.append(f"⚠️ {unresolved_ratio:.0%} tiket belum terselesaikan")
                    if tenure_days < 90:           sigs.append(f"⚠️ Customer baru ({tenure_days} hari)")
                    for s in sigs: st.markdown(s)
                    if not sigs: st.markdown("✅ Tidak ada sinyal risiko yang menonjol")
            except Exception as e:
                st.error(f"Error saat prediksi: {e}")
                st.exception(e)


    # ═══════════════════════════════════════════════════════
    # PAGE 3: SEMUA DATA
    # ═══════════════════════════════════════════════════════
    elif page == "Semua Data":
        st.markdown("## 📋 Semua Data Customer")
        st.markdown("<p class='section-header'>seluruh data dengan semua kolom</p>", unsafe_allow_html=True)

        if not DATA_LOADED:
            st.info("Load `churnguard_predictions.csv` untuk melihat halaman ini.")
            st.stop()

        c1,c2,c3,c4 = st.columns(4)
        with c1:
            risk_filter = st.multiselect("Risk Level",
                ['🔴 High (≥70%)', '🟡 Medium (40-70%)', '🟢 Low (<40%)'],
                default=['🔴 High (≥70%)', '🟡 Medium (40-70%)', '🟢 Low (<40%)'])
        with c2:
            plan_filter = st.multiselect("Plan", df['plan_type'].unique().tolist(),
                default=df['plan_type'].unique().tolist()) if 'plan_type' in df.columns else []
        with c3:
            contract_filter = st.multiselect("Contract", df['contract_type'].unique().tolist(),
                default=df['contract_type'].unique().tolist()) if 'contract_type' in df.columns else []
        with c4:
            search_id = st.text_input("Cari Customer ID", placeholder="ketik sebagian ID...")

        mask = pd.Series([True]*len(df), index=df.index)
        if 'churn_proba' in df.columns:
            if '🔴 High (≥70%)'     not in risk_filter: mask &= (df['churn_proba'] < 0.7)
            if '🟡 Medium (40-70%)' not in risk_filter: mask &= ~((df['churn_proba'] >= 0.4) & (df['churn_proba'] < 0.7))
            if '🟢 Low (<40%)'      not in risk_filter: mask &= (df['churn_proba'] >= 0.4)
        if plan_filter     and 'plan_type'     in df.columns: mask &= df['plan_type'].isin(plan_filter)
        if contract_filter and 'contract_type' in df.columns: mask &= df['contract_type'].isin(contract_filter)
        if search_id       and 'customer_id'   in df.columns:
            mask &= df['customer_id'].astype(str).str.contains(search_id, case=False, na=False)

        df_f = df[mask].copy()
        st.markdown(f"Menampilkan **{len(df_f):,}** dari **{len(df):,}** customer")

        dd = df_f.copy()

        if 'churn_proba' in dd.columns:

            # bikin value risk level
            risk_values = dd['churn_proba'].apply(
                lambda x: '🔴 High' if x >= 0.7
                else ('🟡 Medium' if x >= 0.4 else '🟢 Low')
            )

            # kalau kolom sudah ada → update
            if 'risk_level' in dd.columns:
                dd['risk_level'] = risk_values

            # kalau belum ada → insert setelah churn_proba
            else:
                dd.insert(
                    dd.columns.tolist().index('churn_proba') + 1,
                    'risk_level',
                    risk_values
                )

            # format probability jadi persen
            dd['churn_proba'] = dd['churn_proba'].apply(lambda x: f"{x:.1%}")

        if 'churn' in dd.columns:
            dd['churn'] = dd['churn'].map({1:'✅ Yes',0:'—'})

        if 'churn_pred' in dd.columns:
            dd['churn_pred'] = dd['churn_pred'].map({1:'✅ Yes',0:'—'})
            prio  = ['customer_id','churn_proba','risk_level','churn','churn_pred','plan_type','contract_type','tenure_days']
            other = [c for c in dd.columns if c not in prio]
            st.dataframe(dd[[c for c in prio if c in dd.columns]+other], use_container_width=True, height=540)
            st.download_button("⬇️ Download CSV (semua kolom)", df_f.to_csv(index=False),
                "churnguard_all_data.csv", "text/csv")


    # ═══════════════════════════════════════════════════════
    # PAGE 4: DETAIL CUSTOMER
    # ═══════════════════════════════════════════════════════
    elif page == "Detail Customer":
        st.markdown("## 👤 Detail Customer")
        st.markdown("<p class='section-header'>profil lengkap dan analisis risiko per customer</p>", unsafe_allow_html=True)

        if not DATA_LOADED:
            st.info("Load `churnguard_predictions.csv` untuk melihat halaman ini.")
            st.stop()

        cs, csort = st.columns([2,1])
        with csort:
            sort_by = st.selectbox("Urutkan berdasarkan",
                ['Churn Prob (tertinggi)','Churn Prob (terendah)','Customer ID'])
        with cs:
            search_q = st.text_input("🔍 Filter Customer ID", placeholder="ketik sebagian ID...")

        if 'customer_id' in df.columns:
            if sort_by == 'Churn Prob (tertinggi)' and 'churn_proba' in df.columns:
                id_list = df.sort_values('churn_proba', ascending=False)['customer_id'].astype(str).tolist()
            elif sort_by == 'Churn Prob (terendah)' and 'churn_proba' in df.columns:
                id_list = df.sort_values('churn_proba', ascending=True)['customer_id'].astype(str).tolist()
            else:
                id_list = sorted(df['customer_id'].astype(str).tolist())
            if search_q: id_list = [i for i in id_list if search_q.lower() in i.lower()]
            if not id_list: st.warning("Tidak ada customer yang cocok."); st.stop()
            selected_id = st.selectbox(f"Pilih Customer ({len(id_list):,} tersedia)", id_list)
        else:
            st.warning("Kolom `customer_id` tidak ditemukan."); st.stop()

        row = df[df['customer_id'].astype(str) == str(selected_id)]
        if row.empty: st.warning(f"Customer `{selected_id}` tidak ditemukan."); st.stop()
        row = row.iloc[0]

        def _g(col, default=None): return row[col] if col in row.index else default

        proba = _g('churn_proba')
        rl, rc, rclass = risk_info(proba) if proba is not None else ("—","#64748B","risk-card-low")

        st.markdown("---")
        c1,c2,c3 = st.columns([2.2,1.2,1])
        with c1:
            st.markdown(f"### 👤 `{selected_id}`")
            pb = {'starter':'🥉 Starter','professional':'🥈 Professional','enterprise':'🥇 Enterprise'}
            st.markdown(f"**Plan:** {pb.get(str(_g('plan_type','')),str(_g('plan_type','—')))} &nbsp;|&nbsp; "
                        f"**Contract:** {str(_g('contract_type','—')).title()} &nbsp;|&nbsp; "
                        f"**Users:** {int(_g('total_users',0)) if _g('total_users') is not None else '—'}")
            t = _g('tenure_days')
            if t is not None:
                t = int(t)
                st.markdown(f"**Tenure:** {t} hari ({'~'+str(t//30)+' bulan' if t<365 else '~'+f'{t/365:.1f} tahun'})")
            actual = _g('churn'); pred = _g('churn_pred')
            if actual is not None and pred is not None:
                al = "✅ Churn" if actual==1 else "— Tidak Churn"
                pl = "✅ Churn" if pred==1   else "— Tidak Churn"
                mi = "✔️ Benar" if actual==pred else "❌ Salah prediksi"
                st.markdown(f"<div class='customer-detail-box' style='margin-top:12px'>"
                    f"<span style='font-size:.72rem;color:#64748B;text-transform:uppercase;letter-spacing:.08em'>Aktual vs Prediksi</span><br><br>"
                    f"Aktual: <strong>{al}</strong> &nbsp;|&nbsp; Prediksi: <strong>{pl}</strong> &nbsp;|&nbsp; {mi}"
                    f"</div>", unsafe_allow_html=True)
        with c2:
            if proba is not None: st.plotly_chart(make_gauge(proba, rc, height=220), use_container_width=True)
        with c3:
            if proba is not None:
                st.markdown(f"<div class='{rclass}' style='margin-top:30px;text-align:center'>"
                    f"<div style='font-size:1rem;color:{rc};font-weight:700'>{rl}</div>"
                    f"<div style='font-size:2rem;color:#0F172A;font-family:Space Mono, monospace;font-weight:700;margin-top:4px'>{proba:.1%}</div>"
                    f"<div style='font-size:.72rem;color:#64748B;margin-top:4px'>churn probability</div>"
                    f"</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ⚠️ Signal Risiko yang Terdeteksi")
        sigs = []
        if (_g('days_since_last_login') or 0) > 30:
            sigs.append(f"Tidak login selama **{int(_g('days_since_last_login',0))} hari**")
        if (_g('payment_delay_rate') or 0) > 0.3:
            sigs.append(f"Keterlambatan bayar **{_g('payment_delay_rate',0):.0%}**")
        if (_g('avg_feature_adoption') or 100) < 30:
            sigs.append(f"Feature adoption sangat rendah (**{_g('avg_feature_adoption',0):.1f}%**)")
        if str(_g('nps_category','')).lower() == 'detractor':
            sigs.append("Customer adalah **Detractor** (NPS rendah)")
        if (_g('unresolved_ratio') or 0) > 0.5:
            sigs.append(f"**{_g('unresolved_ratio',0):.0%}** tiket belum terselesaikan")
        if 0 < (_g('tenure_days') or 999) < 90:
            sigs.append(f"Customer baru (**{int(_g('tenure_days',0))} hari**)")
        if (_g('dunning_rate') or 0) > 0.2:
            sigs.append(f"Dunning rate tinggi (**{_g('dunning_rate',0):.0%}**)")
        if (_g('nps_trend') or 0) < -1:
            sigs.append(f"NPS menurun (trend: **{_g('nps_trend',0):.1f}**)")

        if sigs:
            sc = st.columns(2)
            for i,s in enumerate(sigs):
                with sc[i%2]: st.markdown(f"<div class='risk-card-medium'>⚠️ {s}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='risk-card-low'>✅ Tidak ada sinyal risiko yang menonjol</div>", unsafe_allow_html=True)

        st.markdown("---")
        tb, tu, tn, tt, tr = st.tabs(["💳 Billing","📱 Usage","💬 NPS","🎫 Support Tickets","🗂️ Semua Kolom"])

        def mrow(items):
            cols = st.columns(len(items))
            for col,(label,val) in zip(cols,items): col.metric(label,val)

        with tb:
            st.markdown("<p class='section-header'>data pembayaran & tagihan</p>", unsafe_allow_html=True)
            mrow([("Total Payments",    fmt(_g('total_payments'),'.0f')),
                ("Total Revenue",     fmt(_g('total_revenue'),',.0f',prefix='$')),
                ("Avg Payment Value", fmt(_g('avg_payment_value'),',.0f',prefix='$'))])
            mrow([("Payment Delay Rate", fmt(_g('payment_delay_rate'),'.1%')),
                ("Avg Delay Days",     fmt(_g('avg_delay_days'),'.1f',suffix=' hari')),
                ("Max Delay Days",     fmt(_g('max_delay_days'),'.0f',suffix=' hari'))])
            mrow([("Dunning Count", fmt(_g('dunning_count'),'.0f')),
                ("Dunning Rate",  fmt(_g('dunning_rate'),'.1%')),
                ("Payment CV",    fmt(_g('payment_cv'),'.3f'))])
        with tu:
            st.markdown("<p class='section-header'>pola penggunaan produk</p>", unsafe_allow_html=True)
            mrow([("Avg Monthly Usage", fmt(_g('avg_monthly_usage_hrs'),'.1f',suffix=' hrs')),
                ("Min Monthly Usage", fmt(_g('min_monthly_usage_hrs'),'.1f',suffix=' hrs')),
                ("Max Monthly Usage", fmt(_g('max_monthly_usage_hrs'),'.1f',suffix=' hrs'))])
            mrow([("Avg Feature Adoption", fmt(_g('avg_feature_adoption'),'.1f',suffix='%')),
                ("Days Since Last Login", fmt(_g('days_since_last_login'),'.0f',suffix=' hari')),
                ("Usage Drop Ratio",      fmt(_g('usage_drop_ratio'),'.1%'))])
            mrow([("Engagement Score", fmt(_g('engagement_score'),'.2f')),
                ("Usage per Tenure", fmt(_g('usage_per_tenure'),'.2f')),
                ("Inactive Level",   "Ya ⚠️" if _g('inactive_level')==1 else "Tidak ✅")])
        with tn:
            st.markdown("<p class='section-header'>net promoter score & sentimen</p>", unsafe_allow_html=True)
            nc = str(_g('nps_category','—'))
            ncc = {'Promoter':'#22C55E','Passive':'#F59E0B','Detractor':'#EF4444'}.get(nc,'#64748B')
            st.markdown(f"**NPS Category:** <span style='color:{ncc};font-weight:700;font-size:1.1rem'>{nc}</span>", unsafe_allow_html=True)
            mrow([("Avg NPS Score",    fmt(_g('avg_nps_score'),'.1f',suffix=' / 10')),
                ("Latest NPS Score", fmt(_g('latest_nps_score'),'.1f',suffix=' / 10')),
                ("Min NPS Score",    fmt(_g('min_nps_score'),'.1f',suffix=' / 10'))])
            mrow([("NPS Trend",          fmt(_g('nps_trend'),'+.2f')),
                ("NPS Std Dev",        fmt(_g('nps_std'),'.2f')),
                ("NPS Response Count", fmt(_g('nps_response_count'),'.0f'))])
        with tt:
            st.markdown("<p class='section-header'>support ticket & prioritas</p>", unsafe_allow_html=True)
            mrow([("Total Tickets",     fmt(_g('total_tickets'),'.0f')),
                ("Open (Unresolved)", fmt(_g('open_tickets'),'.0f')),
                ("Unresolved Ratio",  fmt(_g('unresolved_ratio'),'.1%'))])
            mrow([("Technical Tickets",     fmt(_g('technical_tickets'),'.0f')),
                ("Billing Tickets",       fmt(_g('billing_tickets'),'.0f')),
                ("High Priority Tickets", fmt(_g('high_priority_tickets'),'.0f'))])
            mrow([("High Priority Ratio",  fmt(_g('high_priority_ratio'),'.1%')),
                ("Billing Ticket Ratio", fmt(_g('billing_ticket_ratio'),'.1%')),
                ("Tickets per Month",    fmt(_g('tickets_per_month'),'.2f'))])
        with tr:
            st.markdown("<p class='section-header'>seluruh kolom data mentah</p>", unsafe_allow_html=True)
            raw_df = row.to_frame(name='Nilai').reset_index()
            raw_df.columns = ['Kolom','Nilai']
            st.dataframe(raw_df, use_container_width=True, height=520)


    # ═══════════════════════════════════════════════════════
    # PAGE 5: FEATURE IMPORTANCE
    # ═══════════════════════════════════════════════════════
    elif page == "Feature Importance":
        st.markdown("## 📈 Feature Importance")
        st.markdown("<p class='section-header'>fitur yang paling mempengaruhi prediksi model</p>", unsafe_allow_html=True)

        if not MODEL_LOADED:
            st.error("Model belum di-load.")
            st.stop()

        try:
            importances = model.named_steps['clf'].feature_importances_
            fi = pd.DataFrame({'Feature': list(forward_features), 'Importance': importances}
                            ).sort_values('Importance', ascending=True)
            top_n = st.slider("Tampilkan Top N Fitur", 5, len(fi), min(20, len(fi)))
            fis   = fi.tail(top_n)

            fig = go.Figure(go.Bar(
                x=fis['Importance'], y=fis['Feature'], orientation='h',
                marker=dict(color=fis['Importance'],
                    colorscale=[[0,'#DBEAFE'],[0.5,'#2563EB'],[1,'#1E3A8A']])
            ))
            fig.update_layout(template='plotly_white', plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
                xaxis=dict(gridcolor='#E2E8F0', title='Importance Score'),
                yaxis=dict(gridcolor='#E2E8F0'),
                height=max(400, top_n * 28), margin=dict(l=0,r=20,t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Tabel Detail")
            fit = fi.sort_values('Importance', ascending=False).copy()
            fit['Rank']       = range(1, len(fit)+1)
            fit['Importance'] = fit['Importance'].apply(lambda x: f"{x:.4f}")
            st.dataframe(fit[['Rank','Feature','Importance']], use_container_width=True)

        except Exception as e:
            st.error(f"Gagal mengambil feature importance: {e}")
            st.info("Pastikan model pipeline memiliki named step 'clf' dengan `feature_importances_`.")
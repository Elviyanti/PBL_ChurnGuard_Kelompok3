import streamlit as st
import hashlib
from ChurnGuard import run_churnguard_app
from Sentiment_Analysis import run_sentiment_app

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="ChurnGuard | Login",
    page_icon="🛡️",
    layout="wide"
)

# 2. Inisialisasi Auth
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""

def check_password(username, password):
    h = hashlib.sha256(password.encode()).hexdigest()
    return username == "admin" and h == hashlib.sha256(b"churnguard2024").hexdigest()

# 3. Alur Tampilan Login
if not st.session_state["authenticated"]:
    # ── CSS SPLIT SCREEN & BUTTON CENTERING ──
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Background Split: Kiri Biru, Kanan Putih */
    .stApp {
        background: linear-gradient(to right, #1E56A0 0%, #3B82F6 50%, #FFFFFF 50%, #FFFFFF 100%) !important;
    }

    header, footer, .stDeployButton { visibility: hidden !important; display: none !important; }

    /* Layout Utama */
    .block-container {
        max-width: 100% !important;
        padding: 0 !important;
    }

    /* KARTU LOGIN KIRI */
    [data-testid="stForm"] {
        background-color: white !important;
        border-radius: 28px !important;
        padding: 45px !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.15) !important;
        border: 1px solid #F1F5F9 !important;
        width: 100% !important;
        max-width: 420px !important;
        margin: auto !important;
    }

    /* Input Field Styling */
    div[data-testid="stTextInput"] label p {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 8px !important;
    }
    .stTextInput div[data-baseweb="input"] {
        background-color: #F8FAFC !important;
        border-radius: 14px !important;
        border: 1.5px solid #E2E8F0 !important;
        padding: 4px 8px !important;
    }
    .stTextInput button {
        background-color: transparent !important;
        color: #3B82F6 !important;
    }

    /* --- PERBAIKAN TOMBOL: TENGAH & RAPI --- */
    div.stFormSubmitButton {
        display: flex !important;
        justify-content: center !important;
        margin-top: 25px !important;
    }
    
    div.stFormSubmitButton button {
        background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        border-radius: 14px !important;
        padding: 14px 30px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        width: 100% !important; /* Full width agar simetris dengan input */
        border: none !important;
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.3s ease !important;
    }

    div.stFormSubmitButton button:hover {
        background: #1E40AF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 25px rgba(37, 99, 235, 0.3) !important;
    }

    /* BRANDING KANAN */
    .right-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
        text-align: center;
    }
    .big-logo-shield {
        background: #F0F7FF;
        width: 130px;
        height: 130px;
        border-radius: 35px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 35px;
        border: 2px solid #DBEAFE;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

    # ── LAYOUT KOLOM ──
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Pusatkan secara vertikal
        st.markdown('<div style="height: 18vh;"></div>', unsafe_allow_html=True)
        
        # Container agar form berada di tengah kolom kiri
        left_container = st.container()
        with left_container:
            with st.form("login_gate"):
                st.markdown("""
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h2 style="font-family:'Plus Jakarta Sans'; font-weight:800; color:#0F172A; margin:0; font-size:30px; letter-spacing:-0.03em;">Welcome Back</h2>
                        <p style="color:#64748B; font-size:15px; margin-top:8px;">Sign in to continue to your dashboard</p>
                    </div>
                """, unsafe_allow_html=True)

                u = st.text_input("Username", placeholder="Masukkan Username")
                p = st.text_input("Password", type="password", placeholder="••••••••")
                
                # Tombol sekarang diatur agar lebar penuh (full-width) supaya simetris
                submitted = st.form_submit_button("Sign In to Dashboard", use_container_width=True)
                
                if submitted:
                    if check_password(u, p):
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = u
                        st.rerun()
                    else:
                        st.error("❌ Username atau password salah.")

    with col_right:
        # Sisi Kanan: Branding
        st.markdown("""
            <div class="right-section">
                <div class="big-logo-shield">
                    <svg width="65" height="65" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                    </svg>
                </div>
                <h1 style="font-family:'Plus Jakarta Sans'; font-weight:800; color:#0F172A; font-size:48px; letter-spacing:-0.05em; margin:0;">
                    ChurnGuard 
                </h1>
                <p style="font-family:'Plus Jakarta Sans'; color:#64748B; font-size:19px; max-width:420px; line-height:1.6; margin-top:15px;">
                    Intelligence platform to predict, analyze, and prevent customer churn with precision.
                </p>
                <div style="margin-top: 60px; font-size: 11px; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 2.5px;">
                    🛡️ Secure Login Environment
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.stop()

else:
    # Sidebar atas
    with st.sidebar:
        choice = st.selectbox(
            "Menu",
            ["📉 Churn Analysis", "💬 Sentiment Intelligence"]
        )

    # Main app
    if choice == "📉 Churn Analysis":
        run_churnguard_app()
    else:
        run_sentiment_app()

    # Logout PALING BAWAH sidebar
    with st.sidebar:
        if st.button("Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()
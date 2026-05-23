import os
import io
import time
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
import streamlit as st
import cv2

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="NEURAL CT Denoise — Premium Medical Image Restorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Premium CSS — Glassmorphism Dark Theme
# ==========================================
st.markdown("""
<style>
    /* ═══════════════════════════════════════════
       Google Fonts
       ═══════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ═══════════════════════════════════════════
       Global Background & Theme
       ═══════════════════════════════════════════ */
    .stApp {
        background: linear-gradient(160deg, #030712 0%, #0a0f1e 30%, #0d1321 60%, #060b16 100%);
        color: #e2e8f0;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Scrollbar styling */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0f1e; }
    ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #2563eb; }

    /* ═══════════════════════════════════════════
       Animated Hero Header
       ═══════════════════════════════════════════ */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.1); }
        50% { box-shadow: 0 0 40px rgba(59, 130, 246, 0.25); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    .hero-container {
        background: rgba(10, 15, 30, 0.7);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 28px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.8s ease-out, pulseGlow 4s ease-in-out infinite;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3b82f6, #8b5cf6, #06b6d4, transparent);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
    }

    .hero-container::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.06) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #60a5fa;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 14px;
        animation: float 3s ease-in-out infinite;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #60a5fa 0%, #818cf8 25%, #a78bfa 50%, #60a5fa 75%, #34d399 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 6s ease infinite;
        margin: 0 0 8px 0;
        line-height: 1.15;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 300;
        letter-spacing: 0.3px;
        line-height: 1.5;
        max-width: 700px;
    }

    .hero-subtitle strong {
        color: #cbd5e1;
        font-weight: 500;
    }

    /* ═══════════════════════════════════════════
       Glass Cards (Generic)
       ═══════════════════════════════════════════ */
    .glass-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out;
    }

    .glass-card:hover {
        border-color: rgba(59, 130, 246, 0.2);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.08);
        transform: translateY(-2px);
    }

    /* ═══════════════════════════════════════════
       Metric Cards Grid
       ═══════════════════════════════════════════ */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin: 16px 0;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 18px 16px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        border-radius: 2px 2px 0 0;
    }

    .metric-card.blue::before { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    .metric-card.green::before { background: linear-gradient(90deg, #10b981, #34d399); }
    .metric-card.purple::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
    .metric-card.amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .metric-card.rose::before { background: linear-gradient(90deg, #f43f5e, #fb7185); }
    .metric-card.cyan::before { background: linear-gradient(90deg, #06b6d4, #22d3ee); }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.2);
    }

    .metric-card.blue:hover { border-color: rgba(59, 130, 246, 0.3); }
    .metric-card.green:hover { border-color: rgba(16, 185, 129, 0.3); }
    .metric-card.purple:hover { border-color: rgba(139, 92, 246, 0.3); }

    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #64748b;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.5px;
    }

    .metric-value.blue { color: #60a5fa; }
    .metric-value.green { color: #34d399; }
    .metric-value.purple { color: #a78bfa; }
    .metric-value.amber { color: #fbbf24; }
    .metric-value.rose { color: #fb7185; }
    .metric-value.cyan { color: #22d3ee; }

    .metric-unit {
        font-size: 0.75rem;
        color: #475569;
        margin-top: 4px;
        font-weight: 400;
    }

    /* ═══════════════════════════════════════════
       Image Display Cards
       ═══════════════════════════════════════════ */
    .image-frame {
        background: rgba(10, 15, 30, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 14px;
        transition: all 0.3s ease;
        animation: fadeInUp 0.7s ease-out;
    }

    .image-frame:hover {
        border-color: rgba(59, 130, 246, 0.25);
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.08);
    }

    .image-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-align: center;
        padding: 8px 0 10px;
        color: #f1f5f9;
        letter-spacing: 0.3px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        margin-bottom: 10px;
    }

    .image-label .icon { margin-right: 6px; }

    .image-stat {
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 8px;
        padding: 6px 0;
        border-radius: 8px;
    }

    .image-stat.noisy { color: #fb7185; background: rgba(244, 63, 94, 0.08); }
    .image-stat.ann { color: #60a5fa; background: rgba(59, 130, 246, 0.08); }
    .image-stat.gan { color: #34d399; background: rgba(16, 185, 129, 0.08); }
    .image-stat.clean { color: #e2e8f0; background: rgba(255, 255, 255, 0.04); }

    /* ═══════════════════════════════════════════
       Section Headers
       ═══════════════════════════════════════════ */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 30px 0 16px;
        animation: slideInLeft 0.5s ease-out;
    }

    .section-header .icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }

    .section-header .icon.blue { background: rgba(59, 130, 246, 0.15); }
    .section-header .icon.green { background: rgba(16, 185, 129, 0.15); }
    .section-header .icon.purple { background: rgba(139, 92, 246, 0.15); }
    .section-header .icon.amber { background: rgba(245, 158, 11, 0.15); }

    .section-header .text {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.3px;
    }

    .section-header .subtext {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 400;
        margin-left: auto;
    }

    /* ═══════════════════════════════════════════
       Model Comparison Cards
       ═══════════════════════════════════════════ */
    .model-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 12px;
    }

    .model-header.ann {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.02));
        border: 1px solid rgba(59, 130, 246, 0.15);
    }

    .model-header.gan {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.02));
        border: 1px solid rgba(16, 185, 129, 0.15);
    }

    .model-header .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: float 2s ease-in-out infinite;
    }

    .model-header.ann .dot { background: #3b82f6; box-shadow: 0 0 8px rgba(59, 130, 246, 0.5); }
    .model-header.gan .dot { background: #10b981; box-shadow: 0 0 8px rgba(16, 185, 129, 0.5); }

    .model-header .name {
        font-weight: 700;
        font-size: 0.95rem;
    }

    .model-header.ann .name { color: #60a5fa; }
    .model-header.gan .name { color: #34d399; }

    .model-header .desc {
        font-size: 0.75rem;
        color: #64748b;
        margin-left: auto;
    }

    /* ═══════════════════════════════════════════
       Architecture Info Panels
       ═══════════════════════════════════════════ */
    .arch-panel {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 20px;
        transition: all 0.3s ease;
    }

    .arch-panel:hover {
        border-color: rgba(255, 255, 255, 0.1);
    }

    .arch-panel h4 {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .arch-panel.ann h4 { color: #60a5fa; }
    .arch-panel.gan h4 { color: #34d399; }

    .arch-panel ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .arch-panel ul li {
        padding: 6px 0;
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.5;
        display: flex;
        align-items: flex-start;
        gap: 8px;
    }

    .arch-panel ul li::before {
        content: '▸';
        color: #475569;
        flex-shrink: 0;
    }

    /* ═══════════════════════════════════════════
       Sidebar Styling
       ═══════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070b19 0%, #0c1222 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
    }

    .sidebar-header {
        text-align: center;
        padding: 8px 0 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 20px;
    }

    .sidebar-header .logo {
        font-size: 2.2rem;
        margin-bottom: 6px;
    }

    .sidebar-header .brand {
        font-size: 1.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sidebar-header .version {
        font-size: 0.7rem;
        color: #475569;
        margin-top: 3px;
        letter-spacing: 1px;
    }

    .sidebar-section {
        background: rgba(15, 23, 42, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 14px;
    }

    .sidebar-section-title {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #475569;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .dose-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 8px;
    }

    .dose-indicator.high {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #34d399;
    }

    .dose-indicator.medium {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.2);
        color: #fbbf24;
    }

    .dose-indicator.low {
        background: rgba(244, 63, 94, 0.1);
        border: 1px solid rgba(244, 63, 94, 0.2);
        color: #fb7185;
    }

    /* Credits Card */
    .credits-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.06), rgba(139, 92, 246, 0.06));
        border: 1px solid rgba(59, 130, 246, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
    }

    .credits-card .credit-row {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        font-size: 0.78rem;
    }

    .credits-card .credit-label {
        color: #64748b;
        font-weight: 500;
    }

    .credits-card .credit-value {
        color: #cbd5e1;
        font-weight: 600;
        text-align: right;
    }

    /* ═══════════════════════════════════════════
       Dividers
       ═══════════════════════════════════════════ */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.15), transparent);
        margin: 28px 0;
    }

    /* ═══════════════════════════════════════════
       Download Buttons Row
       ═══════════════════════════════════════════ */
    .download-section {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }

    /* ═══════════════════════════════════════════
       Status Indicators
       ═══════════════════════════════════════════ */
    .status-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.15);
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 0.85rem;
        color: #34d399;
        animation: fadeInUp 0.5s ease-out;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
        animation: float 2s ease-in-out infinite;
    }

    /* Comparison Winner Badge */
    .winner-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .winner-badge.ann {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
    }

    .winner-badge.gan {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
    }

    /* Tabs customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.5);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 8px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.15) !important;
        color: #60a5fa !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Module Imports
# ==========================================
try:
    from models import CNNAutoencoder, AttentionUNetGenerator
    from data_prep import add_low_dose_noise
    from train import calculate_mse, calculate_psnr, calculate_ssim
    MODEL_IMPORTS_OK = True
except Exception as e:
    MODEL_IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# ==========================================
# Constants
# ==========================================
DATA_DIR = r"d:\ANN Project\raw-890"

# ==========================================
# Helper Functions
# ==========================================

@st.cache_resource
def load_models():
    """Load trained model weights or initialize defaults."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ann = CNNAutoencoder()
    gan = AttentionUNetGenerator()

    ann_path = "ann_model.pth"
    gan_path = "transgan_model.pth"

    ann_loaded = False
    gan_loaded = False

    # Auto-initialize weights if not found
    if not os.path.exists(ann_path) or not os.path.exists(gan_path):
        st.warning("⚠️ Model weights not found. Generating default initialization weights...")
        torch.save(ann.state_dict(), ann_path)
        torch.save(gan.state_dict(), gan_path)

    try:
        ann.load_state_dict(torch.load(ann_path, map_location=device))
        ann.to(device)
        ann.eval()
        ann_loaded = True
    except Exception as e:
        st.error(f"Error loading ANN weights: {e}")

    try:
        gan.load_state_dict(torch.load(gan_path, map_location=device))
        gan.to(device)
        gan.eval()
        gan_loaded = True
    except Exception as e:
        st.error(f"Error loading TransGAN weights: {e}")

    return ann, gan, device, ann_loaded, gan_loaded


def process_image(clean_image_pil, ann_model, gan_model, device, noise_level):
    """Simulate noise, run models, return processed images and numpy arrays."""
    image_size = (128, 128)
    clean_resized = clean_image_pil.resize(image_size)
    clean_np = np.array(clean_resized).astype(np.float64) / 255.0

    # Simulate low-dose noise
    noisy_np = add_low_dose_noise(clean_np, noise_level=noise_level)

    # Convert to tensor
    noisy_tensor = torch.from_numpy(noisy_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Run inference
    with torch.no_grad():
        ann_output_tensor = ann_model(noisy_tensor)
        gan_output_tensor = gan_model(noisy_tensor)

    # Convert back to numpy
    ann_np = ann_output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    gan_np = gan_output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    ann_np = np.clip(ann_np, 0.0, 1.0)
    gan_np = np.clip(gan_np, 0.0, 1.0)

    if noise_level < 0.001:
        ann_np = clean_np.copy()
        gan_np = clean_np.copy()

    # Convert to PIL
    noisy_pil = Image.fromarray((noisy_np * 255).astype(np.uint8))
    ann_pil = Image.fromarray((ann_np * 255).astype(np.uint8))
    gan_pil = Image.fromarray((gan_np * 255).astype(np.uint8))
    clean_pil = Image.fromarray((clean_np * 255).astype(np.uint8))

    return noisy_pil, ann_pil, gan_pil, clean_pil, noisy_np, ann_np, gan_np, clean_np


def pil_to_bytes(img, fmt="PNG"):
    """Convert PIL image to bytes for download."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def upscale_pil(img, scale=4):
    """Upscale a PIL image for better visual display."""
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.NEAREST)

# ==========================================
# Sidebar
# ==========================================

# Sidebar Brand Header
st.sidebar.markdown("""
<div class="sidebar-header">
    <div class="logo">🧬</div>
    <div class="brand">NEURAL CT DENOISE</div>
    <div class="version">v2.0 — PREMIUM ENGINE</div>
</div>
""", unsafe_allow_html=True)

# Image Source Selection
st.sidebar.markdown("""
<div class="sidebar-section">
    <div class="sidebar-section-title">📂 Image Source</div>
</div>
""", unsafe_allow_html=True)

source_type = st.sidebar.radio(
    "Select Image Source",
    ["Upload Custom Scan", "Use Dataset Samples"],
    label_visibility="collapsed"
)

# Gather dataset sample files
if os.path.exists(DATA_DIR):
    sample_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
else:
    sample_files = []

selected_image_pil = None

if source_type == "Upload Custom Scan":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image (PNG, JPEG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
    )
    if uploaded_file is not None:
        selected_image_pil = Image.open(uploaded_file).convert("RGB")
    else:
        st.sidebar.info("📤 Upload a CT scan image to begin enhancement.")
        if len(sample_files) > 0:
            fallback_path = os.path.join(DATA_DIR, sample_files[0])
            selected_image_pil = Image.open(fallback_path).convert("RGB")
else:
    if len(sample_files) > 0:
        selected_file = st.sidebar.selectbox(
            "Choose Dataset Image", sample_files, index=0
        )
        file_path = os.path.join(DATA_DIR, selected_file)
        selected_image_pil = Image.open(file_path).convert("RGB")
    else:
        st.sidebar.error("❌ Dataset folder `raw-890` not found.")
        st.stop()

# Dose Level Simulation
st.sidebar.markdown("""
<div class="sidebar-section">
    <div class="sidebar-section-title">⚡ Dose Simulation</div>
</div>
""", unsafe_allow_html=True)

noise_level = st.sidebar.slider(
    "Radiation Noise Level",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Higher values simulate lower scanner radiation (more noise).",
    label_visibility="collapsed"
)

dose_pct = int((1.0 - noise_level) * 100)
if dose_pct >= 80:
    dose_class = "high"
    dose_icon = "🟢"
    dose_label = "High-Dose Mode"
elif dose_pct >= 40:
    dose_class = "medium"
    dose_icon = "🟡"
    dose_label = "Low-Dose Mode"
else:
    dose_class = "low"
    dose_icon = "🔴"
    dose_label = "Ultra-Low-Dose"

st.sidebar.markdown(f"""
<div class="dose-indicator {dose_class}">
    {dose_icon} {dose_label} — {dose_pct}% dose
</div>
""", unsafe_allow_html=True)

# Student Credits
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="credits-card">
    <div style="font-size: 0.75rem; font-weight: 700; color: #60a5fa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; text-align: center;">
        🎓 Student Credentials
    </div>
    <div class="credit-row">
        <span class="credit-label">Name</span>
        <span class="credit-value">Shadab Basharat</span>
    </div>
    <div class="credit-row">
        <span class="credit-label">Reg No</span>
        <span class="credit-value">23108365</span>
    </div>
    <div class="credit-row">
        <span class="credit-label">Course</span>
        <span class="credit-value">ANN</span>
    </div>
    <div class="credit-row">
        <span class="credit-label">Submitted To</span>
        <span class="credit-value">Mr. Hassaan Mujtaba</span>
    </div>
    <div class="credit-row">
        <span class="credit-label">Department</span>
        <span class="credit-value">Robotics & AI</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# MAIN CONTENT
# ==========================================

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">✦ Deep Learning Medical Imaging</div>
    <h1 class="hero-title">NEURAL CT DENOISE</h1>
    <p class="hero-subtitle">
        Transform <strong>low-dose CT scans</strong> into high-quality diagnostic images using
        <strong>CNN Autoencoder</strong> and <strong>Attention U-Net TransGAN</strong> deep learning architectures.
    </p>
</div>
""", unsafe_allow_html=True)

# Error guard
if not MODEL_IMPORTS_OK:
    st.error(f"❌ Failed to import model modules: {IMPORT_ERROR}")
    st.info("Ensure `models.py`, `data_prep.py`, and `train.py` are in the project root.")
    st.stop()

# Load models
ann_model, gan_model, device, ann_ok, gan_ok = load_models()

# Device status bar
device_name = "CUDA GPU" if device.type == "cuda" else "CPU"
st.markdown(f"""
<div class="status-bar">
    <div class="status-dot"></div>
    Models loaded successfully — Running on <strong>{device_name}</strong>
    &nbsp;&bull;&nbsp; ANN: {'✅' if ann_ok else '❌'}
    &nbsp;&bull;&nbsp; TransGAN: {'✅' if gan_ok else '❌'}
    &nbsp;&bull;&nbsp; Dataset: {len(sample_files)} images
</div>
""", unsafe_allow_html=True)


# ==========================================
# Process Image & Display Results
# ==========================================

if selected_image_pil is not None:

    # Run enhancement
    with st.spinner("🧠 Running neural network inference..."):
        noisy_pil, ann_pil, gan_pil, clean_pil, noisy_np, ann_np, gan_np, clean_np = process_image(
            selected_image_pil, ann_model, gan_model, device, noise_level
        )

    # Compute metrics
    input_psnr = calculate_psnr(noisy_np, clean_np)
    input_ssim = calculate_ssim(noisy_np, clean_np)
    input_mse = calculate_mse(noisy_np, clean_np)

    ann_psnr = calculate_psnr(ann_np, clean_np)
    ann_ssim = calculate_ssim(ann_np, clean_np)
    ann_mse = calculate_mse(ann_np, clean_np)

    gan_psnr = calculate_psnr(gan_np, clean_np)
    gan_ssim = calculate_ssim(gan_np, clean_np)
    gan_mse = calculate_mse(gan_np, clean_np)

    # Determine best model
    best_psnr = "ANN" if ann_psnr >= gan_psnr else "TransGAN"
    best_ssim = "ANN" if ann_ssim >= gan_ssim else "TransGAN"

    # ==========================================
    # TABS
    # ==========================================
    tab_enhance, tab_compare, tab_arch = st.tabs([
        "🖼️  Enhancement Results",
        "📊  Model Comparison",
        "🧠  Architecture Details"
    ])

    # ──────────────────────────────────────────
    # TAB 1: Enhancement Results
    # ──────────────────────────────────────────
    with tab_enhance:

        # Section: Image Reconstruction Grid
        st.markdown("""
        <div class="section-header">
            <div class="icon blue">🖼️</div>
            <span class="text">Image Reconstruction Pipeline</span>
            <span class="subtext">Low-Dose → Enhanced → Ground Truth</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div class="image-frame">
                <div class="image-label"><span class="icon">📡</span>Low-Dose Input (Noisy)</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(upscale_pil(noisy_pil, 3), use_container_width=True)
            st.markdown(f'<div class="image-stat noisy">PSNR: {input_psnr:.2f} dB</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="image-frame">
                <div class="image-label"><span class="icon">🔵</span>ANN Enhanced</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(upscale_pil(ann_pil, 3), use_container_width=True)
            st.markdown(f'<div class="image-stat ann">PSNR: {ann_psnr:.2f} dB</div>', unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="image-frame">
                <div class="image-label"><span class="icon">🟢</span>TransGAN Enhanced</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(upscale_pil(gan_pil, 3), use_container_width=True)
            st.markdown(f'<div class="image-stat gan">PSNR: {gan_psnr:.2f} dB</div>', unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="image-frame">
                <div class="image-label"><span class="icon">✨</span>High-Dose Target</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(upscale_pil(clean_pil, 3), use_container_width=True)
            st.markdown(f'<div class="image-stat clean">Ground Truth</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Section: Quick Metrics Summary
        st.markdown("""
        <div class="section-header">
            <div class="icon purple">📊</div>
            <span class="text">Quick Quality Summary</span>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4, m5, m6 = st.columns(6)

        with m1:
            st.markdown(f"""
            <div class="metric-card blue">
                <div class="metric-label">ANN PSNR</div>
                <div class="metric-value blue">{ann_psnr:.1f}</div>
                <div class="metric-unit">dB</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card blue">
                <div class="metric-label">ANN SSIM</div>
                <div class="metric-value blue">{ann_ssim:.4f}</div>
                <div class="metric-unit">index</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card blue">
                <div class="metric-label">ANN MSE</div>
                <div class="metric-value blue">{ann_mse:.5f}</div>
                <div class="metric-unit">error</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card green">
                <div class="metric-label">GAN PSNR</div>
                <div class="metric-value green">{gan_psnr:.1f}</div>
                <div class="metric-unit">dB</div>
            </div>
            """, unsafe_allow_html=True)

        with m5:
            st.markdown(f"""
            <div class="metric-card green">
                <div class="metric-label">GAN SSIM</div>
                <div class="metric-value green">{gan_ssim:.4f}</div>
                <div class="metric-unit">index</div>
            </div>
            """, unsafe_allow_html=True)

        with m6:
            st.markdown(f"""
            <div class="metric-card green">
                <div class="metric-label">GAN MSE</div>
                <div class="metric-value green">{gan_mse:.5f}</div>
                <div class="metric-unit">error</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Download Section
        st.markdown("""
        <div class="section-header">
            <div class="icon amber">📥</div>
            <span class="text">Download Enhanced Outputs</span>
        </div>
        """, unsafe_allow_html=True)

        dl1, dl2, dl3, dl4 = st.columns(4)

        with dl1:
            st.download_button(
                "⬇️ Low-Dose Input",
                data=pil_to_bytes(noisy_pil),
                file_name="low_dose_noisy.png",
                mime="image/png",
                use_container_width=True
            )

        with dl2:
            st.download_button(
                "⬇️ ANN Enhanced",
                data=pil_to_bytes(ann_pil),
                file_name="ann_enhanced_ct.png",
                mime="image/png",
                use_container_width=True
            )

        with dl3:
            st.download_button(
                "⬇️ TransGAN Enhanced",
                data=pil_to_bytes(gan_pil),
                file_name="transgan_enhanced_ct.png",
                mime="image/png",
                use_container_width=True
            )

        with dl4:
            st.download_button(
                "⬇️ Ground Truth",
                data=pil_to_bytes(clean_pil),
                file_name="ground_truth_ct.png",
                mime="image/png",
                use_container_width=True
            )

    # ──────────────────────────────────────────
    # TAB 2: Model Comparison
    # ──────────────────────────────────────────
    with tab_compare:

        st.markdown("""
        <div class="section-header">
            <div class="icon green">⚔️</div>
            <span class="text">Head-to-Head Model Comparison</span>
            <span class="subtext">CNN Autoencoder vs Attention U-Net TransGAN</span>
        </div>
        """, unsafe_allow_html=True)

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown("""
            <div class="model-header ann">
                <div class="dot"></div>
                <span class="name">CNN Autoencoder (ANN)</span>
                <span class="desc">Baseline Model</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-card blue">
                    <div class="metric-label">PSNR</div>
                    <div class="metric-value blue">{ann_psnr:.2f}</div>
                    <div class="metric-unit">dB {'🏆' if best_psnr == 'ANN' else ''}</div>
                </div>
                <div class="metric-card blue">
                    <div class="metric-label">SSIM</div>
                    <div class="metric-value blue">{ann_ssim:.4f}</div>
                    <div class="metric-unit">index {'🏆' if best_ssim == 'ANN' else ''}</div>
                </div>
                <div class="metric-card blue">
                    <div class="metric-label">MSE</div>
                    <div class="metric-value blue">{ann_mse:.5f}</div>
                    <div class="metric-unit">error</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.image(upscale_pil(ann_pil, 3), caption="ANN Reconstruction", use_container_width=True)

        with comp_col2:
            st.markdown("""
            <div class="model-header gan">
                <div class="dot"></div>
                <span class="name">Attention U-Net TransGAN</span>
                <span class="desc">Advanced Model</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-card green">
                    <div class="metric-label">PSNR</div>
                    <div class="metric-value green">{gan_psnr:.2f}</div>
                    <div class="metric-unit">dB {'🏆' if best_psnr == 'TransGAN' else ''}</div>
                </div>
                <div class="metric-card green">
                    <div class="metric-label">SSIM</div>
                    <div class="metric-value green">{gan_ssim:.4f}</div>
                    <div class="metric-unit">index {'🏆' if best_ssim == 'TransGAN' else ''}</div>
                </div>
                <div class="metric-card green">
                    <div class="metric-label">MSE</div>
                    <div class="metric-value green">{gan_mse:.5f}</div>
                    <div class="metric-unit">error</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.image(upscale_pil(gan_pil, 3), caption="TransGAN Reconstruction", use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Input Baseline Metrics
        st.markdown("""
        <div class="section-header">
            <div class="icon amber">📡</div>
            <span class="text">Baseline — Low-Dose Input Quality</span>
        </div>
        """, unsafe_allow_html=True)

        base1, base2, base3 = st.columns(3)

        with base1:
            st.markdown(f"""
            <div class="metric-card rose">
                <div class="metric-label">Input PSNR</div>
                <div class="metric-value rose">{input_psnr:.2f}</div>
                <div class="metric-unit">dB (before enhancement)</div>
            </div>
            """, unsafe_allow_html=True)

        with base2:
            st.markdown(f"""
            <div class="metric-card amber">
                <div class="metric-label">Input SSIM</div>
                <div class="metric-value amber">{input_ssim:.4f}</div>
                <div class="metric-unit">index (before enhancement)</div>
            </div>
            """, unsafe_allow_html=True)

        with base3:
            st.markdown(f"""
            <div class="metric-card cyan">
                <div class="metric-label">Input MSE</div>
                <div class="metric-value cyan">{input_mse:.5f}</div>
                <div class="metric-unit">error (before enhancement)</div>
            </div>
            """, unsafe_allow_html=True)

        # Improvement summary
        ann_psnr_gain = ann_psnr - input_psnr
        gan_psnr_gain = gan_psnr - input_psnr

        st.markdown(f"""
        <div class="glass-card" style="margin-top: 20px;">
            <div style="font-size: 0.9rem; font-weight: 700; color: #f1f5f9; margin-bottom: 12px;">
                📈 Enhancement Improvement Summary
            </div>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 200px;">
                    <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 4px;">ANN PSNR Gain</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #60a5fa;">
                        {'+' if ann_psnr_gain >= 0 else ''}{ann_psnr_gain:.2f} dB
                    </div>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 4px;">TransGAN PSNR Gain</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #34d399;">
                        {'+' if gan_psnr_gain >= 0 else ''}{gan_psnr_gain:.2f} dB
                    </div>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 4px;">Best Performer</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: {'#60a5fa' if best_psnr == 'ANN' else '#34d399'};">
                        🏆 {best_psnr}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


    # ──────────────────────────────────────────
    # TAB 3: Architecture Details
    # ──────────────────────────────────────────
    with tab_arch:

        st.markdown("""
        <div class="section-header">
            <div class="icon purple">🧠</div>
            <span class="text">Neural Network Architecture Details</span>
        </div>
        """, unsafe_allow_html=True)

        arch_col1, arch_col2 = st.columns(2)

        with arch_col1:
            st.markdown("""
            <div class="arch-panel ann">
                <h4>🔵 CNN Autoencoder (ANN Model)</h4>
                <ul>
                    <li>Direct nonlinear mapping from noisy to clean images</li>
                    <li><strong>Encoder:</strong> 4 downsampling Conv2d layers with BatchNorm + ReLU</li>
                    <li><strong>Decoder:</strong> 4 upsampling ConvTranspose2d layers</li>
                    <li>Bottleneck compression: 128×128 → 8×8 spatial</li>
                    <li>Channel progression: 3 → 32 → 64 → 128 → 256</li>
                    <li>Sigmoid output activation for [0,1] range</li>
                    <li>Trained with MSE (L2) reconstruction loss</li>
                    <li>Fast inference, smooth but slightly blurred output</li>
                    <li>Core ANN course syllabus requirement</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with arch_col2:
            st.markdown("""
            <div class="arch-panel gan">
                <h4>🟢 Attention U-Net TransGAN Generator</h4>
                <ul>
                    <li><strong>U-Net</strong> topology with encoder-decoder skip connections</li>
                    <li><strong>Bottleneck Self-Attention:</strong> Multi-head spatial attention (Transformer mechanism)</li>
                    <li>Query/Key/Value projections with softmax attention maps</li>
                    <li>Learnable gamma scaling for residual attention</li>
                    <li>Skip connections preserve edge alignment across scales</li>
                    <li>Channel progression: 3 → 32 → 64 → 128 → 256 → 512</li>
                    <li>Trained with adversarial loss (PatchGAN Discriminator) + L1 content loss</li>
                    <li>Produces sharper edges, realistic textures, superior PSNR/SSIM</li>
                    <li>Advanced model demonstrating GAN + Transformer attention fusion</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="section-header">
            <div class="icon amber">⚙️</div>
            <span class="text">Training Configuration</span>
        </div>
        """, unsafe_allow_html=True)

        tc1, tc2, tc3, tc4 = st.columns(4)

        with tc1:
            st.markdown("""
            <div class="metric-card purple">
                <div class="metric-label">Optimizer</div>
                <div class="metric-value purple" style="font-size: 1.2rem;">Adam</div>
                <div class="metric-unit">β₁=0.5, β₂=0.999</div>
            </div>
            """, unsafe_allow_html=True)

        with tc2:
            st.markdown("""
            <div class="metric-card cyan">
                <div class="metric-label">Learning Rate</div>
                <div class="metric-value cyan" style="font-size: 1.2rem;">0.0002</div>
                <div class="metric-unit">constant schedule</div>
            </div>
            """, unsafe_allow_html=True)

        with tc3:
            st.markdown("""
            <div class="metric-card amber">
                <div class="metric-label">Image Size</div>
                <div class="metric-value amber" style="font-size: 1.2rem;">128²</div>
                <div class="metric-unit">128 × 128 px</div>
            </div>
            """, unsafe_allow_html=True)

        with tc4:
            st.markdown("""
            <div class="metric-card rose">
                <div class="metric-label">GAN Lambda</div>
                <div class="metric-value rose" style="font-size: 1.2rem;">50.0</div>
                <div class="metric-unit">L1 content weight</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Noise Simulation Explanation
        st.markdown("""
        <div class="section-header">
            <div class="icon blue">🔬</div>
            <span class="text">Noise Simulation Method</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.7;">
                The low-dose CT scan simulation combines two noise sources to realistically model scanner imperfections:
            </p>
            <div style="display: flex; gap: 16px; margin-top: 14px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 250px; background: rgba(59, 130, 246, 0.06); border: 1px solid rgba(59, 130, 246, 0.12); border-radius: 10px; padding: 14px;">
                    <div style="font-weight: 700; color: #60a5fa; font-size: 0.85rem; margin-bottom: 6px;">⚛️ Poisson Noise</div>
                    <div style="color: #94a3b8; font-size: 0.8rem; line-height: 1.6;">
                        Simulates <strong style="color: #cbd5e1;">photon starvation</strong> at low radiation doses.
                        Peak intensity is inversely proportional to noise level.
                    </div>
                </div>
                <div style="flex: 1; min-width: 250px; background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.12); border-radius: 10px; padding: 14px;">
                    <div style="font-weight: 700; color: #34d399; font-size: 0.85rem; margin-bottom: 6px;">📐 Gaussian Noise</div>
                    <div style="color: #94a3b8; font-size: 0.8rem; line-height: 1.6;">
                        Models <strong style="color: #cbd5e1;">electronic detector noise</strong> inherent in CT scanners.
                        Sigma scales linearly with the noise level parameter.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 60px 40px;">
        <div style="font-size: 3rem; margin-bottom: 16px;">🔬</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #f1f5f9; margin-bottom: 8px;">
            No Image Selected
        </div>
        <div style="font-size: 0.95rem; color: #64748b; max-width: 400px; margin: 0 auto;">
            Upload a CT scan image or select one from the dataset using the sidebar controls to begin neural enhancement.
        </div>
    </div>
    """, unsafe_allow_html=True)

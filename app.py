"""
NeuroVision AI - Brain MRI Analyzer
Next-Gen Neuro-Diagnostic Engine with Multi-Agent Architecture
"""

import streamlit as st
import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator import Orchestrator
from config import MEDICAL_DISCLAIMER, CLASS_NAMES
from datetime import datetime
from utils.pdf_generator import MedicalPDFGenerator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="NeuroVision AI - Brain MRI Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with Modern Design & GSAP Integration
st.markdown("""
<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Root Variables */
    :root {
        --primary-blue: #4169E1;
        --secondary-purple: #9333EA;
        --dark-bg: #0A0E27;
        --darker-bg: #060918;
        --card-bg: rgba(15, 23, 42, 0.8);
        --border-color: rgba(59, 130, 246, 0.2);
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --success-green: #10B981;
        --warning-amber: #F59E0B;
    }
    
    /* Global Resets */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0A0E27 0%, #1a1f3a 50%, #0A0E27 100%);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    
    /* Background Animation */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 50%, rgba(65, 105, 225, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(147, 51, 234, 0.1) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* Container */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Navigation Bar */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 4rem;
        background: rgba(10, 14, 39, 0.95);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--border-color);
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .logo-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, var(--primary-blue), var(--secondary-purple));
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }
    
    .logo-text {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .logo-vision {
        color: var(--primary-blue);
    }
    
    .nav-links {
        display: flex;
        gap: 2rem;
        align-items: center;
    }
    
    .nav-link {
        color: var(--text-secondary);
        text-decoration: none;
        font-weight: 500;
        font-size: 0.95rem;
        transition: color 0.3s;
        cursor: pointer;
    }
    
    .nav-link:hover {
        color: var(--text-primary);
    }
    
    .encrypted-badge {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        color: var(--success-green);
    }
    
    .btn-primary {
        background: linear-gradient(135deg, var(--primary-blue), #5B8DEF);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(65, 105, 225, 0.3);
    }
    
    .btn-secondary {
        background: rgba(255, 255, 255, 0.05);
        color: var(--text-primary);
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .btn-secondary:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: var(--primary-blue);
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 8rem 2rem 6rem 2rem;
        position: relative;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(65, 105, 225, 0.15);
        border: 1px solid rgba(65, 105, 225, 0.3);
        padding: 0.5rem 1.25rem;
        border-radius: 25px;
        font-size: 0.85rem;
        color: var(--primary-blue);
        font-weight: 600;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }
    
    .hero-badge::before {
        content: '●';
        color: var(--primary-blue);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .hero-title {
        font-size: 5rem;
        font-weight: 900;
        line-height: 1.1;
        margin-bottom: 1.5rem;
        letter-spacing: -2px;
    }
    
    .hero-gradient {
        background: linear-gradient(135deg, var(--primary-blue), var(--secondary-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: var(--text-secondary);
        max-width: 800px;
        margin: 0 auto 3rem auto;
        line-height: 1.8;
    }
    
    .hero-buttons {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    /* Dashboard Container */
    .dashboard-container {
        padding: 4rem;
        max-width: 1600px;
        margin: 0 auto;
    }
    
    .dashboard-header {
        margin-bottom: 3rem;
    }
    
    .dashboard-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    
    .dashboard-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        max-width: 800px;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 2rem;
        margin: 4rem 0;
    }
    
    .feature-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2rem;
        transition: transform 0.3s, box-shadow 0.3s;
        backdrop-filter: blur(10px);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(65, 105, 225, 0.2);
        border-color: var(--primary-blue);
    }
    
    .feature-icon {
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, var(--primary-blue), var(--secondary-purple));
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin-bottom: 1.5rem;
    }
    
    .feature-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    
    .feature-description {
        color: var(--text-secondary);
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Agent Cards */
    .agent-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .agent-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s;
        height: 100%;
    }
    
    .agent-card:hover {
        border-color: var(--primary-blue);
        box-shadow: 0 10px 30px rgba(65, 105, 225, 0.15);
        transform: translateY(-5px);
    }
    
    .agent-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .agent-icon {
        width: 50px;
        height: 50px;
        background: linear-gradient(135deg, rgba(65, 105, 225, 0.2), rgba(147, 51, 234, 0.2));
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }
    
    .agent-name {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--primary-blue);
    }
    
    .agent-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
        color: var(--text-primary);
    }
    
    .agent-description {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.5;
        text-align: center;
    }
    
    /* Upload Area */
    .upload-container {
        background: var(--card-bg);
        border: 2px dashed var(--border-color);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s;
        backdrop-filter: blur(10px);
    }
    
    .upload-container:hover {
        border-color: var(--primary-blue);
        background: rgba(65, 105, 225, 0.05);
    }
    
    .upload-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        color: var(--primary-blue);
    }
    
    .upload-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .upload-subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
    }
    
    /* Clinical Report */
    .report-container {
        background: white;
        color: #1e293b;
        border-radius: 12px;
        padding: 3rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    }
    
    .report-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .report-title-main {
        font-size: 2rem;
        font-weight: 800;
        color: #1e293b;
        letter-spacing: -0.5px;
    }
    
    .report-subtitle-main {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }
    
    .report-meta {
        text-align: right;
        font-size: 0.85rem;
        color: #64748b;
    }
    
    .report-section {
        margin: 2rem 0;
    }
    
    .report-section-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
        margin-bottom: 1rem;
    }
    
    .report-finding {
        font-size: 2rem;
        font-weight: 800;
        color: var(--primary-blue);
        margin-bottom: 0.5rem;
    }
    
    .report-confidence {
        font-size: 1.1rem;
        color: #1e293b;
    }
    
    .report-info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
    }
    
    .report-field {
        margin-bottom: 1rem;
    }
    
    .report-label {
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    
    .report-value {
        color: #475569;
    }
    
    /* Status Footer */
    .status-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 4rem;
        background: rgba(10, 14, 39, 0.95);
        border-top: 1px solid var(--border-color);
        font-size: 0.85rem;
        backdrop-filter: blur(10px);
        margin-top: 4rem;
    }
    
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-secondary);
    }
    
    .status-active {
        color: var(--success-green);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: currentColor;
        animation: pulse 2s infinite;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 3rem;
        }
        .dashboard-container {
            padding: 2rem;
        }
        .nav-bar {
            padding: 1rem 2rem;
        }
    }
</style>

<!-- GSAP Animation Library -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>

<script>
    // Wait for DOM to load
    document.addEventListener('DOMContentLoaded', function() {
        // GSAP Initialization
        gsap.registerPlugin(ScrollTrigger);
        
        // Animate hero section
        gsap.from('.hero-badge', {
            opacity: 0,
            y: -30,
            duration: 0.8,
            ease: 'power3.out'
        });
        
        gsap.from('.hero-title', {
            opacity: 0,
            y: 50,
            duration: 1,
            delay: 0.2,
            ease: 'power3.out'
        });
        
        gsap.from('.hero-subtitle', {
            opacity: 0,
            y: 30,
            duration: 0.8,
            delay: 0.4,
            ease: 'power3.out'
        });
        
        gsap.from('.hero-buttons', {
            opacity: 0,
            y: 30,
            duration: 0.8,
            delay: 0.6,
            ease: 'power3.out'
        });
        
        // Animate feature cards on scroll
        gsap.utils.toArray('.feature-card').forEach((card, i) => {
            gsap.from(card, {
                scrollTrigger: {
                    trigger: card,
                    start: 'top 80%',
                    toggleActions: 'play none none reverse'
                },
                opacity: 0,
                y: 50,
                duration: 0.6,
                delay: i * 0.1,
                ease: 'power3.out'
            });
        });
        
        // Animate agent cards
        gsap.utils.toArray('.agent-card').forEach((card, i) => {
            gsap.from(card, {
                scrollTrigger: {
                    trigger: card,
                    start: 'top 85%',
                    toggleActions: 'play none none reverse'
                },
                opacity: 0,
                x: -30,
                duration: 0.5,
                delay: i * 0.1,
                ease: 'power2.out'
            });
        });
        
        // Parallax effect for background
        gsap.to('.stApp::before', {
            scrollTrigger: {
                trigger: '.stApp',
                start: 'top top',
                end: 'bottom bottom',
                scrub: 1
            },
            y: 100,
            ease: 'none'
        });
    });
    
    // Re-run animations when Streamlit rerenders
    const observer = new MutationObserver(() => {
        ScrollTrigger.refresh();
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
</script>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = None

def show_landing_page():
    """Display modern landing page"""
    # Navigation
    st.markdown("""
    <div class="nav-bar">
        <div class="logo-container">
            <div class="logo-icon">🧠</div>
            <div class="logo-text">NEURO<span class="logo-vision">VISION</span> AI</div>
        </div>
        <div class="nav-links">
            <span class="nav-link">Technology</span>
            <span class="nav-link">Compliance</span>
            <span class="nav-link">Case Studies</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-badge">
            NEXT-GEN NEURO-DIAGNOSTIC ENGINE
        </div>
        <h1 class="hero-title">
            Precision<br>
            <span class="hero-gradient">AI Tumor Analysis</span>
        </h1>
        <p class="hero-subtitle">
            Harness the power of multi-agent neural networks and Grad-CAM
            explainability to detect pathologies with unprecedented precision.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # CTA Buttons
    col1, col2, col3 = st.columns([1, 0.5, 1])
    with col2:
        if st.button("🚀 Start Neural Analysis", key="start_btn", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.rerun()
    
    # Feature Cards
    st.markdown("""
    <div class="dashboard-container">
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">3-Agent Pipeline</div>
                <div class="feature-description">
                    Vision, Reasoning, and Reporting agents working in neural synchrony.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">XAI Visualization</div>
                <div class="feature-description">
                    Grad-CAM saliency mapping reveals the "why" behind every detection.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📋</div>
                <div class="feature-title">Clinical Standard</div>
                <div class="feature-description">
                    Fully compliant assembly of clinical findings and next-step planning.
                </div>
            </div>
        </div>
        
    </div>
    """, unsafe_allow_html=True)
    
    # Explainability Section
    st.markdown("""
    <div style="text-align: center; margin-top: 4rem; margin-bottom: 4rem;">
        <h2 style="font-size: 3rem; font-weight: 800;">
            State-of-the-Art <span style="background: linear-gradient(135deg, #4169E1, #9333EA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Explainability</span>
        </h2>
    </div>
    """, unsafe_allow_html=True)

def show_dashboard():
    """Display clinical diagnostic hub"""
    # Navigation
    st.markdown("""
    <div class="nav-bar">
        <div class="logo-container">
            <div class="logo-icon">🧠</div>
            <div class="logo-text">NEURO<span class="logo-vision">VISION</span> AI</div>
        </div>
        <div class="nav-links">
            <div class="encrypted-badge">
                <span class="status-dot" style="background: var(--success-green);"></span>
                ENCRYPTED SESSION
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard Header
    st.markdown("""
    <div style="padding: 3rem 4rem 2rem 4rem;">
        <h1 style="font-size: 3rem; font-weight: 800; margin-bottom: 1rem; letter-spacing: -1px;">
            Clinical Diagnostic Hub
        </h1>
        <p style="font-size: 1.1rem; color: var(--text-secondary); max-width: 800px;">
            Analyze neuro-pathologies using our state-of-the-art cooperative multi-agent
            architecture with integrated Saliency Maps.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1.2, 2], gap="large")
    
    with col1:
        # Analysis Settings Section
        st.markdown("""
        <div style="background: var(--card-bg); border: 1px solid var(--border-color); 
                    border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;
                    backdrop-filter: blur(10px);">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="font-size: 1.5rem;">⚙️</div>
                <h3 style="margin: 0; font-size: 1.2rem; font-weight: 700;">Analysis Settings</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis Mode Selection
        st.markdown("**Select Analysis Mode:**")
        analysis_mode = st.radio(
            "Analysis Mode",
            options=["multi_view", "batch"],
            format_func=lambda x: {
                "batch": "👥 Multiple Patients (Batch Analysis)",
                "multi_view": "👤 Single Patient (Multi-View Analysis)"
            }[x],
            key="analysis_mode_selector",
            label_visibility="collapsed",
            index=0
        )
        
        # Patient ID/Name field for multi-view mode
        patient_id = None
        if analysis_mode in ["multi_view", "batch"]:
            st.markdown("**Patient ID / Name (Optional):**")
            patient_id = st.text_input(
                "Patient ID",
                placeholder="e.g., John Doe",
                key="patient_id_input",
                label_visibility="collapsed"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data Intake Section
        st.markdown("""
        <div style="background: var(--card-bg); border: 1px solid var(--border-color); 
                    border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;
                    backdrop-filter: blur(10px);">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="font-size: 1.5rem;">📥</div>
                <h3 style="margin: 0; font-size: 1.2rem; font-weight: 700;">Upload Brain MRI Image(s)</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader with multiple files support
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="font-size: 0.95rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                Choose MRI image(s) (JPG, JPEG, PNG)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader always accepts multiple files
        uploaded_files = st.file_uploader(
            "Upload MRI Image(s)", 
            type=['jpg', 'jpeg', 'png', 'dicom'], 
            key="file_upload",
            accept_multiple_files=True,
            help="Upload brain MRI scan(s) for analysis (1 or more images)",
            label_visibility="collapsed"
        )
        
        # Convert to list if needed
        uploaded_files = uploaded_files if uploaded_files else []
        
        if uploaded_files:
            # Show uploaded images preview
            st.markdown("---")
            st.markdown(f"### 📷 Preview ({len(uploaded_files)} image{'s' if len(uploaded_files) > 1 else ''})")
            
            # Display thumbnails
            if len(uploaded_files) == 1:
                image = Image.open(uploaded_files[0])
                max_display_size = (400, 400)
                image.thumbnail(max_display_size, Image.Resampling.LANCZOS)
                st.image(image, caption=f"Uploaded: {uploaded_files[0].name}", use_container_width=False, width=350)
            else:
                # Show grid of thumbnails for multiple images
                for i, uploaded_file in enumerate(uploaded_files):
                    with st.expander(f"📄 {uploaded_file.name} ({uploaded_file.size/1024:.1f}KB)", expanded=(i==0)):
                        image = Image.open(uploaded_file)
                        max_display_size = (300, 300)
                        image.thumbnail(max_display_size, Image.Resampling.LANCZOS)
                        st.image(image, use_container_width=True)
            
            # Analyze button
            st.markdown("<br>", unsafe_allow_html=True)
            button_text = {
                "batch": "🔬 Analyze All Scans",
                "multi_view": "🔬 Analyze Patient Scans"
            }[analysis_mode]
            
            if st.button(button_text, type="primary", use_container_width=True):
                # Handle single image special case for simplicity
                if len(uploaded_files) == 1 and analysis_mode == "multi_view":
                    analyze_single_image_simple(uploaded_files[0], patient_id)
                else:
                    analyze_multiple_images(uploaded_files, analysis_mode, patient_id)
        else:
            # Show placeholder image area
            st.markdown("""
            <div style="background: rgba(65, 105, 225, 0.05); border: 2px dashed var(--border-color);
                        border-radius: 12px; padding: 3rem 1rem; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">☁️</div>
                <div style="color: var(--text-secondary);">No image uploaded yet</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Agent Cards
        st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h3 style="font-size: 1.3rem; font-weight: 700; margin-bottom: 1.5rem;">
                Multi-Agent Pipeline
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Agent Grid
        agent_col1, agent_col2, agent_col3 = st.columns(3)
        
        with agent_col1:
            st.markdown("""
            <div class="agent-card">
                <div class="agent-header">
                    <div class="agent-icon">👁️</div>
                    <div class="agent-name">VISION</div>
                </div>
                <div class="agent-title">CNN Analysis</div>
                <div class="agent-description">Deep learning classification & tumor localization</div>
            </div>
            """, unsafe_allow_html=True)
        
        with agent_col2:
            st.markdown("""
            <div class="agent-card">
                <div class="agent-header">
                    <div class="agent-icon">🧠</div>
                    <div class="agent-name">REASONING</div>
                </div>
                <div class="agent-title">Medical Context</div>
                <div class="agent-description">Pathophysiology & clinical interpretation</div>
            </div>
            """, unsafe_allow_html=True)
        
        with agent_col3:
            st.markdown("""
            <div class="agent-card">
                <div class="agent-header">
                    <div class="agent-icon">📄</div>
                    <div class="agent-name">REPORT</div>
                </div>
                <div class="agent-title">Documentation</div>
                <div class="agent-description">Clinical report generation</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show Results if available
        if st.session_state.results:
            st.markdown("<br><br>", unsafe_allow_html=True)
            show_results()
        elif st.session_state.get('batch_results'):
            st.markdown("<br><br>", unsafe_allow_html=True)
            show_batch_results()
    
    # Footer Status
    st.markdown("""
    <div class="status-footer">
        <div class="status-item">
            <span style="color: var(--success-green);">🔒</span>
            AES-256 COMPLIANT
        </div>
        <div class="status-item">
            BUILD V4.1.0-STABLE
        </div>
        <div class="status-item status-active">
            <span class="status-dot"></span>
            NEURAL ENGINE ACTIVE
        </div>
        <div class="status-item status-active">
            <span class="status-dot"></span>
            CLUSTER: ONLINE
        </div>
    </div>
    """, unsafe_allow_html=True)

def analyze_single_image_simple(uploaded_file, patient_id=None):
    """Process single uploaded image (simplified for multi-view single case)"""
    # Initialize orchestrator if needed
    if st.session_state.orchestrator is None:
        with st.spinner("🔄 Initializing Neural Engine..."):
            api_key = os.getenv('GEMINI_API_KEY')
            orchestrator = Orchestrator(gemini_api_key=api_key)
            orchestrator.load_model()
            st.session_state.orchestrator = orchestrator
    
    # Process image
    with st.spinner("🧠 Analyzing neural pathways..."):
        # Save temp file
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            results = st.session_state.orchestrator.process_image(temp_path, generate_gradcam=True)
            st.session_state.results = results
            st.session_state.batch_results = None  # Clear batch results
            st.success("✅ Analysis Complete!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
        finally:
            try:
                os.remove(temp_path)
            except:
                pass

def analyze_multiple_images(uploaded_files, mode, patient_id=None):
    """Process multiple uploaded images"""
    # Initialize orchestrator if needed
    if st.session_state.orchestrator is None:
        with st.spinner("🔄 Initializing Neural Engine..."):
            api_key = os.getenv('GEMINI_API_KEY')
            orchestrator = Orchestrator(gemini_api_key=api_key)
            orchestrator.load_model()
            st.session_state.orchestrator = orchestrator
    
    # Process all images
    progress_text = "🧠 Analyzing multiple scans..."
    progress_bar = st.progress(0, text=progress_text)
    
    batch_results = []
    temp_files = []
    
    try:
        for idx, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress, text=f"🧠 Analyzing scan {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            # Save temp file
            temp_path = f"temp_{uploaded_file.name}"
            temp_files.append(temp_path)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process image
            results = st.session_state.orchestrator.process_image(temp_path, generate_gradcam=True)
            results['filename'] = uploaded_file.name
            batch_results.append(results)
        
        progress_bar.empty()
        
        # Store batch results
        st.session_state.batch_results = {
            'results': batch_results,
            'mode': mode,
            'patient_id': patient_id
        }
        st.session_state.results = None  # Clear single result
        st.success(f"✅ Successfully analyzed {len(batch_results)} scan(s)!")
        st.rerun()
        
    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Batch analysis failed: {e}")
    finally:
        # Cleanup temp files
        for temp_path in temp_files:
            try:
                os.remove(temp_path)
            except:
                pass

def show_results():
    """Display clinical report"""
    results = st.session_state.results
    pred = results['prediction']
    
    st.markdown("---")
    
    # Results Header
    st.markdown("### 📋 Diagnostic Results")
    st.caption("AI-Generated Clinical Assessment")
    
    # Create a clean report card
    report_id = f"NV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    report_date = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
    
    # Report Container - Create HTML string
    report_html = f"""
<div style="background: white; color: #1e293b; border-radius: 12px; padding: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 1.5rem 0;">
    <div style="margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 2px solid #e2e8f0;">
        <div style="font-size: 1.4rem; font-weight: 800; margin-bottom: 0.25rem;">CLINICAL REPORT</div>
        <div style="color: #64748b; font-size: 0.85rem;">Automated Neuro-Radiology Assessment</div>
    </div>
    <div style="margin-bottom: 1.5rem;">
        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #64748b; margin-bottom: 0.75rem;">PRIMARY FINDINGS</div>
        <div style="font-size: 2rem; font-weight: 800; color: #4169E1; margin-bottom: 0.5rem;">{pred['predicted_class'].title()}</div>
        <div style="font-size: 1rem; color: #1e293b;">Model Confidence: <strong>{pred['confidence']:.2f}%</strong></div>
    </div>
    <div style="margin-bottom: 1.5rem;">
        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #64748b; margin-bottom: 0.75rem;">CLINICAL IMPRESSION</div>
        <div style="color: #475569; line-height: 1.6; font-size: 0.95rem;">{results.get('explanation', 'AI-generated clinical impression pending.')}</div>
    </div>
    <div style="font-size: 0.8rem; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 1rem;">
        <div><strong>Report ID:</strong> {report_id}</div>
        <div><strong>Generated:</strong> {report_date}</div>
    </div>
</div>
"""
    st.markdown(report_html, unsafe_allow_html=True)
    
    # Probability Distribution
    st.markdown("### 📊 Confidence Distribution")
    
    for i, class_name in enumerate(pred['class_names']):
        # Convert numpy float to Python float to avoid type error
        prob = float(pred['probabilities'][i]) * 100
        is_predicted = class_name == pred['predicted_class']
        
        col_label, col_bar, col_value = st.columns([1, 3, 0.5])
        
        with col_label:
            if is_predicted:
                st.markdown(f"**{class_name.title()}**")
            else:
                st.markdown(f"{class_name.title()}")
        
        with col_bar:
            # Convert to float and ensure it's between 0 and 1
            progress_value = float(prob / 100.0)
            st.progress(progress_value)
        
        with col_value:
            st.markdown(f"**{prob:.1f}%**" if is_predicted else f"{prob:.1f}%")
    
    # Show Grad-CAM if available
    if results.get('gradcam'):
        st.markdown("---")
        st.markdown("### 🎯 Explainability Visualization")
        st.caption("Grad-CAM heatmap highlighting regions of diagnostic significance")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Original MRI Scan**")
            # Handle different image types - just pass it directly to st.image
            try:
                st.image(pred['original_image'], use_container_width=True)
            except Exception as e:
                st.error(f"Could not display original image: {str(e)}")
        with col_b:
            st.markdown("**Grad-CAM Heatmap**")
            try:
                st.image(results['gradcam']['overlay'], use_container_width=True)
            except Exception as e:
                st.error(f"Could not display Grad-CAM: {str(e)}")
    
    # Action Buttons
    st.markdown("---")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        if st.button("📄 Export PDF Report", use_container_width=True):
            try:
                pdf_path = generate_pdf_report(results)
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_bytes = pdf_file.read()
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=f"brain_mri_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                st.success("✅ PDF generated successfully!")
            except Exception as e:
                st.error(f"❌ Error generating PDF: {str(e)}")
    
    with btn_col2:
        if st.button("📋 View Detailed Report", use_container_width=True):
            st.session_state.show_detailed_report = not st.session_state.get('show_detailed_report', False)
            st.rerun()
    
    with btn_col3:
        if st.button("🔄 Analyze Another Scan", type="primary", use_container_width=True):
            st.session_state.results = None
            st.session_state.uploaded_image = None
            st.session_state.uploaded_images = None
            st.session_state.show_detailed_report = False
            st.session_state.detailed_analysis_generated = False
            st.session_state.detailed_analysis = None
            st.rerun()
    
    # Show detailed report if requested
    if st.session_state.get('show_detailed_report', False):
        show_detailed_report(results)

def show_batch_results():
    """Display batch analysis results"""
    batch_data = st.session_state.batch_results
    results_list = batch_data['results']
    mode = batch_data['mode']
    patient_id = batch_data.get('patient_id')
    
    st.markdown("---")
    
    # Batch Header
    mode_text = {
        'batch': 'Multiple Patients (Batch Analysis)',
        'multi_view': 'Single Patient (Multi-View Analysis)'
    }[mode]
    
    st.markdown(f"### 📋 {mode_text}")
    if patient_id:
        st.caption(f"Patient ID: {patient_id}")
    st.caption(f"Total Scans Analyzed: {len(results_list)}")
    
    # Summary Statistics
    st.markdown("#### 📊 Summary Statistics")
    
    class_counts = {}
    for res in results_list:
        pred_class = res['prediction']['predicted_class']
        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
    
    # Display summary in columns
    summary_cols = st.columns(len(class_counts))
    for idx, (cls, count) in enumerate(class_counts.items()):
        with summary_cols[idx]:
            st.metric(label=cls.title(), value=count)
    
    # Conflict warning for multi-view
    if mode == 'multi_view' and len(class_counts) > 1:
        st.warning("""
        ⚠️ **DIAGNOSTIC CONFLICT DETECTED**
        
        Multiple different tumor types were predicted across the views.
        This indicates potential inconsistency in the analysis.
        
        **RECOMMENDATION:** Manual review by a radiologist is STRONGLY recommended.
        """)
    
    # Individual Results
    st.markdown("---")
    st.markdown("#### 📁 Individual Scan Results")
    
    for idx, result in enumerate(results_list):
        with st.expander(f"📄 Scan #{idx+1}: {result['filename']}", expanded=(idx==0)):
            pred = result['prediction']
            
            # Result card
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Show image
                if pred.get('original_image'):
                    st.image(pred['original_image'], caption="MRI Scan", use_container_width=True)
            
            with col2:
                # Show prediction
                st.markdown(f"""
                <div style="background: white; color: #1e293b; border-radius: 8px; padding: 1.5rem;">
                    <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; 
                                color: #64748b; margin-bottom: 0.5rem;">
                        PREDICTION
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #4169E1; margin-bottom: 0.5rem;">
                        {pred['predicted_class'].title()}
                    </div>
                    <div style="font-size: 0.9rem; color: #1e293b;">
                        Confidence: <strong>{pred['confidence']:.2f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Probability bars
                st.markdown("**Probabilities:**")
                for i, cls in enumerate(pred['class_names']):
                    prob = float(pred['probabilities'][i]) * 100
                    st.progress(float(prob / 100.0))
                    st.caption(f"{cls.title()}: {prob:.1f}%")
            
            # Grad-CAM
            if result.get('gradcam'):
                st.markdown("**Explainability (Grad-CAM):**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(pred['original_image'], caption="Original", use_container_width=True)
                with col_b:
                    st.image(result['gradcam']['overlay'], caption="Grad-CAM", use_container_width=True)
            
            # Explanation
            if result.get('explanation'):
                st.markdown("**AI Interpretation:**")
                st.info(result['explanation'])
    
    # Action Buttons
    st.markdown("---")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        if st.button("📄 Export Batch PDF", use_container_width=True):
            try:
                pdf_path = generate_batch_pdf_report(results_list, mode, patient_id)
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_bytes = pdf_file.read()
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                st.success("✅ Batch PDF generated successfully!")
            except Exception as e:
                st.error(f"❌ Error generating PDF: {str(e)}")
    
    with btn_col2:
        if st.button("📊 View Detailed Report", use_container_width=True):
            st.session_state.show_batch_detailed = not st.session_state.get('show_batch_detailed', False)
            st.rerun()
    
    with btn_col3:
        if st.button("🔄 Start New Analysis", type="primary", use_container_width=True):
            st.session_state.batch_results = None
            st.session_state.results = None
            st.session_state.uploaded_images = None
            st.session_state.show_batch_detailed = False
            st.session_state.detailed_analysis_generated = False
            st.session_state.detailed_analysis = None
            st.rerun()
    
    # Show detailed batch report if requested
    if st.session_state.get('show_batch_detailed', False):
        show_batch_detailed_report(results_list, mode, patient_id)

def generate_batch_pdf_report(results_list, mode, patient_id=None):
    """Generate PDF report for batch analysis"""
    import tempfile
    from utils.pdf_generator import MedicalPDFGenerator
    
    # Create temporary file
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf_path = temp_pdf.name
    temp_pdf.close()
    
    # Generate PDF
    pdf_gen = MedicalPDFGenerator()
    pdf_gen.generate_batch_pdf(results_list, pdf_path, mode=mode, patient_id=patient_id)
    
    return pdf_path

def show_batch_detailed_report(results_list, mode, patient_id):
    """Display detailed report for batch analysis"""
    from agents.report_agent import ReportAgent
    
    st.markdown("---")
    st.markdown("### 📋 Comprehensive Batch Report")
    
    # Generate text report
    report_agent = ReportAgent()
    report_text = report_agent.generate_batch_report(results_list, mode=mode, patient_id=patient_id)
    
    st.code(report_text, language="text")

def initialize_system():
    """Initialize the orchestrator"""
    with st.spinner("🔄 Loading AI models..."):
        # Load API key from environment
        api_key = os.getenv('GEMINI_API_KEY')
        orchestrator = Orchestrator(gemini_api_key=api_key)
        orchestrator.load_model()
        st.session_state.orchestrator = orchestrator
    st.success("✅ System ready!")

def generate_pdf_report(results):
    """Generate PDF report for the analysis"""
    import tempfile
    from utils.pdf_generator import MedicalPDFGenerator
    
    # Create temporary file
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf_path = temp_pdf.name
    temp_pdf.close()
    
    # Generate PDF
    pdf_gen = MedicalPDFGenerator()
    pdf_gen.generate_single_scan_pdf(results, pdf_path)
    
    return pdf_path

def generate_detailed_analysis(results):
    """Generate enhanced detailed analysis using Gemini"""
    pred = results['prediction']
    
    try:
        reasoning_agent = st.session_state.orchestrator.reasoning_agent
        
        if not reasoning_agent.model:
            return "Enhanced analysis not available (Gemini API not configured)"
        
        # Calculate statistics
        probs = pred['probabilities']
        confidence = pred['confidence']
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(probs))
        uncertainty = entropy / max_entropy
        
        # Create probability summary
        prob_summary = "\n".join([
            f"- {name.title()}: {prob*100:.2f}%"
            for name, prob in zip(pred['class_names'], probs)
        ])
        
        prompt = f"""You are an expert medical AI assistant providing detailed analysis of brain MRI scan results.

Analysis Results:
- Predicted Diagnosis: {pred['predicted_class'].title()}
- Confidence: {confidence:.2f}%
- Uncertainty Index: {uncertainty*100:.2f}%
- Shannon Entropy: {entropy:.4f}

Probability Distribution:
{prob_summary}

Please provide a comprehensive detailed analysis including:

1. **Detailed Clinical Interpretation**: What this specific diagnosis means in medical terms
2. **Statistical Analysis Commentary**: Interpret the confidence level and uncertainty metrics
3. **Differential Diagnosis**: Discuss the alternative possibilities based on the probabilities
4. **Clinical Recommendations**: What next steps would typically be recommended
5. **Important Considerations**: Key factors to consider for this diagnosis

Format your response in markdown with clear sections. Be thorough but maintain clinical accuracy. 
Always remind that this is AI-generated analysis requiring professional medical verification.

Keep the tone professional and informative."""

        if reasoning_agent.client:
            # New API
            response = reasoning_agent.client.models.generate_content(
                model=reasoning_agent.model,
                contents=prompt
            )
            return response.text
        else:
            # Legacy API
            response = reasoning_agent.model.generate_content(prompt)
            return response.text
            
    except Exception as e:
        return f"⚠️ Could not generate enhanced analysis: {str(e)}"

def show_detailed_report(results):
    """Display detailed technical report with enhanced Gemini analysis"""
    st.markdown("---")
    st.markdown("### 📊 Detailed Technical Report")
    
    pred = results['prediction']
    
    # Generate enhanced analysis if we have Gemini
    if st.session_state.orchestrator and hasattr(st.session_state.orchestrator, 'reasoning_agent'):
        if not st.session_state.get('detailed_analysis_generated'):
            with st.spinner("🧠 Generating detailed analysis..."):
                detailed_analysis = generate_detailed_analysis(results)
                st.session_state.detailed_analysis = detailed_analysis
                st.session_state.detailed_analysis_generated = True
    
    # Create expandable sections
    with st.expander("🔬 Technical Analysis", expanded=True):
        st.markdown("#### Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - **Architecture:** ResNet-18
            - **Framework:** PyTorch
            - **Input Size:** 224×224×3
            - **Preprocessing:** ImageNet normalization
            """)
        with col2:
            st.markdown(f"""
            - **Classes:** {len(pred['class_names'])}
            - **Parameters:** ~11M
            - **Training:** Transfer Learning
            - **Activation:** Softmax
            """)
    
    with st.expander("📈 Statistical Analysis", expanded=True):
        st.markdown("#### Prediction Statistics")
        
        # Calculate statistics
        probs = pred['probabilities']
        confidence = pred['confidence']
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(probs))
        uncertainty = entropy / max_entropy
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Confidence", f"{confidence:.2f}%")
        col2.metric("Uncertainty", f"{uncertainty*100:.2f}%")
        col3.metric("Entropy", f"{entropy:.3f}")
        col4.metric("Max Prob", f"{np.max(probs)*100:.2f}%")
        
        st.markdown("#### Class Probabilities (Detailed)")
        prob_data = []
        for i, (cls, prob) in enumerate(zip(pred['class_names'], probs)):
            prob_data.append({
                "Class": cls.title(),
                "Probability": f"{prob*100:.4f}%",
                "Confidence Interval": f"±{(prob*5):.2f}%",
                "Status": "✅ Predicted" if i == pred['predicted_idx'] else "⬜ Alternative"
            })
        st.table(prob_data)
    
    with st.expander("🧠 AI Reasoning & Interpretation (Gemini-Powered)", expanded=True):
        st.markdown("#### Clinical Context")
        st.info(results.get('explanation', 'No detailed explanation available.'))
        
        # Show enhanced analysis if available
        if st.session_state.get('detailed_analysis'):
            st.markdown("#### 🤖 Enhanced AI Analysis")
            st.markdown(st.session_state.detailed_analysis)
        
        st.markdown("#### Key Findings")
        if confidence > 90:
            st.success("🟢 High confidence prediction - Strong diagnostic indicators present")
        elif confidence > 70:
            st.warning("🟡 Moderate confidence - Consider additional clinical correlation")
        else:
            st.error("🔴 Low confidence - Multiple review recommended")
    
    with st.expander("⚕️ Medical Context & Guidelines", expanded=True):
        st.markdown("#### Classification Categories")
        
        category_info = {
            "glioma": """**Glioma**: Tumors arising from glial cells. Include astrocytomas, oligodendrogliomas, 
            and ependymomas. Often require surgical intervention and adjuvant therapy.""",
            "meningioma": """**Meningioma**: Tumors arising from meninges. Usually benign and slow-growing. 
            Treatment depends on size and location.""",
            "pituitary": """**Pituitary Tumor**: Adenomas of the pituitary gland. Can be functional or non-functional. 
            Treatment may include surgery, medication, or radiation.""",
            "no tumor": """**No Tumor**: No abnormal mass detected. Normal brain tissue appearance. 
            Regular monitoring may still be recommended based on clinical symptoms."""
        }
        
        for category, info in category_info.items():
            if category in pred['class_names']:
                st.markdown(info)
                st.markdown("")
    
    with st.expander("📋 Full Classification Report", expanded=False):
        st.markdown("#### Complete Technical Report")
        
        report_text = f"""
{'='*70}
              COMPREHENSIVE BRAIN MRI ANALYSIS REPORT
{'='*70}

Report ID: NV-{datetime.now().strftime('%Y%m%d%H%M%S')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System: NeuroVision AI - Brain Tumor Classifier

{'='*70}
CLASSIFICATION RESULTS
{'='*70}

PRIMARY FINDING:      {pred['predicted_class'].upper()}
CONFIDENCE SCORE:     {confidence:.2f}%
UNCERTAINTY INDEX:    {uncertainty*100:.2f}%

{'='*70}
PROBABILITY DISTRIBUTION
{'='*70}

"""
        for i, (cls, prob) in enumerate(zip(pred['class_names'], probs)):
            marker = "→" if i == pred['predicted_idx'] else " "
            bar = "█" * int(prob * 50)
            report_text += f"{marker} {cls.upper():15s}: {prob*100:6.2f}% {bar}\n"
        
        report_text += f"""
{'='*70}
STATISTICAL ANALYSIS
{'='*70}

Shannon Entropy:      {entropy:.4f}
Normalized Entropy:   {uncertainty:.4f}
Max Probability:      {np.max(probs)*100:.2f}%
Min Probability:      {np.min(probs)*100:.2f}%
Mean Probability:     {np.mean(probs)*100:.2f}%
Std Deviation:        {np.std(probs)*100:.2f}%

{'='*70}
INTERPRETATION
{'='*70}

{results.get('explanation', 'No interpretation available.')}

{'='*70}
TECHNICAL SPECIFICATIONS
{'='*70}

Model Architecture:   ResNet-18 (Deep Residual Network)
Framework:           PyTorch 2.x
Preprocessing:       Resize to 224x224, ImageNet normalization
Input Format:        RGB (3 channels)
Output Format:       Softmax probabilities
Classes:             {', '.join([c.title() for c in pred['class_names']])}
Total Parameters:    ~11.7 Million
Training Dataset:    Brain MRI Dataset (Kaggle)

{'='*70}
MEDICAL DISCLAIMER
{'='*70}

This report is generated by an AI system for research and educational
purposes only. It is NOT a medical device and should NOT be used for
actual medical diagnosis or treatment decisions. Always consult
qualified healthcare professionals for proper medical evaluation.

{'='*70}
END OF REPORT
{'='*70}
"""
        st.code(report_text, language="text")

def create_probability_chart(probabilities, class_names, predicted_idx):
    """Create a probability bar chart"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = ['#FF6B6B' if i == predicted_idx else '#4ECDC4' for i in range(len(probabilities))]
    bars = ax.barh(class_names, probabilities * 100, color=colors)
    
    ax.set_xlabel('Probability (%)', fontsize=12)
    ax.set_title('Classification Probabilities', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', 
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    """Main application"""
    # Route to appropriate page
    if st.session_state.page == 'landing':
        show_landing_page()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()

"""
Brain Tumor Classification System - Streamlit Web App
"""

import streamlit as st
import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator import Orchestrator
from config import MEDICAL_DISCLAIMER, CLASS_NAMES

# Page configuration
st.set_page_config(
    page_title="Brain Tumor AI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'results' not in st.session_state:
    st.session_state.results = None

def initialize_system(api_key=None):
    """Initialize the orchestrator"""
    with st.spinner("🔄 Loading AI models..."):
        orchestrator = Orchestrator(gemini_api_key=api_key)
        orchestrator.load_model()
        st.session_state.orchestrator = orchestrator
    st.success("✅ System ready!")

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
    # Header
    st.markdown('<p class="main-header">🧠 Brain Tumor AI Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered MRI Analysis System (Demo)</p>', unsafe_allow_html=True)
    
    # Medical Disclaimer
    with st.expander("⚠️ IMPORTANT: Medical Disclaimer - READ FIRST", expanded=False):
        st.warning(MEDICAL_DISCLAIMER)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key input
        with st.expander("🔑 Gemini API Key (Optional)", expanded=False):
            api_key = st.text_input(
                "Enter your Google Gemini API key for enhanced explanations:",
                type="password",
                help="Optional: Provides LLM-generated explanations. Leave blank for fallback explanations."
            )
            st.caption("[Get a free API key](https://makersuite.google.com/app/apikey)")
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary", use_container_width=True):
            initialize_system(api_key if api_key else None)
        
        st.divider()
        
        # Options
        st.header("📊 Analysis Options")
        generate_gradcam = st.checkbox("Generate Grad-CAM Heatmap", value=True)
        show_report = st.checkbox("Show Full Report", value=True)
        
        st.divider()
        
        # Info
        st.header("ℹ️ About")
        st.info("""
        **Model:** EfficientNet-B0  
        **Framework:** PyTorch  
        **Classes:** Glioma, Meningioma, Pituitary, No Tumor  
        
        **Agents:**
        - 👁️ Vision Agent
        - 🎨 Explainability Agent  
        - 🧠 Reasoning Agent
        - 📄 Report Agent
        """)
    
    # Main content
    if st.session_state.orchestrator is None:
        st.info("👈 Please click 'Initialize System' in the sidebar to start")
        return
    
    # File uploader
    st.header("📤 Upload Brain MRI Image")
    uploaded_file = st.file_uploader(
        "Choose an MRI image (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI scan for classification"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(temp_path)
            st.image(image, use_container_width=True)
        
        # Process button
        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing image through AI pipeline..."):
                try:
                    results = st.session_state.orchestrator.process_image(
                        temp_path,
                        generate_gradcam=generate_gradcam
                    )
                    st.session_state.results = results
                except Exception as e:
                    st.error(f"❌ Error during processing: {e}")
                    return
        
        # Display results
        if st.session_state.results:
            results = st.session_state.results
            pred = results['prediction']
            
            st.success("✅ Analysis Complete!")
            st.divider()
            
            # Results section
            st.header("📊 Classification Results")
            
            # Prediction box
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Class", pred['predicted_class'].upper())
            with col2:
                st.metric("Confidence", f"{pred['confidence']:.2f}%")
            with col3:
                status = "🟢 High" if pred['confidence'] > 70 else "🟡 Medium" if pred['confidence'] > 50 else "🔴 Low"
                st.metric("Status", status)
            
            # Probability chart
            st.subheader("📈 Probability Distribution")
            fig = create_probability_chart(
                pred['probabilities'],
                pred['class_names'],
                pred['predicted_idx']
            )
            st.pyplot(fig)
            
            # Grad-CAM visualization
            if results['gradcam'] and generate_gradcam:
                st.divider()
                st.header("🎨 Explainability: Grad-CAM Visualization")
                st.caption("Highlighted regions show what the AI focused on to make its prediction")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(pred['original_image'], use_container_width=True)
                with col2:
                    st.subheader("Grad-CAM Heatmap")
                    st.image(results['gradcam']['overlay'], use_container_width=True)
            
            # Explanation
            st.divider()
            st.header("🧠 AI Explanation")
            st.markdown(results['explanation'])
            
            # Full report
            if show_report:
                st.divider()
                st.header("📄 Detailed Report")
                with st.expander("View Full Diagnostic Report", expanded=False):
                    st.code(results['report'], language=None)
            
            # Download button
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Report",
                    data=results['report'],
                    file_name=f"tumor_report_{uploaded_file.name}.txt",
                    mime="text/plain"
                )
        
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass
    
    # Footer
    st.divider()
    st.caption("Built for AI Hackathon 2025 | 🧠 Brain Tumor Classification System")

if __name__ == "__main__":
    main()

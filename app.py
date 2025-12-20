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
from datetime import datetime
from utils.pdf_generator import MedicalPDFGenerator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

def initialize_system():
    """Initialize the orchestrator"""
    with st.spinner("🔄 Loading AI models..."):
        # Load API key from environment
        api_key = os.getenv('GEMINI_API_KEY')
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
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary", use_container_width=True):
            initialize_system()
        
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
    
    
    # Analysis Mode Selector
    st.header("⚙️ Analysis Settings")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_mode = st.radio(
            "Select Analysis Mode:",
            ["👥 Multiple Patients (Batch Analysis)", "👤 Single Patient (Multi-View Analysis)"],
            help="Choose how to interpret the uploaded images."
        )
        
    mode_key = 'single_patient' if "Single Patient" in analysis_mode else 'batch'
    patient_id = None
    
    if mode_key == 'single_patient':
        with col2:
            patient_id = st.text_input("Patient ID / Name (Optional)", placeholder="e.g., John Doe")

    st.divider()

    # File uploader
    st.header("📤 Upload Brain MRI Image(s)")
    uploaded_files = st.file_uploader(
        "Choose MRI image(s) (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload one or more brain MRI scans for classification"
    )
    
    if uploaded_files:
        # Display uploaded images count
        st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        # Process button
        if st.button("🔬 Analyze Batch", type="primary", use_container_width=True):
            
            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each image
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing image {idx+1}/{len(uploaded_files)}: {uploaded_file.name}...")
                
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Process image
                    results = st.session_state.orchestrator.process_image(
                        temp_path,
                        generate_gradcam=generate_gradcam
                    )
                    results['filename'] = uploaded_file.name
                    all_results.append(results)
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                finally:
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            # Save results to session state
            st.session_state.results = all_results
            st.session_state.analysis_metadata = {'mode': mode_key, 'patient_id': patient_id}
            status_text.empty()
            st.success("✅ Batch Analysis Complete!")
        
        # Display results if available
        if st.session_state.results:
            results_list = st.session_state.results
            metadata = st.session_state.get('analysis_metadata', {'mode': 'batch', 'patient_id': None})
            
            st.divider()
            st.header("📊 Batch Results Summary")
            
            # Generate consolidated report
            batch_report = st.session_state.orchestrator.generate_batch_report(
                results_list, 
                mode=metadata['mode'], 
                patient_id=metadata['patient_id']
            )
            
            
            # Download PDF Report
            pdf_path = f"temp_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            try:
                pdf_gen = MedicalPDFGenerator()
                pdf_gen.generate_batch_pdf(
                    results_list, 
                    pdf_path, 
                    mode=metadata['mode'], 
                    patient_id=metadata.get('patient_id')
                )
                
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_file,
                        file_name=f"medical_report_{metadata['mode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                # Clean up
                try:
                    os.remove(pdf_path)
                except:
                    pass
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
            
            st.divider()
            
            # Display full report text
            st.subheader("📄 Full Report")
            with st.expander("View Complete Report", expanded=True):
                st.text(batch_report)
            
            st.divider()
            st.subheader("Results by Image")
            
            # Display individual results
            for idx, res in enumerate(results_list):
                with st.expander(f"Image {idx+1}: {res['filename']} - {res['prediction']['predicted_class'].upper()} ({res['prediction']['confidence']:.1f}%)", expanded=False):
                    
                    pred = res['prediction']
                    
                    # Layout: Image | GradCAM | Stats
                    c1, c2, c3 = st.columns([1, 1, 1])
                    
                    with c1:
                        st.caption("Original Image")
                        st.image(pred['original_image'], use_container_width=True)
                    
                    with c2:
                        if res.get('gradcam') and generate_gradcam:
                            st.caption("Grad-CAM Explanation")
                            st.image(res['gradcam']['overlay'], use_container_width=True)
                        elif res.get('gradcam_error'):
                            st.error(f"Grad-CAM Failed: {res['gradcam_error']}")
                        else:
                            st.info("Grad-CAM disabled or not generated")
                            
                    with c3:
                        st.caption("Prediction Details")
                        st.metric("Class", pred['predicted_class'].upper())
                        st.metric("Confidence", f"{pred['confidence']:.2f}%")
                        
                        # Show probability chart
                        fig = create_probability_chart(
                            pred['probabilities'],
                            pred['class_names'],
                            pred['predicted_idx']
                        )
                        st.pyplot(fig)
                    
                    # Explanation text
                    st.markdown("---")
                    st.markdown(f"**AI Explanation:** {res['explanation']}")
    
    # Footer
    st.divider()
    st.caption("Built for AI Hackathon 2025 | 🧠 Brain Tumor Classification System")

if __name__ == "__main__":
    main()

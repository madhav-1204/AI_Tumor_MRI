from fpdf import FPDF
import datetime
import os
from PIL import Image as PILImage
import numpy as np

class StructuredMRIReport(FPDF):
    def header(self):
        # Professional Header
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(33, 37, 41)
        self.cell(0, 10, 'NEURORADIOLOGY PEER REVIEW REPORT', 0, 1, 'C')
        self.set_draw_color(0, 102, 204)
        self.set_line_width(1)
        self.line(10, 25, 200, 25)
        self.ln(15)

    def chapter_title(self, label):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(240, 240, 240) # Light gray
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, label, 0, 1, 'L', fill=True)
        self.ln(4)

    def section_content(self, text):
        self.set_font('Helvetica', '', 11)
        clean_text = text.replace("**", "").replace("__", "")
        self.multi_cell(0, 6, clean_text)
        self.ln(6)

    def footer(self):
        # Footer with Disclaimer
        self.set_y(-25)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, "DISCLAIMER: AI-assisted analysis for research/educational use only. Not a medical device.", 0, 1, 'C')
        self.cell(0, 5, "Final verification by a qualified radiologist is mandatory.", 0, 0, 'C')

class MedicalPDFGenerator:
    def generate_batch_pdf(self, results_list, output_path, mode='batch', patient_id='Unknown', global_impression=None):
        pdf = StructuredMRIReport()
        pdf.set_auto_page_break(auto=True, margin=25)
        
        # Determine Title Context
        report_context = "Single Patient (Multi-View)" if mode == 'single_patient' else "Batch Analysis"
        
        pdf.add_page()
        
        # 1. Patient & Study Info
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(100, 7, f"Patient Name/ID: {patient_id}")
        pdf.cell(0, 7, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", 0, 1, 'R')
        pdf.cell(100, 7, f"Mode: {report_context}")
        pdf.ln(10)

        # AI Summary (Aggregated)
        pdf.chapter_title("AI SUMMARY (AGGREGATED)")
        
        # Calculate stats
        class_counts = {}
        for res in results_list:
            p = res['prediction']['predicted_class']
            class_counts[p] = class_counts.get(p, 0) + 1
        
        summary_txt = f"Total Scans Processed: {len(results_list)}\n"
        summary_txt += "Tumor Distribution: " + ", ".join([f"{k.upper()}: {v}" for k,v in class_counts.items()])
        pdf.section_content(summary_txt)
        pdf.ln(5)

        # Iterate through scans
        for i, res in enumerate(results_list):
            if i > 0:
                pdf.add_page()
                pdf.ln(5) # Space after header
                
            prediction = res['prediction']
            explanation = res.get('explanation', '')
            
            # Extract Observations & Impression from explanation if possible
            # Robust parsing for various formats
            observations = ""
            impression = ""
            
            # Try to split by "Clinical Impression" or "Impression"
            if "Clinical Impression" in explanation:
                parts = explanation.split("Clinical Impression")
                # Handle Findings OR Diagnostic Observations header in the first part
                obs_part = parts[0]
                obs_part = obs_part.replace("Diagnostic Observations", "").replace("Findings", "").replace(":", "")
                observations = obs_part.strip()
                
                # Cleaning the impression part
                impression = parts[1].replace(":", "").strip()
            elif "Impression" in explanation:
                 parts = explanation.split("Impression")
                 obs_part = parts[0]
                 obs_part = obs_part.replace("Diagnostic Observations", "").replace("Findings", "").replace(":", "")
                 observations = obs_part.strip()
                 impression = parts[1].replace(":", "").strip()
            else:
                observations = explanation # Fallback if no split found
            
            # --- Page Content ---
            
            # Scan Header with Prediction
            pdf.set_font('Helvetica', 'B', 14)
            prediction_text = f"{prediction['predicted_class'].upper()} ({prediction['confidence']:.2f}%)"
            # Get filename safely with fallback
            filename = res.get('filename', f'Scan_{i+1}')
            # Clean filename for PDF (remove special characters that might cause issues)
            filename = filename.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(0, 10, f"Scan: {filename}  |  AI: {prediction_text}", 0, 1)
            pdf.ln(2)

            # 2. AI Model Output (REMOVED as requested - integrated into header)
            # pdf.chapter_title("AI CLASSIFICATION DATA")
            # pdf.section_content(ai_text)
            
            # Image Placement (Original & GradCAM)
            try:
                # Save temp images for PDF
                temp_orig = f"temp_orig_{i}.jpg"
                if prediction.get('original_image'):
                    prediction['original_image'].save(temp_orig)
                    
                    # Layout: Side by side images
                    y_pos = pdf.get_y()
                    max_h = 60 # Max height in mm
                    max_w = 80 # Max width in mm
                    
                    # Original Image
                    # Let FPDF handle aspect ratio by setting one dimension, but we want to fit in box.
                    # We'll use a fixed width of 80mm unless height exceeds max_h
                    img_w, img_h = prediction['original_image'].size
                    aspect = img_h / img_w
                    
                    # Calculate dimensions to fit in 80x60 box
                    display_h = max_w * aspect
                    display_w = max_w
                    if display_h > max_h:
                        display_h = max_h
                        display_w = max_h / aspect
                    
                    pdf.image(temp_orig, x=10, y=y_pos, w=display_w, h=display_h)
                    
                    if res.get('gradcam') and res['gradcam'].get('overlay') is not None:
                        temp_grad = f"temp_grad_{i}.jpg"
                        grad_pil = PILImage.fromarray(res['gradcam']['overlay'])
                        grad_pil.save(temp_grad)
                        
                        # Use same calculated dims for consistency if images are same size
                        pdf.image(temp_grad, x=100, y=y_pos, w=display_w, h=display_h)
                        
                        # Labels
                        pdf.set_xy(10, y_pos + display_h + 2)
                        pdf.set_font('Helvetica', 'I', 9)
                        pdf.cell(display_w, 5, "Original MRI", 0, 0, 'C')
                        
                        pdf.set_xy(100, y_pos + display_h + 2)
                        pdf.cell(display_w, 5, "Grad-CAM Heatmap", 0, 1, 'C')
                        
                        # Cleanup later
                        if os.path.exists(temp_grad):
                            os.remove(temp_grad)
                    else:
                        # Label for single image
                        pdf.set_xy(10, y_pos + display_h + 2)
                        pdf.set_font('Helvetica', 'I', 9)
                        pdf.cell(display_w, 5, "Original MRI", 0, 0, 'C')

                    if os.path.exists(temp_orig):
                         os.remove(temp_orig)
                    
                    # Ensure we move down past the images before writing text
                    # Dynamic offset based on actual image height + text space
                    pdf.set_y(y_pos + display_h + 15) 
                    
                else:
                    pdf.ln(5)
            except Exception as e:
                pdf.section_content(f"[Image Embedding Failed: {e}]")


            # 3. Findings (was Diagnostic Observations)
            pdf.chapter_title("FINDINGS") 
            
            # Clean markdown: Remove **markers** and * bullets
            clean_obs = observations if observations else "No details provided."
            clean_obs = clean_obs.replace("**", "").replace("* ", "- ").replace("__", "")
            
            pdf.section_content(clean_obs)

            # 4. Clinical Impression
            if impression:
                pdf.chapter_title("CLINICAL IMPRESSION")
                clean_imp = impression.replace("**", "").replace("* ", "- ").replace("__", "")
                pdf.section_content(clean_imp)
            
            pdf.ln(5)

            # 4. Global Impression (Aggregated)
            # Only show if provided (it will be None if conflict detected)
            if global_impression:
                 pdf.chapter_title("FINAL CLINICAL IMPRESSION (CASE SYNTHESIS)")
                 if hasattr(global_impression, 'text'):
                     pdf.section_content(global_impression.text)
                 else:
                     pdf.section_content(str(global_impression))
            pdf.ln(5)

        # Technical Specifications (At the end)
        pdf.add_page()
        pdf.chapter_title("TECHNICAL SPECIFICATIONS")
        tech_specs = (
            "Model Architecture: EfficientNet-B0 / ResNet-18 (PyTorch)\n"
            "Preprocessing: Resize to 224x224, ImageNet Normalization\n"
            "Inference Device: CPU\n"
            "Explainability: Grad-CAM (Gradient-weighted Class Activation Mapping)\n"
            "AI Assistant: Google Gemini 1.5 Flash (Vision-Enabled)"
        )
        pdf.section_content(tech_specs)

        pdf.output(output_path)
    
    # Keep legacy methods to avoid import errors if referred elsewhere, mapping to new logic if possible
    def generate_single_scan_pdf(self, result, output_path, patient_id=None):
        self.generate_batch_pdf([result], output_path, mode='single_patient', patient_id=patient_id)

"""
PDF Report Generator for Brain Tumor Classification System
Generates professional medical-style PDF reports with images
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from datetime import datetime
import io
from PIL import Image as PILImage
import numpy as np


class MedicalPDFGenerator:
    """Generates medical-style PDF reports"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles for medical reports"""
        
        # Title style
        if 'ReportTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='ReportTitle',
                parent=self.styles['Heading1'],
                fontSize=18,
                textColor=colors.HexColor('#1E88E5'),
                spaceAfter=12,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))
        
        # Section header
        if 'SectionHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#424242'),
                spaceAfter=6,
                spaceBefore=12,
                fontName='Helvetica-Bold'
            ))
        
        # Body text
        if 'BodyText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontSize=10,
                alignment=TA_JUSTIFY,
                spaceAfter=6
            ))
        
        # Warning style
        if 'Warning' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='Warning',
                parent=self.styles['Normal'],
                fontSize=10,
                textColor=colors.red,
                spaceAfter=6,
                spaceBefore=6,
                fontName='Helvetica-Bold'
            ))
    
    def _pil_to_reportlab_image(self, pil_image, width=3*inch):
        """Convert PIL Image to ReportLab Image"""
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Calculate height maintaining aspect ratio
        aspect = pil_image.height / pil_image.width
        height = width * aspect
        
        return Image(img_buffer, width=width, height=height)
    
    def generate_single_scan_pdf(self, result, output_path, patient_id=None):
        """Generate PDF for a single scan"""
        
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                                rightMargin=0.75*inch, leftMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        story = []
        pred = result['prediction']
        
        # Title
        story.append(Paragraph("BRAIN MRI CLASSIFICATION REPORT", self.styles['ReportTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Report metadata
        metadata_text = f"""
        <b>Report Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br/>
        <b>Patient ID:</b> {patient_id if patient_id else 'N/A'}<br/>
        <b>System:</b> AI Brain Tumor Classifier (EfficientNet-B0)
        """
        story.append(Paragraph(metadata_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))
        
        # Classification Results
        story.append(Paragraph("CLASSIFICATION RESULTS", self.styles['SectionHeader']))
        results_text = f"""
        <b>Primary Finding:</b> {pred['predicted_class'].upper()}<br/>
        <b>Confidence Score:</b> {pred['confidence']:.2f}%
        """
        story.append(Paragraph(results_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))
        
        # Probability Distribution Table
        story.append(Paragraph("Probability Distribution", self.styles['SectionHeader']))
        prob_data = [['Class', 'Probability']]
        for cls, prob in zip(pred['class_names'], pred['probabilities']):
            prob_data.append([cls.upper(), f"{prob*100:.2f}%"])
        
        prob_table = Table(prob_data, colWidths=[2.5*inch, 1.5*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E88E5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Images Section
        story.append(Paragraph("IMAGING ANALYSIS", self.styles['SectionHeader']))
        
        # Create image table (Original | Grad-CAM)
        img_data = []
        img_row = []
        
        # Original image
        if pred.get('original_image'):
            orig_img = self._pil_to_reportlab_image(pred['original_image'], width=2.5*inch)
            img_row.append(orig_img)
        
        # Grad-CAM
        if result.get('gradcam') and result['gradcam'].get('overlay') is not None:
            gradcam_pil = PILImage.fromarray(result['gradcam']['overlay'])
            gradcam_img = self._pil_to_reportlab_image(gradcam_pil, width=2.5*inch)
            img_row.append(gradcam_img)
        
        if img_row:
            img_data.append(img_row)
            img_data.append(['Original MRI Scan', 'Grad-CAM Heatmap'])
            
            img_table = Table(img_data)
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 1), (-1, 1), 10),
            ]))
            story.append(img_table)
        
        story.append(Spacer(1, 0.3*inch))
        
        # AI Explanation
        story.append(Paragraph("INTERPRETATION", self.styles['SectionHeader']))
        story.append(Paragraph(result.get('explanation', 'No explanation available.'), self.styles['BodyText']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Disclaimer
        story.append(Paragraph("MEDICAL DISCLAIMER", self.styles['SectionHeader']))
        disclaimer = """
        This report is generated by an AI system for research and educational purposes only. 
        It is NOT a medical device and should NOT be used for actual medical diagnosis or treatment decisions. 
        Always consult qualified healthcare professionals for proper medical evaluation.
        """
        story.append(Paragraph(disclaimer, self.styles['Warning']))
        
        # Build PDF
        doc.build(story)
    
    def generate_batch_pdf(self, results_list, output_path, mode='batch', patient_id=None):
        """Generate PDF for batch analysis"""
        
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                                rightMargin=0.75*inch, leftMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        story = []
        
        # Title
        if mode == 'single_patient':
            title = "PATIENT CASE REPORT (MULTI-VIEW ANALYSIS)"
        else:
            title = "BATCH ANALYSIS SUMMARY REPORT"
        
        story.append(Paragraph(title, self.styles['ReportTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Metadata
        metadata_text = f"""
        <b>Report Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br/>
        <b>Patient ID:</b> {patient_id if patient_id else 'N/A'}<br/>
        <b>Total Scans:</b> {len(results_list)}
        """
        story.append(Paragraph(metadata_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))
        
        # Summary statistics
        class_counts = {}
        for res in results_list:
            pred = res['prediction']['predicted_class']
            class_counts[pred] = class_counts.get(pred, 0) + 1
        
        story.append(Paragraph("SUMMARY OF FINDINGS", self.styles['SectionHeader']))
        summary_text = "<br/>".join([f"<b>{cls.upper()}:</b> {count}" for cls, count in class_counts.items()])
        story.append(Paragraph(summary_text, self.styles['BodyText']))
        
        # Conflict warning for single patient
        if mode == 'single_patient' and len(class_counts) > 1:
            story.append(Spacer(1, 0.2*inch))
            warning_text = """
            <b>⚠️ DIAGNOSTIC CONFLICT DETECTED</b><br/>
            Multiple different tumor types were predicted across the views. 
            Manual review by a radiologist is STRONGLY recommended.
            """
            story.append(Paragraph(warning_text, self.styles['Warning']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Individual scan results
        story.append(Paragraph("DETAILED SCAN RESULTS", self.styles['SectionHeader']))
        
        for idx, res in enumerate(results_list):
            if idx > 0:
                story.append(PageBreak())
            
            pred = res['prediction']
            
            story.append(Paragraph(f"Scan #{idx+1}: {res['filename']}", self.styles['SectionHeader']))
            
            results_text = f"""
            <b>Prediction:</b> {pred['predicted_class'].upper()}<br/>
            <b>Confidence:</b> {pred['confidence']:.2f}%
            """
            story.append(Paragraph(results_text, self.styles['BodyText']))
            story.append(Spacer(1, 0.2*inch))
            
            # Images
            img_row = []
            if pred.get('original_image'):
                orig_img = self._pil_to_reportlab_image(pred['original_image'], width=2.5*inch)
                img_row.append(orig_img)
            
            if res.get('gradcam') and res['gradcam'].get('overlay') is not None:
                gradcam_pil = PILImage.fromarray(res['gradcam']['overlay'])
                gradcam_img = self._pil_to_reportlab_image(gradcam_pil, width=2.5*inch)
                img_row.append(gradcam_img)
            
            if img_row:
                img_data = [img_row, ['Original MRI', 'Grad-CAM Heatmap']]
                img_table = Table(img_data)
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
                ]))
                story.append(img_table)
        
        # Build PDF
        doc.build(story)

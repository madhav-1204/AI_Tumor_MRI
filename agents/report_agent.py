"""
Report Agent - Generates structured diagnostic-style reports
"""

from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class ReportAgent:
    """
    Generates structured, diagnostic-style reports
    mimicking clinical report format
    """
    
    def generate_report(self, prediction_results, explanation_text):
        """
        Generate a structured medical-style report
        
        Args:
            prediction_results: Dict from Vision Agent
            explanation_text: Text from Reasoning Agent
            
        Returns:
            Formatted report string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
{'='*70}
                   BRAIN MRI CLASSIFICATION REPORT
                      (AI-GENERATED - FOR DEMO ONLY)
{'='*70}

Report Generated: {timestamp}
System: EfficientNet-B0 Brain Tumor Classifier

{'='*70}
CLASSIFICATION RESULTS
{'='*70}

PRIMARY FINDING:      {prediction_results['predicted_class'].upper()}
CONFIDENCE SCORE:     {prediction_results['confidence']:.2f}%

{'='*70}
PROBABILITY DISTRIBUTION
{'='*70}

"""
        
        # Add probability table
        for idx, (class_name, prob) in enumerate(zip(
            prediction_results['class_names'],
            prediction_results['probabilities']
        )):
            marker = "→" if idx == prediction_results['predicted_idx'] else " "
            bar = "█" * int(prob * 50)  # Visual bar
            report += f"{marker} {class_name.upper():15s}: {prob*100:6.2f}% {bar}\n"
        
        report += f"""
{'='*70}
INTERPRETATION
{'='*70}

{explanation_text}

{'='*70}
TECHNICAL DETAILS
{'='*70}

Model Architecture:   EfficientNet-B0
Framework:           PyTorch
Preprocessing:       Resize to 224x224, ImageNet normalization
Classes:             {', '.join(prediction_results['class_names'])}

{'='*70}

END OF REPORT
"""
        
        return report

    def generate_batch_report(self, results_list, mode='batch', patient_id=None):
        """Generate a consolidated report for multiple images"""
        
        # Calculate summary statistics
        total_scans = len(results_list)
        class_counts = {}
        processed_findings = []
        
        for res in results_list:
            pred = res['prediction']['predicted_class']
            conf = res['prediction']['confidence']
            class_counts[pred] = class_counts.get(pred, 0) + 1
            processed_findings.append((pred, conf))
            
        summary_lines = [f"- {cls.upper()}: {count}" for cls, count in class_counts.items()]
        summary_text = "\n".join(summary_lines)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if mode == 'single_patient':
            # Determine if there's a conflict
            num_different_predictions = len(class_counts)
            dominant_diagnosis = max(class_counts, key=class_counts.get).upper()
            
            # Calculate average confidence for dominant diagnosis
            dominant_confidences = [conf for pred, conf in processed_findings if pred == dominant_diagnosis.lower()]
            avg_confidence = sum(dominant_confidences) / len(dominant_confidences) if dominant_confidences else 0
            
            # Conflict warning
            conflict_warning = ""
            if num_different_predictions > 1:
                conflict_warning = f"""
⚠️ DIAGNOSTIC CONFLICT DETECTED ⚠️
-----------------------------------
Multiple different tumor types were predicted across the {total_scans} views.
This indicates potential inconsistency in the analysis.

RECOMMENDATION: Manual review by a radiologist is STRONGLY recommended.
The predictions may be unreliable due to:
- Image quality variations
- Different anatomical views showing different features
- Model uncertainty

Distribution of predictions:
{summary_text}
"""
            
            report = f"""
PATIENT CASE REPORT (MULTI-VIEW ANALYSIS)
-----------------------------------------
Date: {timestamp}
Patient ID: {patient_id if patient_id else 'ANONYMOUS'}
Case ID: CASE_{datetime.now().strftime("%Y%m%d_%H%M%S")}

CLINICAL SUMMARY
----------------
Patient underwent multi-view MRI analysis for tumor classification.
Total Views Analyzed: {total_scans}

{conflict_warning}

PRIMARY DIAGNOSTIC IMPRESSION
------------------------------
--> {dominant_diagnosis}
    Average Confidence: {avg_confidence:.1f}%
    Consensus: {class_counts.get(dominant_diagnosis.lower(), 0)}/{total_scans} views

DETAILED VIEW-BY-VIEW FINDINGS
===============================
"""
            # Add compact view summaries
            for i, res in enumerate(results_list):
                pred = res['prediction']
                report += f"""
View #{i+1}: {res['filename']}
  Prediction: {pred['predicted_class'].upper()}
  Confidence: {pred['confidence']:.1f}%
  Probabilities: {', '.join([f"{cls}: {prob*100:.1f}%" for cls, prob in zip(pred['class_names'], pred['probabilities'])])}
"""
            
        else:
            # Batch mode - keep individual reports
            report = f"""
BATCH ANALYSIS SUMMARY REPORT
-----------------------------
Date: {timestamp}
Batch ID: BATCH_{datetime.now().strftime("%Y%m%d_%H%M%S")}

OVERVIEW
--------
Total Scans Processed: {total_scans}

Distribution of Findings:
{summary_text}

INDIVIDUAL SCAN RESULTS
======================="""
        
            for i, res in enumerate(results_list):
                report += f"\n\nScan #{i+1} : {res['filename']}"
                report += "\n" + "-" * 40
                report += self.generate_report(res['prediction'], res['explanation'])
                report += "\n" + "="*40
            
        return report
    
    def generate_summary(self, prediction_results):
        """Generate a short summary for quick viewing"""
        return f"""
**Quick Summary**
- **Result:** {prediction_results['predicted_class'].upper()}
- **Confidence:** {prediction_results['confidence']:.1f}%
- **Top 2 Predictions:** 
  1. {prediction_results['class_names'][prediction_results['predicted_idx']]} ({prediction_results['probabilities'][prediction_results['predicted_idx']]*100:.1f}%)
  2. {prediction_results['class_names'][prediction_results['probabilities'].argsort()[-2]]} ({prediction_results['probabilities'][prediction_results['probabilities'].argsort()[-2]]*100:.1f}%)
"""

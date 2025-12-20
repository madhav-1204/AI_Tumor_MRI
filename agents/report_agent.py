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
{MEDICAL_DISCLAIMER}
{'='*70}

END OF REPORT
"""
        
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

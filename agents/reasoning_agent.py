"""
Reasoning Agent - Generates medical explanations using LLM
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class ReasoningAgent:
    """
    Uses LLM to generate human-readable explanations
    of the classification results
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = None
        
        if self.api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(GEMINI_MODEL)
                print("✅ Gemini API initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize Gemini: {e}")
                self.model = None
        else:
            print("⚠️  Gemini API not available - using fallback explanations")
    
    def generate_explanation(self, predicted_class, confidence, probabilities, class_names):
        """
        Generate medical explanation
        
        Args:
            predicted_class: Predicted tumor type
            confidence: Confidence score
            probabilities: Array of all class probabilities
            class_names: List of class names
            
        Returns:
            Explanation text
        """
        if self.model:
            return self._generate_with_llm(predicted_class, confidence, probabilities, class_names)
        else:
            return self._generate_fallback(predicted_class, confidence, probabilities, class_names)
    
    def _generate_with_llm(self, predicted_class, confidence, probabilities, class_names):
        """Generate explanation using Gemini"""
        try:
            # Create probability summary
            prob_summary = "\n".join([
                f"- {name}: {prob*100:.1f}%"
                for name, prob in zip(class_names, probabilities)
            ])
            
            prompt = f"""You are a medical AI assistant helping to explain brain tumor classification results.

Prediction Results:
- Predicted Type: {predicted_class}
- Confidence: {confidence:.1f}%

All Probabilities:
{prob_summary}

Please provide a brief, simple explanation (2-3 sentences) about:
1. What this result means
2. The confidence level
3. A reminder that this is AI-generated and needs professional verification

Keep it simple and easy to understand for non-medical users. Include the medical disclaimer."""

            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"⚠️  LLM generation failed: {e}")
            return self._generate_fallback(predicted_class, confidence, probabilities, class_names)
    
    def _generate_fallback(self, predicted_class, confidence, probabilities, class_names):
        """Generate simple rule-based explanation"""
        
        # Tumor type descriptions
        descriptions = {
            'glioma': 'Gliomas are tumors that originate from glial cells in the brain.',
            'meningioma': 'Meningiomas are tumors that arise from the meninges (membranes covering the brain).',
            'pituitary': 'Pituitary tumors develop in the pituitary gland at the base of the skull.',
            'notumor': 'No tumor detected - the scan appears normal.'
        }
        
        desc = descriptions.get(predicted_class, 'Unknown tumor type.')
        
        # Confidence interpretation
        if confidence >= 80:
            conf_text = "The model has high confidence in this prediction."
        elif confidence >= 60:
            conf_text = "The model has moderate confidence in this prediction."
        else:
            conf_text = "The model has low confidence - further analysis recommended."
        
        explanation = f"""
**AI Classification Result:**

The system predicts: **{predicted_class.upper()}**
Confidence: {confidence:.1f}%

{desc}

{conf_text}

**Important:** This is an AI-generated prediction for educational purposes only. It should NOT be used for medical diagnosis. Always consult qualified healthcare professionals for proper medical evaluation.
"""
        
        return explanation

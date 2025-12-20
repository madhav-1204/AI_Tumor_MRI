"""
Reasoning Agent - Generates medical explanations using LLM
"""

import os
import sys
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

try:
    from google import genai
    from google.genai import types
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
        
        # DEBUG: Check if key is loaded (don't print full key)
        if self.api_key:
            print(f"🔑 API Key found (length: {len(self.api_key)})")
        else:
            print("❌ No API Key found in env or init")
            
        if self.api_key and GEMINI_AVAILABLE:
            try:
                # Initialize new Gemini client
                self.client = genai.Client(api_key=self.api_key)
                
                # Universal Model Finder: List available models and pick a valid one
                print("🔄 Discovering available Gemini models...")
                try:
                    all_models = list(self.client.models.list())
                    vision_models = [m.name for m in all_models if 'generateContent' in m.supported_actions and ('flash' in m.name or 'pro' in m.name)]
                    
                    if vision_models:
                        # Prefer 1.5-flash if available, otherwise pick the first valid one
                        preferred = next((m for m in vision_models if 'gemini-1.5-flash' in m), vision_models[0])
                        self.model = preferred
                        print(f"✅ Gemini Vision API initialized using model: {self.model}")
                    else:
                        print("⚠️  No suitable Gemini models found. Using default fallback.")
                        self.model = 'gemini-1.5-flash' # Last resort
                except Exception as list_err:
                     print(f"⚠️  Model discovery failed: {list_err}. Defaulting to 'gemini-1.5-flash'")
                     self.model = 'gemini-1.5-flash'

            except Exception as e:
                print(f"⚠️  Could not initialize Gemini: {e}")
                self.client = None
                self.model = None
        else:
            print("⚠️  Gemini API not available - using fallback explanations")
            self.client = None
            self.model = None
    
    def generate_explanation(self, predicted_class, confidence, probabilities, class_names, image_path=None):
        """
        Generate medical explanation
        
        Args:
            predicted_class: Predicted tumor type
            confidence: Confidence score
            probabilities: Array of all class probabilities
            class_names: List of class names
            image_path: Path to the MRI image (for vision-based analysis)
            
        Returns:
            Explanation text
        """
        if self.client and self.model and image_path:
            return self._generate_with_llm(predicted_class, confidence, probabilities, class_names, image_path)
        else:
            return self._generate_fallback(predicted_class, confidence, probabilities, class_names)
    
    def _generate_with_llm(self, predicted_class, confidence, probabilities, class_names, image_path):
        """Generate explanation using Gemini Vision by analyzing the actual MRI image"""
        try:
            # Load and prepare the MRI image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Create probability summary
            prob_summary = "\n".join([
                f"- {name}: {prob*100:.1f}%"
                for name, prob in zip(class_names, probabilities)
            ])
            
            prompt = f"""
You are a specialized Neuroradiologist. Analyze the provided brain MRI and the AI model's data.

AI Data: Predicted Type: {predicted_class.upper()}, Confidence: {confidence:.2f}%.

Instructions:

1. **Signal Interpretation**: Describe the intensity (e.g., hyperintense, hypointense) of the mass.

2. **Morphological Features**: Note the location (supratentorial vs. infratentorial), borders (well-defined vs. infiltrative), and mass effect.

3. **Clinical Recommendation**: Suggest next steps (e.g., contrast-enhanced MRI, neurosurgical consult).

Output Format: Provide a structured 'Findings' section followed by a 'Clinical Impression'.
Please provide your analysis in the following strict structure:
            
            Findings:
            • Signal Interpretation: [Description]
            • Morphological Features:
              • Location: [Description]
              • Borders: [Description]
              • Mass Effect: [Description]
            
            Diagnostic Impression:
            [Concise summary]
            
            Differential Diagnosis:
            • [Condition 1]
            • [Condition 2]
            
            Recommendations:
            1. [Recommendation 1]
            2. [Recommendation 2]
            
            IMPORTANT:
            - Use the bullet character '•' instead of hyphens.
            - Ensure there is a COLON ':' after every heading (e.g., "Location:", "Borders:").
            - Do not include standard medical disclaimers in the text.
            """

            # Send image and prompt to Gemini
            # Use explicit types.Part construction to avoid argument errors
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text=prompt),
                            types.Part(inline_data=types.Blob(data=image_data, mime_type='image/jpeg'))
                        ]
                    )
                ]
            )
            
            return response.text
            
        except Exception as e:
            error_msg = f"Gemini Error: {str(e)}"
            print(f"⚠️  {error_msg}")
            print(f"   Error type: {type(e).__name__}")
            
            # Fallback with error info
            fallback = self._generate_fallback(predicted_class, confidence, probabilities, class_names)
            return f"{fallback}\n\n[System Note: AI Explanation failed - {error_msg}]"
    
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
        
        explanation = f"{desc} {conf_text}"
        
        return explanation

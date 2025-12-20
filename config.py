"""
Configuration file for Brain Tumor Classification System
"""

# Model Configuration
MODEL_PATH = "models/brain_tumor_resnet18.pth"  # Trained ResNet-18 for brain tumor classification
MODEL_ARCHITECTURE = "resnet18"
NUM_CLASSES = 4
MODEL_TYPE = "pytorch"

# Class Names (standard brain tumor types)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Image preprocessing settings
IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Grad-CAM settings
TARGET_LAYER_NAME = "conv_head"  # EfficientNet-B0 final conv layer

# Medical Disclaimer
MEDICAL_DISCLAIMER = """
⚠️ IMPORTANT MEDICAL DISCLAIMER ⚠️

This AI system is a DEMONSTRATION PROJECT developed for educational and 
research purposes only. It is NOT a medical device and should NOT be used 
for actual medical diagnosis or treatment decisions.

Key Limitations:
- Not FDA approved or medically validated
- May produce incorrect or misleading results
- Should never replace professional medical consultation
- Developed as a proof-of-concept only

ALWAYS consult qualified healthcare professionals for medical advice, 
diagnosis, or treatment. Never rely solely on AI predictions for health 
decisions.
"""

# LLM Configuration (Gemini API)
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_API_KEY = None  # Will be set via environment variable or user input

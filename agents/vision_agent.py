"""
Vision Agent - Handles model loading and inference (PyTorch version)
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class VisionAgent:
    """
    Vision Agent responsible for:
    1. Loading your trained EfficientNet-B2 model
    2. Preprocessing MRI images
    3. Running inference
    4. Returning predictions and probabilities
    
    Uses your actual trained model for real brain tumor predictions.
    """
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = self._get_transform()
        
    def _get_transform(self):
        """Create image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    
    def load_model(self):
        """Load your trained EfficientNet-B2 model"""
        if self.model is not None:
            return self.model
        
        try:
            print(f"🔄 Loading your trained ResNet-18 model...")
            print(f"Model path: {MODEL_PATH}")
            print(f"Device: {self.device}")
            
            # Create ResNet-18 model with 4 output classes
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)
            
            # Load your trained weights
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Your trained model loaded successfully!")
            print(f"   Architecture: ResNet-18")
            print(f"   Classes: {NUM_CLASSES}")
            print(f"   Mode: TRAINED MODEL (Real brain tumor predictions)")
            return self.model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"   Using pretrained ResNet-18 as fallback...")
            
            # Fallback: Create model with pretrained weights
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)
            self.model.to(self.device)
            self.model.eval()
            print("   ⚠️  Using ImageNet weights (not your trained model)")
            return self.model
    
    def preprocess_image(self, image_path):
        """
        Preprocess the input image
        
        Args:
            image_path: Path to the MRI image
            
        Returns:
            Preprocessed tensor ready for model input and original image
        """
        try:
            # Load and convert image to RGB
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            
            return tensor.to(self.device), image
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            raise
    
    def predict(self, image_path):
        """
        Run inference on the image
        
        Args:
            image_path: Path to the MRI image
            
        Returns:
            Dictionary containing:
                - probabilities: Numpy array of class probabilities
                - predicted_class: Name of predicted class
                - predicted_idx: Index of predicted class
                - confidence: Confidence score (0-100)
                - original_image: PIL Image object
                - preprocessed_tensor: Preprocessed image tensor
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Preprocess image
        tensor, original_image = self.preprocess_image(image_path)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Get prediction
        predicted_idx = int(probabilities.argmax())
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx] * 100)
        
        return {
            'probabilities': probabilities,
            'predicted_class': predicted_class,
            'predicted_idx': predicted_idx,
            'confidence': confidence,
            'original_image': original_image,
            'class_names': CLASS_NAMES,
            'preprocessed_tensor': tensor
        }
    
    def get_model_for_gradcam(self):
        """Return model for Grad-CAM visualization"""
        if self.model is None:
            self.load_model()
        return self.model


# Test function
def test_vision_agent():
    """Test the Vision Agent"""
    print("Testing Vision Agent...")
    print("="*50)
    
    agent = VisionAgent()
    agent.load_model()
    
    print("\n✅ Vision Agent is ready!")
    print(f"   Framework: PyTorch")
    print(f"   Device: {agent.device}")
    print(f"   Classes: {CLASS_NAMES}")
    print("\n⚠️  NOTE: Using ImageNet pretrained weights for demo")
    print("   For production, fine-tune on brain tumor dataset")


if __name__ == "__main__":
    test_vision_agent()

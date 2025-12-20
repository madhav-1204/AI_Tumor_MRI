"""
Explainability Agent - Generates Grad-CAM visualizations
"""

import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class ExplainabilityAgent:
    """
    Generates Grad-CAM heatmaps to visualize which regions
    of the MRI influenced the model's prediction
    """
    
    def __init__(self, model):
        self.model = model
        self.grad_cam = None
        
    def _get_target_layer(self):
        """Get the target layer for Grad-CAM"""
        # For ResNet-18, use the final convolutional layer (layer4)
        try:
            target_layer = self.model.layer4[-1]
            return [target_layer]
        except:
            # Fallback: try to find last conv layer
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    print(f"Using layer: {name}")
                    return [module]
            raise ValueError("Could not find suitable target layer")
    
    def generate_gradcam(self, image_tensor, predicted_idx):
        """
        Generate Grad-CAM heatmap
        
        Args:
            image_tensor: Preprocessed image tensor
            predicted_idx: Index of predicted class
            
        Returns:
            Grad-CAM visualization as numpy array
        """
        try:
            # Get target layer
            target_layers = self._get_target_layer()
            
            # Create Grad-CAM object
            cam = GradCAM(model=self.model, target_layers=target_layers)
            
            # Generate CAM
            targets = [ClassifierOutputTarget(predicted_idx)]
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]  # Get first image from batch
            
            return grayscale_cam
            
        except Exception as e:
            print(f"❌ Error generating Grad-CAM: {e}")
            return None
    
    def create_overlay(self, original_image, grayscale_cam):
        """
        Create overlay of Grad-CAM on original image
        
        Args:
            original_image: PIL Image
            grayscale_cam: Grayscale CAM array
            
        Returns:
            RGB image with CAM overlay
        """
        try:
            # Resize original image to match expected size
            img = original_image.resize(IMAGE_SIZE)
            rgb_img = np.array(img).astype(np.float32) / 255.0
            
            # Create visualization
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            return cam_image
            
        except Exception as e:
            print(f"❌ Error creating overlay: {e}")
            return None
    
    def explain(self, image_tensor, original_image, predicted_idx):
        """
        Generate complete explanation with Grad-CAM
        
        Args:
            image_tensor: Preprocessed tensor
            original_image: Original PIL Image
            predicted_idx: Predicted class index
            
        Returns:
            Dictionary with explanation visualizations
        """
        # Generate Grad-CAM
        grayscale_cam = self.generate_gradcam(image_tensor, predicted_idx)
        
        if grayscale_cam is None:
            return None
        
        # Create overlay
        cam_overlay = self.create_overlay(original_image, grayscale_cam)
        
        return {
            'heatmap': grayscale_cam,
            'overlay': cam_overlay,
            'explanation': f"The highlighted regions show areas the model focused on to make its prediction."
        }

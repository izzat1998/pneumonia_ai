import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class GradCAM:
    """Grad-CAM implementation for visualization of model predictions"""
    
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        self.model = model
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        # Find target layer if not specified
        if target_layer:
            self.target_layer = self._find_layer_by_name(model, target_layer)
        else:
            self.target_layer = self._find_target_layer(model)
        
        if self.target_layer is None:
            raise ValueError("Could not find a suitable target layer for Grad-CAM")
        
        # Register hooks
        self._register_hooks()
        
    def _find_layer_by_name(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Find layer by name in the model"""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def _find_target_layer(self, model: nn.Module) -> Optional[nn.Module]:
        """Automatically find the last convolutional layer"""
        target = None
        
        # For ResNet models
        if hasattr(model, 'layer4'):
            target = model.layer4[-1]
        # For VGG models
        elif hasattr(model, 'features'):
            for layer in reversed(list(model.features)):
                if isinstance(layer, nn.Conv2d):
                    target = layer
                    break
        # For EfficientNet models
        elif hasattr(model, 'conv_head'):
            target = model.conv_head
        # For Vision Transformer models
        elif hasattr(model, 'encoder'):
            # ViT doesn't have conv layers, use the last attention layer
            if hasattr(model.encoder, 'layer'):
                target = model.encoder.layer[-1]
        # Generic search for last conv layer
        else:
            for module in reversed(list(model.modules())):
                if isinstance(module, nn.Conv2d):
                    target = module
                    break
        
        return target
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get the predicted class if not specified
        if class_idx is None:
            if output.dim() > 2:
                output = output.squeeze()
            class_idx = output.argmax().item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients.squeeze()
        activations = self.activations.squeeze()
        
        # Global average pooling
        weights = gradients.mean(dim=(1, 2)) if gradients.dim() > 1 else gradients
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def visualize(self, image: Image.Image, input_tensor: torch.Tensor, 
                  class_idx: Optional[int] = None, alpha: float = 0.5) -> Tuple[Image.Image, np.ndarray]:
        """Generate visualization overlay"""
        
        # Generate CAM
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (image.width, image.height))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        img_array = np.array(image)
        overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
        
        result_image = Image.fromarray(overlay)
        
        return result_image, cam_resized


class PneumoniaVisualizer:
    """Specialized visualizer for pneumonia detection models"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_explanation_overlay(
        self, 
        image: Image.Image, 
        model: nn.Module, 
        input_tensor: torch.Tensor,
        prediction_result: Dict[str, Any],
        model_type: str = "cnn"
    ) -> Dict[str, Any]:
        """Create visual explanation for pneumonia prediction"""
        
        try:
            if model_type == "vit":
                # For Vision Transformers, use attention visualization
                visualization, heatmap = self._visualize_vit_attention(model, input_tensor, image)
            else:
                # For CNNs, use Grad-CAM
                gradcam = GradCAM(model)
                visualization, heatmap = gradcam.visualize(image, input_tensor)
            
            # Add annotations
            annotated_image = self._add_annotations(
                visualization, 
                prediction_result['prediction_class'],
                prediction_result['confidence_score']
            )
            
            # Identify regions of interest
            roi_info = self._identify_roi(heatmap)
            
            return {
                'visualization': annotated_image,
                'heatmap': heatmap,
                'roi_info': roi_info,
                'explanation': self._generate_explanation(
                    prediction_result['prediction_class'],
                    prediction_result['confidence_score'],
                    roi_info
                )
            }
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            return {
                'visualization': image,
                'error': str(e),
                'explanation': "Visualization not available"
            }
    
    def _visualize_vit_attention(
        self, 
        model: nn.Module, 
        input_tensor: torch.Tensor,
        image: Image.Image
    ) -> Tuple[Image.Image, np.ndarray]:
        """Visualize attention for Vision Transformer models"""
        
        # This is a simplified version - full ViT attention visualization
        # would require extracting attention weights from the model
        
        # For now, return the original image with a placeholder heatmap
        heatmap = np.ones((image.height, image.width)) * 0.5
        
        return image, heatmap
    
    def _add_annotations(
        self, 
        image: Image.Image, 
        prediction_class: str,
        confidence: float
    ) -> Image.Image:
        """Add text annotations to the image"""
        
        img_array = np.array(image)
        
        # Add prediction text
        text = f"{prediction_class.upper()} ({confidence:.1%})"
        color = (0, 255, 0) if prediction_class == "normal" else (255, 0, 0)
        
        # Add text with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(
            img_array, 
            (10, 10), 
            (10 + text_width + 10, 10 + text_height + 10),
            (0, 0, 0), 
            -1
        )
        
        # Draw text
        cv2.putText(
            img_array, 
            text, 
            (15, 10 + text_height), 
            font, 
            font_scale, 
            color, 
            thickness
        )
        
        return Image.fromarray(img_array)
    
    def _identify_roi(self, heatmap: np.ndarray) -> Dict[str, Any]:
        """Identify regions of interest from heatmap"""
        
        # Threshold the heatmap
        threshold = 0.7
        roi_mask = (heatmap > threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        roi_info = {
            'num_regions': len(contours),
            'regions': [],
            'total_area': 0
        }
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            roi_info['regions'].append({
                'bbox': (x, y, w, h),
                'area': area,
                'center': (x + w // 2, y + h // 2)
            })
            roi_info['total_area'] += area
        
        # Sort regions by area
        roi_info['regions'].sort(key=lambda r: r['area'], reverse=True)
        
        return roi_info
    
    def _generate_explanation(
        self, 
        prediction_class: str,
        confidence: float,
        roi_info: Dict[str, Any]
    ) -> str:
        """Generate textual explanation of the prediction"""
        
        if prediction_class == "pneumonia":
            if roi_info['num_regions'] > 0:
                explanation = (
                    f"The model detected pneumonia with {confidence:.1%} confidence. "
                    f"The analysis identified {roi_info['num_regions']} region(s) of concern "
                    "in the chest X-ray that show patterns consistent with pneumonia."
                )
            else:
                explanation = (
                    f"The model detected pneumonia with {confidence:.1%} confidence "
                    "based on overall patterns in the chest X-ray."
                )
        else:
            explanation = (
                f"The model classified this as normal with {confidence:.1%} confidence. "
                "No significant abnormalities were detected in the chest X-ray."
            )
        
        return explanation


# Global visualizer instance
visualizer = PneumoniaVisualizer()
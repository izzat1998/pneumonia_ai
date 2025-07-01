"""
AI Engine Module for Pneumonia Detection

This module provides advanced AI capabilities for pneumonia detection using:
- PyTorch 2.7 with torch.compile optimization
- Transformers 4.48 with latest model architectures  
- Pillow 11.2 with zlib-ng optimization
- Medical image preprocessing and enhancement
- Multi-model ensemble support
"""

from .model_loader import model_manager, ModelManager
from .predictor import predictor, PneumoniaPredictor
from .preprocessing import preprocessor, MedicalImagePreprocessor

__version__ = "1.0.0"
__author__ = "Pneumonia Detection Team"

# Export main classes and instances
__all__ = [
    'model_manager',
    'ModelManager', 
    'predictor',
    'PneumoniaPredictor',
    'preprocessor', 
    'MedicalImagePreprocessor'
]
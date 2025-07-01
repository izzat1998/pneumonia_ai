import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union
import json
from django.conf import settings
from django.core.cache import cache
import time

logger = logging.getLogger(__name__)

class ModelManager:
    """PyTorch 2.7 optimized model manager with caching and device management"""
    
    def __init__(self):
        self.device = self._get_device()
        self.models: Dict[str, torch.nn.Module] = {}
        self.processors: Dict[str, Any] = {}
        self.model_configs = self._load_model_configs()
        
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """Auto-detect optimal device with PyTorch 2.7 optimizations"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        return device
    
    def _load_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load model configurations with PyTorch 2.7 features"""
        return {
            "resnet50": {
                "name": "ResNet-50 Chest X-ray",
                "version": "v1.0",
                "source": "torchvision",
                "classes": ["normal", "pneumonia"],
                "input_size": (224, 224),
                "compile": True,
                "optimization_level": "default"
            },
            "vit_base": {
                "name": "Vision Transformer Base",
                "version": "v1.0", 
                "source": "huggingface",
                "model_id": "google/vit-base-patch16-224-in21k",
                "classes": ["normal", "pneumonia"],
                "input_size": (224, 224),
                "compile": True,
                "optimization_level": "max-autotune"
            }
        }
    
    def load_model(self, model_name: str = "resnet50") -> torch.nn.Module:
        """Load and cache model with PyTorch 2.7 compile optimization"""
        cache_key = f"model_{model_name}_{self.device}"
        
        # Check cache first
        if model_name in self.models:
            logger.info(f"Using cached model: {model_name}")
            return self.models[model_name]
        
        config = self.model_configs.get(model_name)
        if not config:
            raise ValueError(f"Unknown model: {model_name}")
        
        try:
            start_time = time.time()
            
            if config["source"] == "torchvision":
                model = self._load_torchvision_model(model_name, config)
            elif config["source"] == "huggingface":
                model = self._load_huggingface_model(model_name, config)
            else:
                raise ValueError(f"Unknown model source: {config['source']}")
            
            # PyTorch 2.7 compile optimization
            if config.get("compile", False):
                logger.info(f"Compiling model with optimization: {config.get('optimization_level', 'default')}")
                model = torch.compile(
                    model, 
                    mode=config.get("optimization_level", "default"),
                    fullgraph=True
                )
            
            # Cache model
            self.models[model_name] = model
            
            load_time = time.time() - start_time
            logger.info(f"Model {model_name} loaded and compiled in {load_time:.2f}s")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def _load_torchvision_model(self, model_name: str, config: Dict[str, Any]) -> torch.nn.Module:
        """Load ResNet-50 from torchvision with custom classifier"""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Modify final layer for binary classification
        num_classes = len(config["classes"])
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        model = model.to(self.device)
        model.eval()
        
        # Create processor
        self.processors[model_name] = transforms.Compose([
            transforms.Resize(config["input_size"]),
            transforms.CenterCrop(config["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return model
    
    def _load_huggingface_model(self, model_name: str, config: Dict[str, Any]) -> torch.nn.Module:
        """Load Vision Transformer from Hugging Face"""
        model_id = config["model_id"]
        
        # Load processor and model
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(
            model_id,
            num_labels=len(config["classes"]),
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )
        
        model = model.to(self.device)
        model.eval()
        
        # Cache processor
        self.processors[model_name] = processor
        
        return model
    
    def get_processor(self, model_name: str):
        """Get image processor for model"""
        if model_name not in self.processors:
            self.load_model(model_name)
        return self.processors[model_name]
    
    def predict(self, image_tensor: torch.Tensor, model_name: str = "resnet50") -> Dict[str, Any]:
        """Make prediction with performance monitoring"""
        model = self.load_model(model_name)
        config = self.model_configs[model_name]
        
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Ensure proper device and dtype
                if image_tensor.device != self.device:
                    image_tensor = image_tensor.to(self.device)
                
                # Add batch dimension if needed
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                
                # Make prediction
                if config["source"] == "huggingface":
                    outputs = model(image_tensor)
                    logits = outputs.logits
                else:
                    logits = model(image_tensor)
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(logits, dim=1)
                
                # Get prediction
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class_idx].item()
                
                # Map to class names
                class_names = config["classes"]
                predicted_class = class_names[predicted_class_idx]
                
                # Get all class probabilities
                class_probabilities = {
                    class_names[i]: probabilities[0, i].item() 
                    for i in range(len(class_names))
                }
                
                processing_time = time.time() - start_time
                
                return {
                    "prediction_class": predicted_class,
                    "confidence_score": confidence,
                    "class_probabilities": class_probabilities,
                    "processing_time": processing_time,
                    "model_version": f"{config['name']}-{config['version']}",
                    "device": str(self.device)
                }
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_model_info(self, model_name: str = "resnet50") -> Dict[str, Any]:
        """Get model information and statistics"""
        config = self.model_configs.get(model_name, {})
        
        info = {
            "name": config.get("name", "Unknown"),
            "version": config.get("version", "Unknown"),
            "source": config.get("source", "Unknown"),
            "classes": config.get("classes", []),
            "input_size": config.get("input_size", (224, 224)),
            "device": str(self.device),
            "loaded": model_name in self.models,
            "compile_enabled": config.get("compile", False)
        }
        
        if model_name in self.models:
            model = self.models[model_name]
            info["parameters"] = sum(p.numel() for p in model.parameters())
            info["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return info
    
    def warmup_model(self, model_name: str = "resnet50") -> float:
        """Warm up model for optimal performance"""
        model = self.load_model(model_name)
        config = self.model_configs[model_name]
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *config["input_size"]).to(self.device)
        
        # Warmup runs
        warmup_times = []
        for _ in range(3):
            start_time = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            warmup_times.append(time.time() - start_time)
        
        avg_warmup_time = sum(warmup_times) / len(warmup_times)
        logger.info(f"Model {model_name} warmed up. Avg inference time: {avg_warmup_time:.3f}s")
        
        return avg_warmup_time
    
    def clear_cache(self):
        """Clear model cache to free memory"""
        self.models.clear()
        self.processors.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model cache cleared")


# Global model manager instance with fallback to mock
try:
    model_manager = ModelManager()
    logger.info("Real ModelManager initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize ModelManager: {e}. Using mock model for testing.")
    from .mock_model import mock_model_manager
    model_manager = mock_model_manager
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import time
import random
import logging

logger = logging.getLogger(__name__)

class MockPneumoniaModel(nn.Module):
    """Mock model for testing pneumonia detection MVP"""
    
    def __init__(self, num_classes=2):
        super(MockPneumoniaModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MockModelManager:
    """Mock model manager for MVP testing with realistic behavior"""
    
    def __init__(self):
        self.device = self._get_device()
        self.models = {}
        self.model_configs = {
            "resnet50": {
                "name": "Mock ResNet-50 Chest X-ray",
                "version": "v1.0-mock",
                "source": "mock",
                "classes": ["normal", "pneumonia"],
                "input_size": (224, 224),
                "compile": False
            }
        }
        logger.info(f"MockModelManager initialized with device: {self.device}")
    
    def _get_device(self):
        """Auto-detect optimal device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        return device
    
    def load_model(self, model_name="resnet50"):
        """Load mock model for testing"""
        if model_name in self.models:
            logger.info(f"Using cached mock model: {model_name}")
            return self.models[model_name]
        
        try:
            logger.info(f"Loading mock model: {model_name}")
            
            # Create mock model
            model = MockPneumoniaModel(num_classes=2)
            model = model.to(self.device)
            model.eval()
            
            # Cache model
            self.models[model_name] = model
            
            logger.info(f"Mock model {model_name} loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading mock model {model_name}: {str(e)}")
            raise
    
    def get_processor(self, model_name):
        """Get mock image processor"""
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image_tensor, model_name="resnet50"):
        """Make realistic mock prediction"""
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
                
                # Make mock prediction with realistic behavior
                # Analyze image characteristics for more realistic results
                image_np = image_tensor.cpu().numpy()
                image_mean = np.mean(image_np)
                image_std = np.std(image_np)
                
                # Create semi-realistic prediction based on image characteristics
                # Dark images might be more likely to be pneumonia (simplified heuristic)
                darkness_factor = 1.0 - (image_mean / 255.0) if image_mean < 255 else 1.0 - image_mean
                noise_factor = random.uniform(0.1, 0.3)
                
                # Generate probabilities
                if darkness_factor > 0.6:  # Darker images more likely pneumonia
                    pneumonia_prob = 0.7 + random.uniform(-0.2, 0.2)
                else:
                    pneumonia_prob = 0.3 + random.uniform(-0.2, 0.2)
                
                # Ensure probabilities are valid
                pneumonia_prob = max(0.1, min(0.9, pneumonia_prob))
                normal_prob = 1.0 - pneumonia_prob
                
                # Create logits that would produce these probabilities
                logits = torch.tensor([[
                    np.log(normal_prob + 1e-8),
                    np.log(pneumonia_prob + 1e-8)
                ]])
                
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
                
                # Add some realistic processing time
                processing_time = time.time() - start_time + random.uniform(0.1, 0.5)
                
                result = {
                    "prediction_class": predicted_class,
                    "confidence_score": confidence,
                    "class_probabilities": class_probabilities,
                    "processing_time": processing_time,
                    "model_version": f"{config['name']}-{config['version']}",
                    "device": str(self.device)
                }
                
                logger.info(f"Mock prediction: {predicted_class} ({confidence:.3f} confidence)")
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Mock prediction error: {str(e)}")
            raise RuntimeError(f"Mock prediction failed: {str(e)}")
    
    def get_model_info(self, model_name="resnet50"):
        """Get mock model information"""
        config = self.model_configs.get(model_name, {})
        
        info = {
            "name": config.get("name", "Unknown"),
            "version": config.get("version", "Unknown"),
            "source": config.get("source", "mock"),
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
    
    def warmup_model(self, model_name="resnet50"):
        """Warm up mock model"""
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
        logger.info(f"Mock model {model_name} warmed up. Avg inference time: {avg_warmup_time:.3f}s")
        
        return avg_warmup_time
    
    def clear_cache(self):
        """Clear model cache"""
        self.models.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Mock model cache cleared")

# Create mock instance for testing
mock_model_manager = MockModelManager()
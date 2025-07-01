import torch
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path
import time
from django.core.files.uploadedfile import InMemoryUploadedFile
from .model_loader import model_manager
# Import moved to avoid circular import issues
try:
    from .medical_predictor import medical_predictor
    MEDICAL_MODE_AVAILABLE = medical_predictor is not None
except ImportError:
    MEDICAL_MODE_AVAILABLE = False
    medical_predictor = None

logger = logging.getLogger(__name__)

class PneumoniaPredictor:
    """High-performance pneumonia detection predictor with PyTorch 2.7 optimizations"""
    
    def __init__(self, default_model: str = "huggingface_vit"):
        self.default_model = default_model
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Warmup default model for optimal performance
        # Skip warmup for medical models as they're handled by medical_model_manager
        if self.default_model not in ['huggingface_vit', 'torchxrayvision', 'efficientnet_b4']:
            try:
                model_manager.warmup_model(self.default_model)
            except Exception as e:
                logger.warning(f"Model warmup failed: {e}")
    
    def predict_from_file(
        self, 
        image_file: Union[InMemoryUploadedFile, str, Path],
        model_name: str = None,
        session_key: str = None,
        request_meta: Dict[str, Any] = None,
        use_medical_grade: bool = True
    ) -> Dict[str, Any]:
        """Main prediction method from uploaded file"""
        
        # Try medical-grade prediction first if available
        if use_medical_grade and MEDICAL_MODE_AVAILABLE:
            try:
                logger.info("Using medical-grade prediction")
                return medical_predictor.predict_from_file(
                    image_file, model_name, session_key, request_meta
                )
            except Exception as e:
                logger.warning(f"Medical prediction failed, falling back to standard: {e}")
        
        # Fallback to standard prediction
        logger.info("Using standard prediction (non-medical)")
        
        model_name = model_name or self.default_model
        start_time = time.time()
        
        try:
            # Load and validate image
            image, image_info = self._load_and_validate_image(image_file)
            
            # Preprocess image
            processed_tensor = self._preprocess_image(image, model_name)
            
            # Make prediction
            prediction_result = model_manager.predict(processed_tensor, model_name)
            
            # Prepare complete result
            result = {
                **prediction_result,
                **image_info,
                'total_processing_time': time.time() - start_time,
                'session_key': session_key,
                'model_type': 'standard_grade',
                'warning': 'Using non-medical model - for research purposes only'
            }
            
            # Log prediction
            self._log_prediction(result, request_meta)
            
            return result
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg)
            
            # Log error
            self._log_error(error_msg, session_key, request_meta)
            
            raise RuntimeError(error_msg)
    
    def _load_and_validate_image(self, image_file) -> tuple[Image.Image, Dict[str, Any]]:
        """Load and validate image file with comprehensive checks"""
        
        try:
            # Handle different input types
            if isinstance(image_file, (str, Path)):
                image_path = Path(image_file)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                image = Image.open(image_path)
                filename = image_path.name
                file_size = image_path.stat().st_size
                
            elif hasattr(image_file, 'read'):
                # Handle uploaded file
                image = Image.open(image_file)
                filename = getattr(image_file, 'name', 'unknown.jpg')
                file_size = getattr(image_file, 'size', 0)
                
            else:
                raise ValueError("Unsupported image file type")
            
            # Validate file format
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {file_ext}")
            
            # Validate image properties
            if image.mode not in ['RGB', 'L', 'RGBA']:
                logger.info(f"Converting image mode from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validate image dimensions
            width, height = image.size
            if width < 32 or height < 32:
                raise ValueError(f"Image too small: {width}x{height}. Minimum size: 32x32")
            
            if width > 4096 or height > 4096:
                logger.warning(f"Large image detected: {width}x{height}. Consider resizing for better performance.")
            
            # Image quality assessment
            image_quality = self._assess_image_quality(image)
            
            image_info = {
                'original_filename': filename,
                'file_size': file_size,
                'image_width': width,
                'image_height': height,
                'image_mode': image.mode,
                'image_quality': image_quality
            }
            
            logger.info(f"Image loaded: {filename} ({width}x{height}, {file_size} bytes)")
            
            return image, image_info
            
        except Exception as e:
            logger.error(f"Image loading failed: {str(e)}")
            raise ValueError(f"Invalid image file: {str(e)}")
    
    def _assess_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Assess image quality for medical analysis"""
        
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Calculate basic statistics
        mean_brightness = np.mean(img_array)
        std_brightness = np.std(img_array)
        
        # Contrast assessment
        contrast_score = std_brightness / (mean_brightness + 1e-8)
        
        # Sharpness assessment (simplified Laplacian variance)
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
            
        laplacian_var = np.var(np.gradient(gray))
        
        # Quality scores (0-1 scale)
        brightness_score = min(1.0, mean_brightness / 128.0)
        contrast_score = min(1.0, contrast_score / 0.5)
        sharpness_score = min(1.0, laplacian_var / 1000.0)
        
        overall_quality = (brightness_score + contrast_score + sharpness_score) / 3.0
        
        return {
            'brightness_score': round(brightness_score, 3),
            'contrast_score': round(contrast_score, 3),
            'sharpness_score': round(sharpness_score, 3),
            'overall_quality': round(overall_quality, 3),
            'quality_rating': self._get_quality_rating(overall_quality)
        }
    
    def _get_quality_rating(self, score: float) -> str:
        """Convert quality score to rating"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _preprocess_image(self, image: Image.Image, model_name: str) -> torch.Tensor:
        """Preprocess image for model inference with optimizations"""
        
        processor = model_manager.get_processor(model_name)
        
        try:
            # Apply model-specific preprocessing
            if hasattr(processor, 'feature_extractor') or hasattr(processor, 'image_processor'):
                # Hugging Face processor
                processed = processor(images=image, return_tensors="pt")
                tensor = processed['pixel_values']
            else:
                # torchvision transforms (Compose object)
                tensor = processor(image).unsqueeze(0)
            
            # Ensure tensor is on correct device
            tensor = tensor.to(model_manager.device)
            
            logger.debug(f"Image preprocessed: {tensor.shape}, device: {tensor.device}")
            
            return tensor
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ValueError(f"Image preprocessing error: {str(e)}")
    
    def _log_prediction(self, result: Dict[str, Any], request_meta: Dict[str, Any] = None):
        """Log successful prediction"""
        
        try:
            # Import here to avoid circular imports
            from core.models import SystemLog, ModelMetrics
            
            # Create system log entry
            SystemLog.objects.create(
                level='INFO',
                message=f"Prediction completed: {result['prediction_class']} ({result['confidence_score']:.3f})",
                category='prediction',
                session_key=result.get('session_key', ''),
                ip_address=request_meta.get('REMOTE_ADDR') if request_meta else None,
                user_agent=request_meta.get('HTTP_USER_AGENT', '') if request_meta else '',
                extra_data={
                    'model_version': result.get('model_version'),
                    'processing_time': result.get('processing_time'),
                    'image_quality': result.get('image_quality', {}),
                    'device': result.get('device')
                }
            )
            
            # Update model metrics
            model_name = result.get('model_version', 'unknown')
            version = result.get('model_version', 'v1.0').split('-')[-1] if '-' in result.get('model_version', '') else 'v1.0'
            
            metrics, created = ModelMetrics.objects.get_or_create(
                model_name=model_name,
                version=version,
                defaults={
                    'total_predictions': 0,
                    'avg_processing_time': 0.0,
                    'avg_confidence_score': 0.0,
                    'normal_predictions': 0,
                    'pneumonia_predictions': 0,
                    'high_confidence_predictions': 0
                }
            )
            
            # Create a mock result object for metrics update
            class MockResult:
                def __init__(self, data):
                    self.prediction_class = data['prediction_class']
                    self.confidence_score = data['confidence_score']
                    self.processing_time = data.get('processing_time', 0)
            
            mock_result = MockResult(result)
            metrics.update_metrics(mock_result)
            
        except Exception as e:
            logger.error(f"Failed to log prediction: {str(e)}")
    
    def _log_error(self, error_msg: str, session_key: str = None, request_meta: Dict[str, Any] = None):
        """Log prediction error"""
        
        try:
            # Import here to avoid circular imports
            from core.models import SystemLog
            
            SystemLog.objects.create(
                level='ERROR',
                message=error_msg,
                category='prediction_error',
                session_key=session_key or '',
                ip_address=request_meta.get('REMOTE_ADDR') if request_meta else None,
                user_agent=request_meta.get('HTTP_USER_AGENT', '') if request_meta else '',
                extra_data={'error_type': 'prediction_failure'}
            )
        except Exception as e:
            logger.error(f"Failed to log error: {str(e)}")
    
    def batch_predict(
        self, 
        image_files: list,
        model_name: str = None,
        session_key: str = None,
        request_meta: Dict[str, Any] = None
    ) -> list[Dict[str, Any]]:
        """Batch prediction for multiple images"""
        
        model_name = model_name or self.default_model
        results = []
        
        for i, image_file in enumerate(image_files):
            try:
                result = self.predict_from_file(
                    image_file, 
                    model_name, 
                    session_key, 
                    request_meta
                )
                result['batch_index'] = i
                results.append(result)
                
            except Exception as e:
                error_result = {
                    'batch_index': i,
                    'error': str(e),
                    'prediction_class': 'error',
                    'confidence_score': 0.0,
                    'session_key': session_key
                }
                results.append(error_result)
                logger.error(f"Batch prediction failed for image {i}: {str(e)}")
        
        return results
    
    def get_supported_formats(self) -> list[str]:
        """Get list of supported image formats"""
        return list(self.supported_formats)
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """Get information about available models"""
        model_name = model_name or self.default_model
        return model_manager.get_model_info(model_name)


# Global predictor instance
predictor = PneumoniaPredictor()
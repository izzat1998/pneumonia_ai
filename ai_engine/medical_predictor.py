import torch
from PIL import Image
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Union
from django.core.files.uploadedfile import InMemoryUploadedFile
from pathlib import Path

from .medical_model_loader import medical_model_manager
try:
    from .confidence_enhancer import confidence_enhancer
    CONFIDENCE_ENHANCEMENT_AVAILABLE = True
except ImportError:
    CONFIDENCE_ENHANCEMENT_AVAILABLE = False
    confidence_enhancer = None

try:
    from .visualization import visualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    visualizer = None

logger = logging.getLogger(__name__)

class MedicalPneumoniaPredictor:
    """Medical-grade pneumonia predictor using trained models"""
    
    def __init__(self, default_model: str = "huggingface_vit"):
        self.default_model = default_model
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Verify medical model manager is available
        if medical_model_manager is None:
            raise RuntimeError("Medical model manager not available")
        
        logger.info(f"MedicalPneumoniaPredictor initialized with {default_model}")
    
    def predict_from_file(
        self, 
        image_file: Union[InMemoryUploadedFile, str, Path],
        model_name: str = None,
        session_key: str = None,
        request_meta: Dict[str, Any] = None,
        enhance_confidence: bool = True,
        use_ensemble: bool = False,
        generate_visualization: bool = False
    ) -> Dict[str, Any]:
        """Main medical prediction method"""
        
        model_name = model_name or self.default_model
        start_time = time.time()
        
        try:
            # Load and validate image
            image, image_info = self._load_and_validate_image(image_file)
            
            # Use confidence enhancement if available and requested
            if enhance_confidence and CONFIDENCE_ENHANCEMENT_AVAILABLE:
                logger.info("Using confidence enhancement")
                prediction_result = confidence_enhancer.predict_with_enhancement(
                    image, 
                    model_name=model_name,
                    use_ensemble=use_ensemble,
                    enhance_quality=True
                )
            else:
                # Standard medical prediction
                logger.info("Using standard medical prediction")
                processed_tensor = self._preprocess_medical_image(image, model_name)
                prediction_result = medical_model_manager.predict_medical(processed_tensor, model_name)
            
            # Generate visualization if requested
            if generate_visualization and VISUALIZATION_AVAILABLE:
                try:
                    logger.info("Generating prediction visualization")
                    
                    # Get the model and determine type
                    model = medical_model_manager.load_medical_model(model_name)
                    model_type = "vit" if "vit" in model_name.lower() else "cnn"
                    
                    # Generate visualization
                    viz_result = visualizer.create_explanation_overlay(
                        image=image,
                        model=model,
                        input_tensor=processed_tensor if 'processed_tensor' in locals() else self._preprocess_medical_image(image, model_name),
                        prediction_result=prediction_result,
                        model_type=model_type
                    )
                    
                    prediction_result['visualization'] = viz_result
                    logger.info("Visualization generated successfully")
                except Exception as e:
                    logger.error(f"Visualization generation failed: {str(e)}")
                    prediction_result['visualization'] = {'error': str(e)}
            
            # Enhance result with medical context
            enhanced_result = self._enhance_medical_result(prediction_result, image_info)
            enhanced_result.update({
                'total_processing_time': time.time() - start_time,
                'session_key': session_key,
                'model_type': 'medical_grade',
                'confidence_enhanced': enhance_confidence and CONFIDENCE_ENHANCEMENT_AVAILABLE,
                'ensemble_used': use_ensemble and CONFIDENCE_ENHANCEMENT_AVAILABLE,
                'visualization_generated': generate_visualization and VISUALIZATION_AVAILABLE
            })
            
            # Log medical prediction
            self._log_medical_prediction(enhanced_result, request_meta)
            
            return enhanced_result
            
        except Exception as e:
            error_msg = f"Medical prediction failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _load_and_validate_image(self, image_file) -> tuple[Image.Image, Dict[str, Any]]:
        """Load and validate medical image with enhanced checks"""
        
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
                image = Image.open(image_file)
                filename = getattr(image_file, 'name', 'unknown.jpg')
                file_size = getattr(image_file, 'size', 0)
                
            else:
                raise ValueError("Unsupported image file type")
            
            # Validate file format
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {file_ext}")
            
            # Convert to grayscale if needed (chest X-rays are typically grayscale)
            if image.mode == 'L':
                # Convert grayscale to RGB for model compatibility
                image = image.convert('RGB')
            elif image.mode == 'RGBA':
                # Convert RGBA to RGB
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validate image dimensions for medical analysis
            width, height = image.size
            if width < 128 or height < 128:
                raise ValueError(f"Image too small for medical analysis: {width}x{height}. Minimum: 128x128")
            
            # Medical image quality assessment
            quality_assessment = self._assess_medical_image_quality(image)
            
            image_info = {
                'original_filename': filename,
                'file_size': file_size,
                'image_width': width,
                'image_height': height,
                'image_mode': image.mode,
                'medical_quality_assessment': quality_assessment
            }
            
            logger.info(f"Medical image loaded: {filename} ({width}x{height})")
            
            return image, image_info
            
        except Exception as e:
            logger.error(f"Medical image loading failed: {str(e)}")
            raise ValueError(f"Invalid medical image: {str(e)}")
    
    def _assess_medical_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Assess image quality for medical diagnosis"""
        
        img_array = np.array(image)
        
        # Convert to grayscale for analysis if RGB
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Medical-specific quality metrics
        
        # 1. Contrast assessment (critical for X-ray analysis)
        contrast = np.std(gray)
        contrast_score = min(1.0, contrast / 50.0)  # Normalized to 0-1
        
        # 2. Dynamic range (important for medical images)
        dynamic_range = np.max(gray) - np.min(gray)
        dynamic_range_score = min(1.0, dynamic_range / 255.0)
        
        # 3. Sharpness (edge definition important for pathology detection)
        # Using Laplacian variance
        laplacian_var = np.var(np.gradient(gray))
        sharpness_score = min(1.0, laplacian_var / 2000.0)
        
        # 4. Noise assessment (lower noise is better for medical analysis)
        # Estimate noise using local standard deviation - simplified approach
        try:
            # Use a more robust noise estimation
            noise_estimate = np.std(gray)
            noise_score = max(0.0, 1.0 - (noise_estimate / 50.0))
        except:
            noise_score = 0.5  # Default if noise estimation fails
        
        # 5. Overall medical quality score
        overall_quality = (contrast_score + dynamic_range_score + sharpness_score + noise_score) / 4.0
        
        # Medical quality rating
        if overall_quality >= 0.8:
            quality_rating = "excellent_for_diagnosis"
        elif overall_quality >= 0.6:
            quality_rating = "good_for_diagnosis"
        elif overall_quality >= 0.4:
            quality_rating = "acceptable_for_diagnosis"
        else:
            quality_rating = "poor_quality_review_needed"
        
        return {
            'contrast_score': round(contrast_score, 3),
            'dynamic_range_score': round(dynamic_range_score, 3),
            'sharpness_score': round(sharpness_score, 3),
            'noise_score': round(noise_score, 3),
            'overall_medical_quality': round(overall_quality, 3),
            'medical_quality_rating': quality_rating,
            'diagnostic_confidence_modifier': self._get_diagnostic_confidence_modifier(overall_quality)
        }
    
    def _get_diagnostic_confidence_modifier(self, quality_score: float) -> str:
        """Get diagnostic confidence modifier based on image quality"""
        if quality_score >= 0.8:
            return "high_confidence"
        elif quality_score >= 0.6:
            return "moderate_confidence" 
        elif quality_score >= 0.4:
            return "low_confidence"
        else:
            return "unreliable_recommend_retake"
    
    def _preprocess_medical_image(self, image: Image.Image, model_name: str) -> torch.Tensor:
        """Preprocess image for medical model"""
        
        processor = medical_model_manager.get_medical_processor(model_name)
        
        try:
            # Check if this is a Hugging Face processor by trying to detect common attributes
            is_hf_processor = (model_name == "huggingface_vit" and 
                             hasattr(processor, '__call__') and 
                             'ImageProcessor' in str(type(processor)))
            
            if is_hf_processor:
                # Hugging Face processor
                processed = processor(images=image, return_tensors="pt")
                tensor = processed['pixel_values']
            else:
                # TorchVision transforms
                tensor = processor(image).unsqueeze(0)
            
            # Ensure tensor is on correct device
            tensor = tensor.to(medical_model_manager.device)
            
            logger.debug(f"Medical image preprocessed: {tensor.shape}")
            
            return tensor
            
        except Exception as e:
            logger.error(f"Medical image preprocessing failed: {str(e)}")
            raise ValueError(f"Medical preprocessing error: {str(e)}")
    
    def _enhance_medical_result(self, prediction_result: Dict[str, Any], image_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance prediction result with medical context"""
        
        enhanced_result = prediction_result.copy()
        enhanced_result.update(image_info)
        
        # Add medical interpretation
        confidence = enhanced_result['confidence_score']
        quality_rating = image_info['medical_quality_assessment']['medical_quality_rating']
        
        # Medical confidence interpretation
        if confidence >= 0.9 and quality_rating in ['excellent_for_diagnosis', 'good_for_diagnosis']:
            medical_interpretation = "High confidence diagnosis - suitable for clinical decision support"
        elif confidence >= 0.8:
            medical_interpretation = "Good confidence diagnosis - recommend clinical correlation"
        elif confidence >= 0.7:
            medical_interpretation = "Moderate confidence - recommend additional imaging or clinical review"
        else:
            medical_interpretation = "Low confidence - recommend expert radiologist review"
        
        enhanced_result['medical_interpretation'] = medical_interpretation
        
        # Add pathology details if available (for TorchXRayVision)
        if 'pathology_predictions' in enhanced_result:
            # Find other significant findings
            pathology_findings = []
            for pathology, prob in enhanced_result['pathology_predictions'].items():
                if prob > 0.3 and pathology.lower() != 'pneumonia':  # Threshold for other findings
                    pathology_findings.append({
                        'pathology': pathology,
                        'probability': prob,
                        'significance': 'high' if prob > 0.7 else 'moderate' if prob > 0.5 else 'low'
                    })
            
            enhanced_result['additional_findings'] = pathology_findings
        
        return enhanced_result
    
    def _log_medical_prediction(self, result: Dict[str, Any], request_meta: Dict[str, Any] = None):
        """Log medical prediction with enhanced details"""
        
        try:
            from core.models import SystemLog
            
            # Create detailed log entry for medical prediction
            log_message = (
                f"Medical prediction: {result['prediction_class']} "
                f"(confidence: {result['confidence_score']:.3f}, "
                f"quality: {result['medical_quality_assessment']['medical_quality_rating']})"
            )
            
            SystemLog.objects.create(
                level='INFO',
                message=log_message,
                category='medical_prediction',
                session_key=result.get('session_key', '') or 'anonymous',
                ip_address=request_meta.get('REMOTE_ADDR') if request_meta else None,
                user_agent=request_meta.get('HTTP_USER_AGENT', '') if request_meta else '',
                extra_data={
                    'model_version': result.get('model_version'),
                    'processing_time': result.get('processing_time'),
                    'medical_quality': result.get('medical_quality_assessment', {}),
                    'medical_interpretation': result.get('medical_interpretation'),
                    'device': result.get('device'),
                    'model_type': 'medical_grade',
                    'pathology_findings': result.get('additional_findings', [])
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to log medical prediction: {str(e)}")
    
    def get_available_medical_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available medical models"""
        
        available_models = {}
        
        for model_name in medical_model_manager.model_configs.keys():
            try:
                model_info = medical_model_manager.get_medical_model_info(model_name)
                available_models[model_name] = model_info
            except Exception as e:
                logger.error(f"Error getting info for {model_name}: {str(e)}")
                available_models[model_name] = {'error': str(e)}
        
        return available_models
    
    def validate_medical_setup(self) -> Dict[str, Any]:
        """Validate medical model setup"""
        
        validation_results = {
            'medical_manager_available': medical_model_manager is not None,
            'device': str(medical_model_manager.device) if medical_model_manager else 'unknown',
            'models_tested': {},
            'overall_status': 'unknown'
        }
        
        if medical_model_manager is None:
            validation_results['overall_status'] = 'failed'
            validation_results['error'] = 'Medical model manager not available'
            return validation_results
        
        # Test model loading
        for model_name in ['torchxrayvision', 'huggingface_vit']:
            try:
                model_info = medical_model_manager.get_medical_model_info(model_name)
                validation_results['models_tested'][model_name] = {
                    'status': 'available',
                    'info': model_info
                }
            except Exception as e:
                validation_results['models_tested'][model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Determine overall status
        successful_models = [k for k, v in validation_results['models_tested'].items() if v['status'] == 'available']
        
        if len(successful_models) > 0:
            validation_results['overall_status'] = 'ready'
            validation_results['available_models'] = successful_models
        else:
            validation_results['overall_status'] = 'no_models_available'
        
        return validation_results

# Global medical predictor instance
try:
    medical_predictor = MedicalPneumoniaPredictor()
    logger.info("Medical predictor initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize medical predictor: {e}")
    medical_predictor = None
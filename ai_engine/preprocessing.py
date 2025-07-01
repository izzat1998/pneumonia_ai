import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, Any, Tuple, Optional, Union
import cv2
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MedicalImagePreprocessor:
    """Advanced medical image preprocessing with Pillow 11.2 optimizations"""
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
    def enhance_chest_xray(
        self, 
        image: Image.Image,
        apply_clahe: bool = True,
        normalize_contrast: bool = True,
        reduce_noise: bool = True,
        enhance_edges: bool = False
    ) -> Image.Image:
        """Enhance chest X-ray image for better pneumonia detection"""
        
        try:
            # Convert to grayscale for processing, then back to RGB
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image.copy()
            
            # Convert to numpy array for advanced processing
            img_array = np.array(gray_image)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if apply_clahe:
                img_array = self.clahe.apply(img_array)
                logger.debug("Applied CLAHE enhancement")
            
            # Convert back to PIL Image
            enhanced_image = Image.fromarray(img_array, mode='L')
            
            # Normalize contrast
            if normalize_contrast:
                enhancer = ImageEnhance.Contrast(enhanced_image)
                enhanced_image = enhancer.enhance(1.2)  # Increase contrast by 20%
                logger.debug("Applied contrast normalization")
            
            # Noise reduction
            if reduce_noise:
                enhanced_image = enhanced_image.filter(ImageFilter.MedianFilter(size=3))
                logger.debug("Applied noise reduction")
            
            # Edge enhancement (optional, can help with pneumonia detection)
            if enhance_edges:
                enhancer = ImageEnhance.Sharpness(enhanced_image)
                enhanced_image = enhancer.enhance(1.1)  # Slight sharpening
                logger.debug("Applied edge enhancement")
            
            # Convert back to RGB for model input
            if enhanced_image.mode != 'RGB':
                enhanced_image = enhanced_image.convert('RGB')
            
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {str(e)}")
            # Return original image if enhancement fails
            return image.convert('RGB') if image.mode != 'RGB' else image
    
    def detect_lung_region(self, image: Image.Image) -> Optional[Image.Image]:
        """Detect and crop to lung region (simplified implementation)"""
        
        try:
            # Convert to grayscale
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
            
            # Apply Otsu's thresholding
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image
            
            # Find the largest contour (assumed to be lung region)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_array.shape[1] - x, w + 2 * padding)
            h = min(img_array.shape[0] - y, h + 2 * padding)
            
            # Crop image
            cropped_image = image.crop((x, y, x + w, y + h))
            
            logger.debug(f"Lung region detected and cropped: {x}, {y}, {w}, {h}")
            
            return cropped_image
            
        except Exception as e:
            logger.warning(f"Lung region detection failed: {str(e)}. Using original image.")
            return image
    
    def normalize_intensity(self, image: Image.Image) -> Image.Image:
        """Normalize image intensity for consistent analysis"""
        
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            if len(img_array.shape) == 3:
                # RGB image
                normalized = np.zeros_like(img_array)
                for i in range(3):
                    channel = img_array[:, :, i].astype(np.float32)
                    # Normalize to 0-255 range
                    channel_min, channel_max = channel.min(), channel.max()
                    if channel_max > channel_min:
                        channel = ((channel - channel_min) / (channel_max - channel_min)) * 255
                    normalized[:, :, i] = channel.astype(np.uint8)
            else:
                # Grayscale image
                channel = img_array.astype(np.float32)
                channel_min, channel_max = channel.min(), channel.max()
                if channel_max > channel_min:
                    normalized = ((channel - channel_min) / (channel_max - channel_min)) * 255
                else:
                    normalized = channel
                normalized = normalized.astype(np.uint8)
            
            result_image = Image.fromarray(normalized)
            logger.debug("Applied intensity normalization")
            
            return result_image
            
        except Exception as e:
            logger.error(f"Intensity normalization failed: {str(e)}")
            return image
    
    def resize_with_aspect_ratio(
        self, 
        image: Image.Image, 
        target_size: Tuple[int, int] = (224, 224),
        maintain_aspect: bool = True
    ) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        
        try:
            if not maintain_aspect:
                return image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Calculate scaling factor to maintain aspect ratio
            original_width, original_height = image.size
            target_width, target_height = target_size
            
            # Calculate ratios
            width_ratio = target_width / original_width
            height_ratio = target_height / original_height
            
            # Use smaller ratio to ensure image fits within target size
            scale_ratio = min(width_ratio, height_ratio)
            
            # Calculate new dimensions
            new_width = int(original_width * scale_ratio)
            new_height = int(original_height * scale_ratio)
            
            # Resize image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste resized image in center
            final_image = Image.new('RGB', target_size, (0, 0, 0))
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            final_image.paste(resized_image, (paste_x, paste_y))
            
            logger.debug(f"Resized image: {original_width}x{original_height} -> {target_width}x{target_height}")
            
            return final_image
            
        except Exception as e:
            logger.error(f"Image resizing failed: {str(e)}")
            # Fallback to simple resize
            return image.resize(target_size, Image.Resampling.LANCZOS)
    
    def apply_data_augmentation(
        self, 
        image: Image.Image,
        rotation_range: int = 10,
        brightness_range: float = 0.1,
        contrast_range: float = 0.1
    ) -> Image.Image:
        """Apply data augmentation for training (optional for inference)"""
        
        try:
            augmented_image = image.copy()
            
            # Random rotation
            if rotation_range > 0:
                angle = np.random.uniform(-rotation_range, rotation_range)
                augmented_image = augmented_image.rotate(angle, Image.Resampling.BILINEAR, fillcolor=0)
            
            # Random brightness adjustment
            if brightness_range > 0:
                brightness_factor = np.random.uniform(1 - brightness_range, 1 + brightness_range)
                enhancer = ImageEnhance.Brightness(augmented_image)
                augmented_image = enhancer.enhance(brightness_factor)
            
            # Random contrast adjustment
            if contrast_range > 0:
                contrast_factor = np.random.uniform(1 - contrast_range, 1 + contrast_range)
                enhancer = ImageEnhance.Contrast(augmented_image)
                augmented_image = enhancer.enhance(contrast_factor)
            
            logger.debug("Applied data augmentation")
            
            return augmented_image
            
        except Exception as e:
            logger.error(f"Data augmentation failed: {str(e)}")
            return image
    
    def preprocess_for_model(
        self, 
        image: Image.Image,
        target_size: Tuple[int, int] = (224, 224),
        enhance: bool = True,
        crop_lungs: bool = False
    ) -> Image.Image:
        """Complete preprocessing pipeline for model input"""
        
        try:
            processed_image = image.copy()
            
            # Step 1: Detect and crop lung region (optional)
            if crop_lungs:
                processed_image = self.detect_lung_region(processed_image)
            
            # Step 2: Enhance image for medical analysis
            if enhance:
                processed_image = self.enhance_chest_xray(processed_image)
            
            # Step 3: Normalize intensity
            processed_image = self.normalize_intensity(processed_image)
            
            # Step 4: Resize to target size
            processed_image = self.resize_with_aspect_ratio(processed_image, target_size)
            
            # Step 5: Ensure RGB format
            if processed_image.mode != 'RGB':
                processed_image = processed_image.convert('RGB')
            
            logger.info("Completed preprocessing pipeline")
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {str(e)}")
            # Fallback to basic processing
            return image.convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
    
    def get_image_statistics(self, image: Image.Image) -> Dict[str, Any]:
        """Get comprehensive image statistics"""
        
        try:
            img_array = np.array(image)
            
            if len(img_array.shape) == 3:
                # RGB image
                stats = {}
                for i, channel in enumerate(['R', 'G', 'B']):
                    channel_data = img_array[:, :, i]
                    stats[f'{channel}_mean'] = float(np.mean(channel_data))
                    stats[f'{channel}_std'] = float(np.std(channel_data))
                    stats[f'{channel}_min'] = int(np.min(channel_data))
                    stats[f'{channel}_max'] = int(np.max(channel_data))
                
                # Overall statistics
                stats['overall_mean'] = float(np.mean(img_array))
                stats['overall_std'] = float(np.std(img_array))
                
            else:
                # Grayscale image
                stats = {
                    'mean': float(np.mean(img_array)),
                    'std': float(np.std(img_array)),
                    'min': int(np.min(img_array)),
                    'max': int(np.max(img_array))
                }
            
            # Additional statistics
            stats['width'], stats['height'] = image.size
            stats['mode'] = image.mode
            stats['format'] = image.format
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate image statistics: {str(e)}")
            return {}


# Global preprocessor instance
preprocessor = MedicalImagePreprocessor()
"""
Dataset Processor for Medical Validation
Handles loading and processing of chest X-ray test datasets for model validation.
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional, Iterator
import json
from datetime import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings

logger = logging.getLogger(__name__)

class MedicalDatasetProcessor:
    """
    Processes medical datasets for validation, ensuring proper loading,
    preprocessing, and ground truth generation.
    """
    
    def __init__(
        self, 
        dataset_path: str = "/home/izzat/PycharmProjects/pneumonia_ai/data_tests/chest_xray",
        cache_predictions: bool = True,
        max_workers: int = 4
    ):
        self.dataset_path = Path(dataset_path)
        self.cache_predictions = cache_predictions
        self.max_workers = max_workers
        self.prediction_cache = {}
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Dataset statistics
        self.dataset_stats = {}
        
        logger.info(f"Dataset processor initialized for: {self.dataset_path}")
    
    def load_test_dataset(
        self, 
        subset: str = "test"
    ) -> Tuple[List[Path], np.ndarray, Dict[str, Any]]:
        """
        Load test dataset with ground truth labels.
        
        Args:
            subset: Dataset subset to load ('test', 'train', 'val')
            
        Returns:
            Tuple of (image_paths, labels, dataset_info)
        """
        logger.info(f"Loading {subset} dataset...")
        
        subset_path = self.dataset_path / subset
        if not subset_path.exists():
            raise FileNotFoundError(f"Dataset subset not found: {subset_path}")
        
        # Get normal and pneumonia directories
        normal_dir = subset_path / "NORMAL"
        pneumonia_dir = subset_path / "PNEUMONIA"
        
        if not normal_dir.exists() or not pneumonia_dir.exists():
            raise FileNotFoundError(f"Required directories not found in {subset_path}")
        
        # Load image paths and labels
        image_paths = []
        labels = []
        
        # Load normal images (label = 0)
        normal_images = self._get_valid_images(normal_dir)
        image_paths.extend(normal_images)
        labels.extend([0] * len(normal_images))
        
        # Load pneumonia images (label = 1)
        pneumonia_images = self._get_valid_images(pneumonia_dir)
        image_paths.extend(pneumonia_images)
        labels.extend([1] * len(pneumonia_images))
        
        labels = np.array(labels)
        
        # Generate dataset statistics
        dataset_info = {
            'subset': subset,
            'total_images': len(image_paths),
            'normal_images': len(normal_images),
            'pneumonia_images': len(pneumonia_images),
            'prevalence': len(pneumonia_images) / len(image_paths),
            'normal_dir': str(normal_dir),
            'pneumonia_dir': str(pneumonia_dir),
            'supported_formats': list(self.supported_formats),
            'load_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Loaded {len(image_paths)} images: "
                   f"{len(normal_images)} normal, {len(pneumonia_images)} pneumonia")
        logger.info(f"Prevalence: {dataset_info['prevalence']:.3f}")
        
        self.dataset_stats[subset] = dataset_info
        
        return image_paths, labels, dataset_info
    
    def _get_valid_images(self, directory: Path) -> List[Path]:
        """Get valid image files from directory."""
        valid_images = []
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                # Verify image can be opened
                try:
                    with Image.open(file_path) as img:
                        # Basic validation - ensure image can be loaded
                        img.verify()
                    valid_images.append(file_path)
                except Exception as e:
                    logger.warning(f"Skipping invalid image {file_path}: {e}")
        
        return sorted(valid_images)
    
    def generate_predictions_batch(
        self, 
        image_paths: List[Path],
        predictor,
        model_name: str = "torchxrayvision",
        batch_size: int = 10,
        progress_bar: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Generate predictions for a batch of images using the specified model.
        
        Args:
            image_paths: List of image file paths
            predictor: Model predictor instance
            model_name: Name of the model to use
            batch_size: Number of images to process concurrently
            progress_bar: Whether to show progress bar
            
        Returns:
            Tuple of (prediction_scores, detailed_results)
        """
        logger.info(f"Generating predictions for {len(image_paths)} images using {model_name}")
        
        # Check cache first
        cache_key = self._generate_cache_key(image_paths, model_name)
        if self.cache_predictions and cache_key in self.prediction_cache:
            logger.info("Using cached predictions")
            cached_result = self.prediction_cache[cache_key]
            return cached_result['scores'], cached_result['details']
        
        prediction_scores = []
        detailed_results = []
        failed_predictions = []
        
        # Process images in batches
        iterator = range(0, len(image_paths), batch_size)
        if progress_bar:
            iterator = tqdm(iterator, desc=f"Processing batches ({model_name})")
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            
            # Process batch with threading
            batch_scores, batch_details, batch_failures = self._process_batch(
                batch_paths, predictor, model_name
            )
            
            prediction_scores.extend(batch_scores)
            detailed_results.extend(batch_details)
            failed_predictions.extend(batch_failures)
        
        # Convert to numpy array
        prediction_scores = np.array(prediction_scores)
        
        # Log results
        success_rate = len(prediction_scores) / len(image_paths)
        logger.info(f"Prediction success rate: {success_rate:.2%} "
                   f"({len(prediction_scores)}/{len(image_paths)})")
        
        if failed_predictions:
            logger.warning(f"{len(failed_predictions)} predictions failed")
            for failure in failed_predictions[:5]:  # Log first 5 failures
                logger.warning(f"Failed: {failure['path']} - {failure['error']}")
        
        # Cache results
        if self.cache_predictions:
            self.prediction_cache[cache_key] = {
                'scores': prediction_scores,
                'details': detailed_results,
                'model_name': model_name,
                'timestamp': datetime.now().isoformat()
            }
        
        return prediction_scores, detailed_results
    
    def _process_batch(
        self, 
        batch_paths: List[Path],
        predictor,
        model_name: str
    ) -> Tuple[List[float], List[Dict[str, Any]], List[Dict[str, str]]]:
        """Process a batch of images with threading."""
        
        batch_scores = []
        batch_details = []
        batch_failures = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit prediction tasks
            future_to_path = {
                executor.submit(self._predict_single_image, path, predictor, model_name): path
                for path in batch_paths
            }
            
            # Collect results
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per image
                    if result['success']:
                        batch_scores.append(result['pneumonia_score'])
                        batch_details.append(result['details'])
                    else:
                        batch_failures.append({
                            'path': str(path),
                            'error': result['error']
                        })
                except Exception as e:
                    batch_failures.append({
                        'path': str(path),
                        'error': f"Processing timeout or error: {str(e)}"
                    })
        
        return batch_scores, batch_details, batch_failures
    
    def _predict_single_image(
        self, 
        image_path: Path,
        predictor,
        model_name: str
    ) -> Dict[str, Any]:
        """Predict on a single image."""
        
        try:
            # Make prediction
            result = predictor.predict_from_file(
                image_file=str(image_path),
                model_name=model_name,
                enhance_confidence=False,  # Use base model performance
                use_ensemble=False,
                generate_visualization=False
            )
            
            # Extract pneumonia probability
            if 'class_probabilities' in result:
                pneumonia_score = result['class_probabilities'].get('pneumonia', 0.0)
            else:
                # Fallback for different result formats
                pneumonia_score = result.get('confidence_score', 0.0)
                if result.get('prediction_class') == 'normal':
                    pneumonia_score = 1.0 - pneumonia_score
            
            return {
                'success': True,
                'pneumonia_score': float(pneumonia_score),
                'details': {
                    'image_path': str(image_path),
                    'predicted_class': result.get('prediction_class', 'unknown'),
                    'confidence_score': result.get('confidence_score', 0.0),
                    'class_probabilities': result.get('class_probabilities', {}),
                    'model_version': result.get('model_version', 'unknown'),
                    'processing_time': result.get('processing_time', 0.0),
                    'image_quality': result.get('medical_quality_assessment', {})
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for {image_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'image_path': str(image_path)
            }
    
    def _generate_cache_key(self, image_paths: List[Path], model_name: str) -> str:
        """Generate cache key for prediction results."""
        
        # Create hash of image paths and model name
        path_string = "|".join(str(p) for p in sorted(image_paths))
        combined_string = f"{model_name}:{path_string}"
        
        return hashlib.md5(combined_string.encode()).hexdigest()
    
    def save_predictions(
        self, 
        image_paths: List[Path],
        labels: np.ndarray,
        prediction_scores: np.ndarray,
        detailed_results: List[Dict[str, Any]],
        model_name: str,
        output_file: str
    ):
        """Save predictions and ground truth to file."""
        
        # Create comprehensive results dataframe
        results_data = []
        
        for i, (path, true_label, pred_score) in enumerate(zip(image_paths, labels, prediction_scores)):
            row = {
                'image_path': str(path),
                'image_name': path.name,
                'true_label': int(true_label),
                'true_class': 'pneumonia' if true_label == 1 else 'normal',
                'predicted_score': float(pred_score),
                'model_name': model_name,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            # Add detailed results if available
            if i < len(detailed_results):
                details = detailed_results[i]
                row.update({
                    'predicted_class': details.get('predicted_class', 'unknown'),
                    'confidence_score': details.get('confidence_score', 0.0),
                    'processing_time': details.get('processing_time', 0.0),
                    'model_version': details.get('model_version', 'unknown')
                })
                
                # Add image quality metrics if available
                quality = details.get('image_quality', {})
                if quality:
                    row.update({
                        'image_quality_overall': quality.get('overall_medical_quality', 0.0),
                        'image_quality_rating': quality.get('medical_quality_rating', 'unknown'),
                        'contrast_score': quality.get('contrast_score', 0.0),
                        'sharpness_score': quality.get('sharpness_score', 0.0)
                    })
            
            results_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(results_data)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        # Also save as JSON for detailed analysis
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        logger.info(f"Detailed results saved to {json_path}")
        
        return df
    
    def validate_dataset_integrity(self, subset: str = "test") -> Dict[str, Any]:
        """Validate dataset integrity and report any issues."""
        
        logger.info(f"Validating {subset} dataset integrity...")
        
        validation_report = {
            'subset': subset,
            'validation_timestamp': datetime.now().isoformat(),
            'issues': [],
            'statistics': {},
            'recommendations': []
        }
        
        try:
            image_paths, labels, dataset_info = self.load_test_dataset(subset)
            
            # Basic statistics
            validation_report['statistics'] = dataset_info
            
            # Check for balanced dataset
            prevalence = dataset_info['prevalence']
            if prevalence < 0.3 or prevalence > 0.7:
                validation_report['issues'].append({
                    'type': 'class_imbalance',
                    'description': f"Dataset imbalance detected (prevalence: {prevalence:.3f})",
                    'severity': 'medium'
                })
            
            # Check minimum sample sizes
            min_samples_per_class = 100
            if dataset_info['normal_images'] < min_samples_per_class:
                validation_report['issues'].append({
                    'type': 'insufficient_normal_samples',
                    'description': f"Insufficient normal samples ({dataset_info['normal_images']} < {min_samples_per_class})",
                    'severity': 'high'
                })
            
            if dataset_info['pneumonia_images'] < min_samples_per_class:
                validation_report['issues'].append({
                    'type': 'insufficient_pneumonia_samples',
                    'description': f"Insufficient pneumonia samples ({dataset_info['pneumonia_images']} < {min_samples_per_class})",
                    'severity': 'high'
                })
            
            # Sample image quality check
            sample_size = min(50, len(image_paths))
            sample_indices = np.random.choice(len(image_paths), sample_size, replace=False)
            sample_paths = [image_paths[i] for i in sample_indices]
            
            quality_issues = 0
            for path in sample_paths:
                try:
                    with Image.open(path) as img:
                        width, height = img.size
                        if width < 224 or height < 224:
                            quality_issues += 1
                except Exception:
                    quality_issues += 1
            
            if quality_issues > sample_size * 0.1:  # > 10% issues
                validation_report['issues'].append({
                    'type': 'image_quality',
                    'description': f"Image quality issues detected in {quality_issues}/{sample_size} samples",
                    'severity': 'medium'
                })
            
            # Generate recommendations
            if not validation_report['issues']:
                validation_report['recommendations'].append("Dataset appears healthy for validation")
            else:
                validation_report['recommendations'].append("Address identified issues before validation")
                
                if prevalence < 0.3:
                    validation_report['recommendations'].append("Consider acquiring more pneumonia cases for balance")
                elif prevalence > 0.7:
                    validation_report['recommendations'].append("Consider acquiring more normal cases for balance")
            
            validation_report['overall_status'] = 'healthy' if not validation_report['issues'] else 'issues_detected'
            
        except Exception as e:
            validation_report['overall_status'] = 'failed'
            validation_report['error'] = str(e)
            logger.error(f"Dataset validation failed: {e}")
        
        logger.info(f"Dataset validation completed: {validation_report['overall_status']}")
        return validation_report
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get comprehensive dataset summary."""
        
        summary = {
            'dataset_path': str(self.dataset_path),
            'supported_formats': list(self.supported_formats),
            'loaded_subsets': list(self.dataset_stats.keys()),
            'cache_enabled': self.cache_predictions,
            'cached_predictions': len(self.prediction_cache),
            'summary_timestamp': datetime.now().isoformat()
        }
        
        # Add statistics for each loaded subset
        for subset, stats in self.dataset_stats.items():
            summary[f'{subset}_statistics'] = stats
        
        return summary

# Global dataset processor instance
dataset_processor = MedicalDatasetProcessor()
"""
Validation Manager for Explainable AI System
Coordinates comprehensive validation testing for clinical deployment

Literature-based validation framework implementing:
- Faithfulness validation (insertion/deletion tests)
- Sensitivity validation (stability under perturbations)
- Clinical readiness assessment
- Comprehensive reporting for medical deployment
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
import time

from .explanation_validation import (
    ExplanationFaithfulnessValidator,
    ExplanationSensitivityValidator,
    ClinicalValidationProtocol
)

logger = logging.getLogger(__name__)

class ExplanationValidationManager:
    """
    Main manager for comprehensive explanation validation
    
    Coordinates faithfulness, sensitivity, and clinical validation
    following medical literature protocols
    """
    
    def __init__(self, streamlined_manager):
        self.streamlined_manager = streamlined_manager
        self.device = streamlined_manager.device
        
        # Initialize validation components
        self.faithfulness_validator = None
        self.sensitivity_validator = None
        self.clinical_protocol = ClinicalValidationProtocol()
        
        # Validation configuration
        self.validation_config = {
            'run_faithfulness': True,
            'run_sensitivity': True,
            'run_clinical_assessment': True,
            'save_detailed_results': True
        }
        
        logger.info("ExplanationValidationManager initialized")
    
    def validate_explanation_system(self,
                                  test_image_path: str,
                                  model_name: str = "huggingface_vit",
                                  explanation_methods: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive validation of explanation system
        
        Runs complete validation pipeline:
        1. Faithfulness validation (insertion/deletion tests)
        2. Sensitivity validation (stability tests)  
        3. Clinical readiness assessment
        4. Literature compliance verification
        
        Args:
            test_image_path: Path to test image
            model_name: Model to validate
            explanation_methods: Methods to validate (default: ['gradcam'])
            
        Returns:
            Complete validation report
        """
        
        if explanation_methods is None:
            explanation_methods = ['gradcam']
        
        validation_start_time = time.time()
        
        print(f"🔬 COMPREHENSIVE EXPLANATION VALIDATION")
        print("=" * 60)
        print(f"Image: {Path(test_image_path).name}")
        print(f"Model: {model_name}")
        print(f"Methods: {', '.join(explanation_methods)}")
        print("=" * 60)
        
        try:
            # Initialize validators
            self._initialize_validators(model_name)
            
            # Generate baseline explanation
            print("\n1️⃣ Generating baseline explanation...")
            baseline_explanation = self.streamlined_manager.generate_enhanced_explanation(
                test_image_path, model_name
            )
            
            if 'error' in baseline_explanation:
                return {
                    'error': f"Baseline explanation failed: {baseline_explanation['error']}",
                    'timestamp': datetime.now().isoformat()
                }
            
            baseline_score = baseline_explanation.get('overall_quality', {}).get('overall_interpretability_score', 0)
            print(f"   Baseline interpretability: {baseline_score:.3f}")
            
            validation_results = {
                'baseline_explanation': baseline_explanation,
                'validation_methods': explanation_methods,
                'validation_config': self.validation_config,
                'validation_start_time': datetime.now().isoformat(),
                'method_results': {}
            }
            
            # Validate each explanation method
            for method in explanation_methods:
                print(f"\n🔬 Validating {method.upper()} method...")
                method_results = self._validate_single_method(
                    test_image_path, model_name, method, baseline_explanation
                )
                validation_results['method_results'][method] = method_results
            
            # Clinical readiness assessment
            if self.validation_config['run_clinical_assessment']:
                print(f"\n🏥 Clinical readiness assessment...")
                clinical_results = self._assess_clinical_readiness(validation_results)
                validation_results['clinical_assessment'] = clinical_results
            
            # Generate comprehensive report
            validation_results['summary'] = self._generate_validation_summary(validation_results)
            validation_results['total_validation_time'] = time.time() - validation_start_time
            validation_results['timestamp'] = datetime.now().isoformat()
            
            # Save results if configured
            if self.validation_config['save_detailed_results']:
                self._save_validation_results(validation_results, test_image_path)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation system failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _initialize_validators(self, model_name: str):
        """Initialize validation components"""
        
        # Get model and preprocessing function
        model = self.streamlined_manager.get_model(model_name)
        
        def preprocess_func(image):
            return self.streamlined_manager.preprocess_image(image, model_name)
        
        # Initialize faithfulness validator
        if self.validation_config['run_faithfulness']:
            self.faithfulness_validator = ExplanationFaithfulnessValidator(
                model, self.device, preprocess_func
            )
        
        # Initialize sensitivity validator
        if self.validation_config['run_sensitivity']:
            def explanation_generator(image_path, model_name):
                return self.streamlined_manager.generate_enhanced_explanation(image_path, model_name)
            
            self.sensitivity_validator = ExplanationSensitivityValidator(explanation_generator)
    
    def _validate_single_method(self,
                               test_image_path: str,
                               model_name: str,
                               method: str,
                               baseline_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single explanation method"""
        
        method_results = {
            'method_name': method,
            'validation_start_time': datetime.now().isoformat()
        }
        
        try:
            # Extract method-specific explanation
            explanations_data = baseline_explanation.get('explanations', {})
            
            if method == 'gradcam' and 'gradcam' in explanations_data:
                gradcam_data = explanations_data['gradcam']
                heatmaps = gradcam_data.get('heatmaps', {})
                
                if heatmaps:
                    # Use first available heatmap for validation
                    heatmap_name = list(heatmaps.keys())[0]
                    explanation_map = heatmaps[heatmap_name]
                    target_class = gradcam_data.get('target_class', 1)
                    
                    # Load original image
                    original_image = self.streamlined_manager.load_image(test_image_path)
                    original_array = np.array(original_image.convert('L'))  # Grayscale
                    
                    # Faithfulness validation
                    if self.validation_config['run_faithfulness'] and self.faithfulness_validator:
                        print(f"   🎯 Testing faithfulness (insertion/deletion)...")
                        faithfulness_results = self.faithfulness_validator.validate_faithfulness(
                            original_array, explanation_map, target_class, method
                        )
                        method_results['faithfulness'] = faithfulness_results
                        
                        if 'error' not in faithfulness_results:
                            print(f"      Faithfulness: {faithfulness_results['faithfulness_score']:.3f}")
                            print(f"      Grade: {faithfulness_results['faithfulness_grade']}")
                    
                    # Sensitivity validation
                    if self.validation_config['run_sensitivity'] and self.sensitivity_validator:
                        print(f"   🔄 Testing sensitivity (stability)...")
                        sensitivity_results = self.sensitivity_validator.validate_sensitivity(
                            test_image_path, model_name, method
                        )
                        method_results['sensitivity'] = sensitivity_results
                        
                        if 'error' not in sensitivity_results:
                            print(f"      Sensitivity: {sensitivity_results['sensitivity_score']:.3f}")
                            print(f"      Grade: {sensitivity_results['sensitivity_grade']}")
                else:
                    method_results['error'] = f"No {method} heatmaps found in explanation"
            
            elif method == 'lime' and 'lime' in explanations_data:
                # LIME validation (simplified for now)
                lime_data = explanations_data['lime']
                
                if 'error' not in lime_data:
                    # Sensitivity validation for LIME
                    if self.validation_config['run_sensitivity'] and self.sensitivity_validator:
                        print(f"   🔄 Testing LIME sensitivity...")
                        sensitivity_results = self.sensitivity_validator.validate_sensitivity(
                            test_image_path, model_name, method
                        )
                        method_results['sensitivity'] = sensitivity_results
                        
                        if 'error' not in sensitivity_results:
                            print(f"      Sensitivity: {sensitivity_results['sensitivity_score']:.3f}")
                else:
                    method_results['error'] = f"LIME explanation failed: {lime_data['error']}"
            
            else:
                method_results['error'] = f"Method {method} not found in baseline explanation"
            
            method_results['validation_end_time'] = datetime.now().isoformat()
            return method_results
            
        except Exception as e:
            logger.error(f"Method validation failed for {method}: {e}")
            method_results['error'] = str(e)
            return method_results
    
    def _assess_clinical_readiness(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall clinical readiness"""
        
        try:
            # Get primary method results (usually gradcam)
            method_results = validation_results.get('method_results', {})
            primary_method = 'gradcam' if 'gradcam' in method_results else list(method_results.keys())[0] if method_results else None
            
            if not primary_method:
                return {'error': 'No method results available for clinical assessment'}
            
            primary_results = method_results[primary_method]
            baseline_explanation = validation_results['baseline_explanation']
            
            # Extract validation components
            faithfulness_results = primary_results.get('faithfulness', {})
            sensitivity_results = primary_results.get('sensitivity', {})
            
            # Run clinical validation protocol
            clinical_assessment = self.clinical_protocol.validate_clinical_readiness(
                faithfulness_results, sensitivity_results, baseline_explanation
            )
            
            if 'error' not in clinical_assessment:
                print(f"   Clinical Ready: {'✅' if clinical_assessment['clinical_ready'] else '❌'}")
                print(f"   Overall Score: {clinical_assessment['overall_score']:.3f}")
                print(f"   Grade: {clinical_assessment['deployment_grade']}")
                print(f"   Recommendation: {clinical_assessment['recommendation']}")
            
            return clinical_assessment
            
        except Exception as e:
            logger.error(f"Clinical assessment failed: {e}")
            return {'error': str(e)}
    
    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation summary"""
        
        try:
            summary = {
                'validation_complete': True,
                'methods_validated': list(validation_results.get('method_results', {}).keys()),
                'validation_metrics': {},
                'literature_compliance': {},
                'clinical_readiness': {},
                'recommendations': []
            }
            
            # Collect metrics from all methods
            for method, results in validation_results.get('method_results', {}).items():
                summary['validation_metrics'][method] = {}
                
                if 'faithfulness' in results and 'error' not in results['faithfulness']:
                    faithfulness = results['faithfulness']
                    summary['validation_metrics'][method]['faithfulness'] = faithfulness['faithfulness_score']
                    summary['literature_compliance'][f'{method}_faithfulness'] = faithfulness.get('literature_compliance', False)
                
                if 'sensitivity' in results and 'error' not in results['sensitivity']:
                    sensitivity = results['sensitivity']
                    summary['validation_metrics'][method]['sensitivity'] = sensitivity['sensitivity_score']
                    summary['literature_compliance'][f'{method}_sensitivity'] = sensitivity.get('literature_compliance', False)
            
            # Clinical assessment summary
            clinical_assessment = validation_results.get('clinical_assessment', {})
            if 'error' not in clinical_assessment:
                summary['clinical_readiness'] = {
                    'ready': clinical_assessment.get('clinical_ready', False),
                    'overall_score': clinical_assessment.get('overall_score', 0),
                    'grade': clinical_assessment.get('deployment_grade', 'Unknown'),
                    'recommendation': clinical_assessment.get('recommendation', 'No recommendation')
                }
                
                # Generate specific recommendations
                if clinical_assessment.get('clinical_ready', False):
                    summary['recommendations'].append("System validated for clinical deployment")
                else:
                    summary['recommendations'].append("System requires improvements before clinical deployment")
                    
                    # Add specific improvement areas
                    criteria = clinical_assessment.get('clinical_criteria', {})
                    if not criteria.get('faithfulness_adequate', True):
                        summary['recommendations'].append("Improve explanation faithfulness (target ≥70%)")
                    if not criteria.get('sensitivity_adequate', True):
                        summary['recommendations'].append("Improve explanation stability (target ≥80%)")
                    if not criteria.get('interpretability_adequate', True):
                        summary['recommendations'].append("Improve explanation interpretability (target ≥85%)")
            
            # Overall literature compliance
            compliance_values = list(summary['literature_compliance'].values())
            summary['overall_literature_compliance'] = all(compliance_values) if compliance_values else False
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {'error': str(e)}
    
    def _save_validation_results(self, results: Dict[str, Any], test_image_path: str):
        """Save detailed validation results"""
        
        try:
            # Create results directory
            results_dir = Path("validation_results")
            results_dir.mkdir(exist_ok=True)
            
            # Generate filename
            image_name = Path(test_image_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_{image_name}_{timestamp}.json"
            
            # Save results
            output_path = results_dir / filename
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Validation results saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")
    
    def get_validation_capabilities(self) -> Dict[str, Any]:
        """Get information about validation capabilities"""
        
        return {
            'available_validators': {
                'faithfulness': {
                    'description': 'Insertion/deletion tests to measure explanation faithfulness',
                    'literature_reference': 'Towards Evaluating Explanations of Vision Transformers for Medical Imaging',
                    'metrics': ['insertion_auc', 'deletion_auc', 'faithfulness_score'],
                    'threshold': 'Minimum 70% for clinical use'
                },
                'sensitivity': {
                    'description': 'Stability tests under input perturbations',
                    'literature_reference': 'Evaluating the Faithfulness and Stability of Explainable AI Methods',
                    'metrics': ['stability_scores', 'sensitivity_score'],
                    'threshold': 'Minimum 80% for clinical use'
                },
                'clinical_protocol': {
                    'description': 'Comprehensive clinical readiness assessment',
                    'literature_reference': 'Quantifying Explanation Quality in Deep Learning Models for Medical Image Analysis',
                    'metrics': ['overall_score', 'clinical_criteria', 'deployment_grade'],
                    'threshold': 'Minimum 80% overall score for deployment'
                }
            },
            'supported_methods': ['gradcam', 'lime'],
            'validation_config': self.validation_config,
            'literature_compliance': True
        }
"""
Comprehensive Test Suite for Medical Validation Framework
Tests all components of the scientific validation system for pneumonia detection.
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json
import warnings
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import validation framework components
try:
    from ai_engine.medical_validator import medical_validator, MedicalValidator
    from ai_engine.threshold_optimizer import threshold_optimizer, MedicalThresholdOptimizer
    from ai_engine.dataset_processor import dataset_processor, MedicalDatasetProcessor
    from ai_engine.performance_analytics import performance_analytics, MedicalPerformanceAnalytics
    from ai_engine.medical_model_loader import MedicalModelManager
    VALIDATION_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import validation framework: {e}")
    VALIDATION_FRAMEWORK_AVAILABLE = False

# Skip tests if framework not available
skip_if_no_framework = unittest.skipIf(
    not VALIDATION_FRAMEWORK_AVAILABLE, 
    "Validation framework not available"
)

class TestMedicalValidator(unittest.TestCase):
    """Test the Medical Validator component."""
    
    @skip_if_no_framework
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = MedicalValidator(save_plots=False)
        
        # Create synthetic test data
        np.random.seed(42)
        self.n_samples = 200
        
        # Create realistic score distributions
        # Normal cases: lower scores, some overlap
        normal_scores = np.random.beta(2, 5, self.n_samples // 2)  # Skewed toward lower values
        
        # Pneumonia cases: higher scores, some overlap
        pneumonia_scores = np.random.beta(5, 2, self.n_samples // 2)  # Skewed toward higher values
        
        self.y_true = np.concatenate([
            np.zeros(self.n_samples // 2),      # Normal cases
            np.ones(self.n_samples // 2)        # Pneumonia cases
        ])
        
        self.y_scores = np.concatenate([normal_scores, pneumonia_scores])
        
        # Shuffle to avoid any ordering bias
        shuffle_indices = np.random.permutation(self.n_samples)
        self.y_true = self.y_true[shuffle_indices]
        self.y_scores = self.y_scores[shuffle_indices]
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @skip_if_no_framework
    def test_validate_model_performance(self):
        """Test comprehensive model validation."""
        
        results = self.validator.validate_model_performance(
            self.y_true, self.y_scores, "test_model", generate_plots=False
        )
        
        # Check required result structure
        self.assertIn('model_name', results)
        self.assertIn('roc_analysis', results)
        self.assertIn('optimal_thresholds', results)
        self.assertIn('medical_metrics', results)
        self.assertIn('recommendations', results)
        
        # Check ROC analysis
        roc_analysis = results['roc_analysis']
        self.assertIn('auc', roc_analysis)
        self.assertGreater(roc_analysis['auc'], 0.5)  # Should be better than random
        self.assertLessEqual(roc_analysis['auc'], 1.0)
        
        # Check optimal thresholds
        optimal_thresholds = results['optimal_thresholds']
        self.assertIn('youden_j', optimal_thresholds)
        
        youden_result = optimal_thresholds['youden_j']
        self.assertIn('threshold', youden_result)
        self.assertIn('sensitivity', youden_result)
        self.assertIn('specificity', youden_result)
        
        # Validate threshold range
        threshold = youden_result['threshold']
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        
        # Validate sensitivity and specificity
        sensitivity = youden_result['sensitivity']
        specificity = youden_result['specificity']
        self.assertGreaterEqual(sensitivity, 0.0)
        self.assertLessEqual(sensitivity, 1.0)
        self.assertGreaterEqual(specificity, 0.0)
        self.assertLessEqual(specificity, 1.0)
    
    @skip_if_no_framework
    def test_medical_metrics_calculation(self):
        """Test medical metrics calculation."""
        
        results = self.validator.validate_model_performance(
            self.y_true, self.y_scores, "test_model", generate_plots=False
        )
        
        medical_metrics = results['medical_metrics']
        
        # Check all required medical metrics
        required_metrics = [
            'sensitivity_recall', 'specificity', 'precision_ppv',
            'negative_predictive_value', 'accuracy', 'f1_score',
            'youden_j_statistic', 'likelihood_ratio_positive'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, medical_metrics)
            self.assertGreaterEqual(medical_metrics[metric], 0.0)
            if metric not in ['likelihood_ratio_positive', 'likelihood_ratio_negative']:
                self.assertLessEqual(medical_metrics[metric], 1.0)
    
    @skip_if_no_framework
    def test_edge_cases(self):
        """Test validation with edge cases."""
        
        # Test with perfect separation
        y_true_perfect = np.array([0, 0, 0, 1, 1, 1])
        y_scores_perfect = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        
        results = self.validator.validate_model_performance(
            y_true_perfect, y_scores_perfect, "perfect_model", generate_plots=False
        )
        
        self.assertEqual(results['roc_analysis']['auc'], 1.0)
        
        # Test with random performance
        y_true_random = np.array([0, 1, 0, 1, 0, 1])
        y_scores_random = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        results = self.validator.validate_model_performance(
            y_true_random, y_scores_random, "random_model", generate_plots=False
        )
        
        # Random performance should have AUC around 0.5
        self.assertLess(abs(results['roc_analysis']['auc'] - 0.5), 0.3)

class TestThresholdOptimizer(unittest.TestCase):
    """Test the Threshold Optimizer component."""
    
    @skip_if_no_framework
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.optimizer = MedicalThresholdOptimizer(output_dir=self.temp_dir)
        
        # Create test data with clear separation
        np.random.seed(42)
        n_samples = 300
        
        # Generate realistic medical data
        normal_scores = np.random.beta(2, 6, n_samples // 2)  # Lower scores for normal
        pneumonia_scores = np.random.beta(6, 2, n_samples // 2)  # Higher scores for pneumonia
        
        self.y_true = np.concatenate([
            np.zeros(n_samples // 2),
            np.ones(n_samples // 2)
        ])
        
        self.y_scores = np.concatenate([normal_scores, pneumonia_scores])
        
        # Shuffle data
        shuffle_indices = np.random.permutation(n_samples)
        self.y_true = self.y_true[shuffle_indices]
        self.y_scores = self.y_scores[shuffle_indices]
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @skip_if_no_framework
    def test_optimize_thresholds(self):
        """Test comprehensive threshold optimization."""
        
        results = self.optimizer.optimize_thresholds(
            self.y_true, self.y_scores, "test_model"
        )
        
        # Check result structure
        self.assertIn('optimal_thresholds', results)
        self.assertIn('threshold_comparison', results)
        self.assertIn('recommendations', results)
        
        # Check Youden's J optimization
        optimal_thresholds = results['optimal_thresholds']
        self.assertIn('youden_j', optimal_thresholds)
        
        youden_result = optimal_thresholds['youden_j']
        self.assertIn('threshold', youden_result)
        self.assertIn('sensitivity', youden_result)
        self.assertIn('specificity', youden_result)
        self.assertIn('youden_j', youden_result)
        
        # Validate Youden's J calculation
        sensitivity = youden_result['sensitivity']
        specificity = youden_result['specificity']
        expected_j = sensitivity + specificity - 1
        actual_j = youden_result['youden_j']
        self.assertAlmostEqual(expected_j, actual_j, places=3)
    
    @skip_if_no_framework
    def test_cost_sensitive_optimization(self):
        """Test cost-sensitive threshold optimization."""
        
        results = self.optimizer.optimize_thresholds(
            self.y_true, self.y_scores, "test_model"
        )
        
        optimal_thresholds = results['optimal_thresholds']
        
        # Check cost-sensitive scenarios
        cost_scenarios = ['cost_screening', 'cost_diagnostic', 'cost_confirmatory']
        
        for scenario in cost_scenarios:
            if scenario in optimal_thresholds:
                scenario_result = optimal_thresholds[scenario]
                self.assertIn('threshold', scenario_result)
                self.assertIn('sensitivity', scenario_result)
                self.assertIn('specificity', scenario_result)
                self.assertIn('expected_cost', scenario_result)
    
    @skip_if_no_framework
    def test_clinical_targets_optimization(self):
        """Test clinical targets optimization."""
        
        results = self.optimizer.optimize_thresholds(
            self.y_true, self.y_scores, "test_model"
        )
        
        optimal_thresholds = results['optimal_thresholds']
        
        # Check clinical targets
        clinical_targets = ['clinical_high_sensitivity', 'clinical_high_specificity', 'clinical_balanced']
        
        for target in clinical_targets:
            if target in optimal_thresholds:
                target_result = optimal_thresholds[target]
                self.assertIn('threshold', target_result)
                self.assertIn('meets_targets', target_result)
                
                # Sensitivity and specificity should be reasonable
                if target_result['meets_targets']:
                    if 'high_sensitivity' in target:
                        self.assertGreaterEqual(target_result['sensitivity'], 0.90)
                    elif 'high_specificity' in target:
                        self.assertGreaterEqual(target_result['specificity'], 0.90)
    
    @skip_if_no_framework
    def test_threshold_comparison(self):
        """Test threshold comparison functionality."""
        
        results = self.optimizer.optimize_thresholds(
            self.y_true, self.y_scores, "test_model"
        )
        
        comparison = results['threshold_comparison']
        self.assertIn('comparison_table', comparison)
        self.assertIn('best_by_metric', comparison)
        
        # Check comparison table structure
        comparison_table = comparison['comparison_table']
        self.assertIsInstance(comparison_table, list)
        self.assertGreater(len(comparison_table), 0)
        
        # Check required columns in comparison
        first_row = comparison_table[0]
        required_columns = ['method', 'threshold', 'sensitivity', 'specificity', 'accuracy']
        
        for column in required_columns:
            self.assertIn(column, first_row)

class TestDatasetProcessor(unittest.TestCase):
    """Test the Dataset Processor component."""
    
    @skip_if_no_framework
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = MedicalDatasetProcessor(cache_predictions=False)
        
        # Create mock dataset structure
        self.dataset_path = Path(self.temp_dir) / "test_dataset"
        self.test_path = self.dataset_path / "test"
        self.normal_path = self.test_path / "NORMAL"
        self.pneumonia_path = self.test_path / "PNEUMONIA"
        
        # Create directories
        self.normal_path.mkdir(parents=True)
        self.pneumonia_path.mkdir(parents=True)
        
        # Create mock image files
        self._create_mock_images()
    
    def _create_mock_images(self):
        """Create mock image files for testing."""
        
        # Create normal images
        for i in range(5):
            image_file = self.normal_path / f"normal_{i:03d}.jpg"
            image_file.write_text(f"mock_image_content_{i}")
        
        # Create pneumonia images
        for i in range(8):
            image_file = self.pneumonia_path / f"pneumonia_{i:03d}.jpg"
            image_file.write_text(f"mock_image_content_{i}")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @skip_if_no_framework
    def test_load_test_dataset(self):
        """Test dataset loading functionality."""
        
        # Update processor to use our test dataset
        self.processor.dataset_path = self.dataset_path
        
        # Mock the image validation since we're using text files
        with patch.object(self.processor, '_get_valid_images') as mock_get_images:
            mock_get_images.side_effect = lambda directory: list(directory.glob("*.jpg"))
            
            image_paths, labels, dataset_info = self.processor.load_test_dataset("test")
            
            # Check results
            self.assertEqual(len(image_paths), 13)  # 5 normal + 8 pneumonia
            self.assertEqual(len(labels), 13)
            self.assertEqual(dataset_info['normal_images'], 5)
            self.assertEqual(dataset_info['pneumonia_images'], 8)
            self.assertAlmostEqual(dataset_info['prevalence'], 8/13, places=3)
            
            # Check label correctness
            self.assertEqual(np.sum(labels), 8)  # 8 pneumonia cases
            self.assertEqual(np.sum(labels == 0), 5)  # 5 normal cases
    
    @skip_if_no_framework
    def test_dataset_validation(self):
        """Test dataset integrity validation."""
        
        self.processor.dataset_path = self.dataset_path
        
        with patch.object(self.processor, '_get_valid_images') as mock_get_images:
            mock_get_images.side_effect = lambda directory: list(directory.glob("*.jpg"))
            
            validation_report = self.processor.validate_dataset_integrity("test")
            
            # Check validation report structure
            self.assertIn('subset', validation_report)
            self.assertIn('statistics', validation_report)
            self.assertIn('issues', validation_report)
            self.assertIn('overall_status', validation_report)
            
            # Should detect class imbalance (5 vs 8 samples)
            issues = validation_report['issues']
            issue_types = [issue['type'] for issue in issues]
            self.assertIn('insufficient_normal_samples', issue_types)
            self.assertIn('insufficient_pneumonia_samples', issue_types)
    
    @skip_if_no_framework
    def test_prediction_batch_processing(self):
        """Test batch prediction processing with mock predictor."""
        
        # Create mock predictor
        mock_predictor = Mock()
        mock_predictor.predict_from_file.return_value = {
            'class_probabilities': {'pneumonia': 0.7, 'normal': 0.3},
            'prediction_class': 'pneumonia',
            'confidence_score': 0.7,
            'processing_time': 1.0,
            'model_version': 'test_v1.0'
        }
        
        # Create mock image paths
        image_paths = [Path(f"mock_image_{i}.jpg") for i in range(10)]
        
        # Test batch processing
        scores, detailed_results = self.processor.generate_predictions_batch(
            image_paths, mock_predictor, "test_model", batch_size=3, progress_bar=False
        )
        
        # Check results
        self.assertEqual(len(scores), 10)
        self.assertEqual(len(detailed_results), 10)
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))
        
        # Verify predictor was called for each image
        self.assertEqual(mock_predictor.predict_from_file.call_count, 10)

class TestMedicalModelManager(unittest.TestCase):
    """Test the Medical Model Manager integration."""
    
    @skip_if_no_framework
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create manager with mock components
        with patch('ai_engine.medical_model_loader.VALIDATION_FRAMEWORK_AVAILABLE', True):
            self.manager = MedicalModelManager()
            self.manager.threshold_config_path = Path(self.temp_dir) / "test_thresholds.json"
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @skip_if_no_framework
    def test_validated_threshold_loading(self):
        """Test loading and saving of validated thresholds."""
        
        # Create test threshold data
        test_thresholds = {
            'test_model': {
                'threshold': 0.123,
                'method': 'youden_j',
                'sensitivity': 0.85,
                'specificity': 0.90,
                'validation_date': '2024-01-01'
            }
        }
        
        # Save test data
        config = {
            'validated_thresholds': test_thresholds,
            'validation_results': {},
            'last_updated': '2024-01-01'
        }
        
        with open(self.manager.threshold_config_path, 'w') as f:
            json.dump(config, f)
        
        # Load thresholds
        self.manager._load_validated_thresholds()
        
        # Verify loading
        self.assertIn('test_model', self.manager.validated_thresholds)
        loaded_threshold = self.manager.validated_thresholds['test_model']
        self.assertAlmostEqual(loaded_threshold['threshold'], 0.123, places=3)
        self.assertEqual(loaded_threshold['method'], 'youden_j')
    
    @skip_if_no_framework
    def test_get_validated_threshold(self):
        """Test getting validated thresholds with fallbacks."""
        
        # Test with validated threshold
        self.manager.validated_thresholds['test_model'] = {
            'threshold': 0.456,
            'method': 'youden_j',
            'sensitivity': 0.88,
            'specificity': 0.92,
            'validation_date': '2024-01-01'
        }
        
        threshold_data = self.manager.get_validated_threshold('test_model')
        self.assertEqual(threshold_data['method'], 'validated_scientific')
        self.assertAlmostEqual(threshold_data['threshold'], 0.456, places=3)
        
        # Test fallback for unknown model
        fallback_data = self.manager.get_validated_threshold('unknown_model')
        self.assertEqual(fallback_data['method'], 'empirical_fallback')
        self.assertIn('warning', fallback_data)
    
    @skip_if_no_framework
    def test_validation_integration(self):
        """Test integration with validation framework."""
        
        # Mock the validation framework components
        with patch('ai_engine.medical_model_loader.dataset_processor') as mock_dataset, \
             patch('ai_engine.medical_model_loader.medical_validator') as mock_validator, \
             patch('ai_engine.medical_model_loader.threshold_optimizer') as mock_optimizer:
            
            # Setup mocks
            mock_dataset.load_test_dataset.return_value = (
                [Path("test1.jpg"), Path("test2.jpg")],
                np.array([0, 1]),
                {'prevalence': 0.5, 'total_images': 2}
            )
            
            mock_dataset.generate_predictions_batch.return_value = (
                np.array([0.3, 0.8]),
                [{'processing_time': 1.0}, {'processing_time': 1.2}]
            )
            
            mock_validator.validate_model_performance.return_value = {
                'roc_analysis': {'auc': 0.85},
                'recommendations': {'primary_threshold': 0.5}
            }
            
            mock_optimizer.optimize_thresholds.return_value = {
                'optimal_thresholds': {
                    'youden_j': {
                        'threshold': 0.6,
                        'sensitivity': 0.85,
                        'specificity': 0.90,
                        'youden_j': 0.75
                    }
                }
            }
            
            # Test validation
            result = self.manager.validate_model_performance('test_model')
            
            # Verify integration
            self.assertIn('validation_results', result)
            self.assertIn('threshold_optimization', result)
            self.assertIn('test_model', self.manager.validated_thresholds)
            
            threshold_data = self.manager.validated_thresholds['test_model']
            self.assertAlmostEqual(threshold_data['threshold'], 0.6, places=3)

class TestIntegrationSuite(unittest.TestCase):
    """Integration tests for the complete validation framework."""
    
    @skip_if_no_framework
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)
        
        # Create comprehensive synthetic dataset
        n_samples = 500
        
        # Generate realistic medical score distributions
        # Normal cases: beta distribution skewed toward lower values
        normal_scores = np.random.beta(2, 6, n_samples // 2)
        
        # Pneumonia cases: beta distribution skewed toward higher values
        pneumonia_scores = np.random.beta(6, 3, n_samples // 2)
        
        self.y_true = np.concatenate([
            np.zeros(n_samples // 2),
            np.ones(n_samples // 2)
        ])
        
        self.y_scores = np.concatenate([normal_scores, pneumonia_scores])
        
        # Shuffle data
        shuffle_indices = np.random.permutation(n_samples)
        self.y_true = self.y_true[shuffle_indices]
        self.y_scores = self.y_scores[shuffle_indices]
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @skip_if_no_framework
    def test_complete_validation_pipeline(self):
        """Test the complete validation pipeline integration."""
        
        # Step 1: Validate model performance
        validation_results = medical_validator.validate_model_performance(
            self.y_true, self.y_scores, "integration_test_model", generate_plots=False
        )
        
        # Step 2: Optimize thresholds
        optimization_results = threshold_optimizer.optimize_thresholds(
            self.y_true, self.y_scores, "integration_test_model"
        )
        
        # Step 3: Perform performance analytics
        analytics_results = performance_analytics.comprehensive_model_analysis(
            "integration_test_model", self.y_true, self.y_scores
        )
        
        # Verify pipeline consistency
        self.assertAlmostEqual(
            validation_results['roc_analysis']['auc'],
            optimization_results['roc_analysis']['auc'],
            places=3
        )
        
        # Check that all components agree on basic metrics
        youden_threshold_validator = validation_results['optimal_thresholds']['youden_j']['threshold']
        youden_threshold_optimizer = optimization_results['optimal_thresholds']['youden_j']['threshold']
        
        self.assertAlmostEqual(youden_threshold_validator, youden_threshold_optimizer, places=3)
        
        # Verify analytics includes validation results
        self.assertIn('basic_validation', analytics_results)
        self.assertIn('extended_metrics', analytics_results)
        self.assertIn('clinical_interpretation', analytics_results)
    
    @skip_if_no_framework
    def test_threshold_consistency_across_components(self):
        """Test that optimal thresholds are consistent across different components."""
        
        # Get optimal thresholds from validator
        validation_results = medical_validator.validate_model_performance(
            self.y_true, self.y_scores, "consistency_test", generate_plots=False
        )
        
        # Get optimal thresholds from optimizer
        optimization_results = threshold_optimizer.optimize_thresholds(
            self.y_true, self.y_scores, "consistency_test"
        )
        
        # Compare Youden's J thresholds
        validator_threshold = validation_results['optimal_thresholds']['youden_j']['threshold']
        optimizer_threshold = optimization_results['optimal_thresholds']['youden_j']['threshold']
        
        # Should be very close (allowing for small numerical differences)
        self.assertAlmostEqual(validator_threshold, optimizer_threshold, places=2)
        
        # Compare sensitivity and specificity
        validator_sens = validation_results['optimal_thresholds']['youden_j']['sensitivity']
        optimizer_sens = optimization_results['optimal_thresholds']['youden_j']['sensitivity']
        
        validator_spec = validation_results['optimal_thresholds']['youden_j']['specificity']
        optimizer_spec = optimization_results['optimal_thresholds']['youden_j']['specificity']
        
        self.assertAlmostEqual(validator_sens, optimizer_sens, places=2)
        self.assertAlmostEqual(validator_spec, optimizer_spec, places=2)
    
    @skip_if_no_framework
    def test_performance_degradation_detection(self):
        """Test detection of performance degradation."""
        
        # Create high-performance baseline
        baseline_results = medical_validator.validate_model_performance(
            self.y_true, self.y_scores, "baseline_model", generate_plots=False
        )
        
        # Create degraded performance (add noise to scores)
        degraded_scores = self.y_scores + np.random.normal(0, 0.2, len(self.y_scores))
        degraded_scores = np.clip(degraded_scores, 0, 1)  # Keep in valid range
        
        degraded_results = medical_validator.validate_model_performance(
            self.y_true, degraded_scores, "degraded_model", generate_plots=False
        )
        
        # Performance should be degraded
        baseline_auc = baseline_results['roc_analysis']['auc']
        degraded_auc = degraded_results['roc_analysis']['auc']
        
        self.assertLess(degraded_auc, baseline_auc)
        self.assertGreater(abs(baseline_auc - degraded_auc), 0.01)  # Meaningful difference

def run_validation_tests():
    """Run all validation framework tests."""
    
    if not VALIDATION_FRAMEWORK_AVAILABLE:
        print("Validation framework not available - skipping tests")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMedicalValidator,
        TestThresholdOptimizer,
        TestDatasetProcessor,
        TestMedicalModelManager,
        TestIntegrationSuite
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"VALIDATION FRAMEWORK TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    print(f"{'='*60}")
    
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == '__main__':
    success = run_validation_tests()
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test the pneumonia AI system on real chest X-ray dataset
Evaluates performance on actual medical images
"""

import os
import sys
import django
from pathlib import Path
import random
import json
from datetime import datetime

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pneumonia_detection.settings')
django.setup()

import torch
import numpy as np
from PIL import Image
from ai_engine.medical_model_loader import medical_model_manager
from ai_engine.medical_predictor import medical_predictor

class RealDatasetTester:
    """Test the medical AI system on real chest X-ray dataset"""
    
    def __init__(self, dataset_path="/home/izzat/PycharmProjects/pneumonia_ai/data_tests/chest_xray"):
        self.dataset_path = Path(dataset_path)
        self.test_path = self.dataset_path / "test"
        self.normal_path = self.test_path / "NORMAL"
        self.pneumonia_path = self.test_path / "PNEUMONIA"
        
        # Check if paths exist
        if not self.test_path.exists():
            raise FileNotFoundError(f"Test dataset not found at {self.test_path}")
        
        print(f"Dataset found at: {self.dataset_path}")
        print(f"Normal images: {self.normal_path}")
        print(f"Pneumonia images: {self.pneumonia_path}")
    
    def get_sample_images(self, sample_size=20):
        """Get random sample of images for testing"""
        normal_images = []
        pneumonia_images = []
        
        # Get normal images
        if self.normal_path.exists():
            normal_files = list(self.normal_path.glob("*.jpeg"))
            normal_images = random.sample(normal_files, min(sample_size, len(normal_files)))
        
        # Get pneumonia images  
        if self.pneumonia_path.exists():
            pneumonia_files = list(self.pneumonia_path.glob("*.jpeg"))
            pneumonia_images = random.sample(pneumonia_files, min(sample_size, len(pneumonia_files)))
        
        return normal_images, pneumonia_images
    
    def test_single_image(self, image_path, expected_class, model_name="torchxrayvision"):
        """Test prediction on a single image"""
        try:
            result = medical_predictor.predict_from_file(
                str(image_path),
                model_name=model_name,
                enhance_confidence=True,
                use_ensemble=False
            )
            
            prediction = result['prediction_class']
            confidence = result['confidence_score']
            correct = prediction == expected_class
            
            return {
                'image_path': str(image_path),
                'expected': expected_class,
                'predicted': prediction,
                'confidence': confidence,
                'correct': correct,
                'model': model_name,
                'model_version': result.get('model_version', 'unknown'),
                'medical_indicators': result.get('medical_indicators', {}),
                'class_probabilities': result.get('class_probabilities', {})
            }
            
        except Exception as e:
            return {
                'image_path': str(image_path),
                'expected': expected_class,
                'predicted': 'error',
                'confidence': 0.0,
                'correct': False,
                'model': model_name,
                'error': str(e)
            }
    
    def run_comprehensive_test(self, sample_size=20, models_to_test=None):
        """Run comprehensive test on the dataset"""
        
        if models_to_test is None:
            models_to_test = ["torchxrayvision", "efficientnet_b4"]
        
        print("=== TESTING PNEUMONIA AI ON REAL CHEST X-RAY DATASET ===\n")
        print(f"Sample size: {sample_size} images per class")
        print(f"Models to test: {models_to_test}")
        print("-" * 60)
        
        # Get sample images
        normal_images, pneumonia_images = self.get_sample_images(sample_size)
        
        print(f"Found {len(normal_images)} normal images")
        print(f"Found {len(pneumonia_images)} pneumonia images")
        print()
        
        all_results = []
        model_performance = {}
        
        for model_name in models_to_test:
            print(f"\n🤖 TESTING MODEL: {model_name.upper()}")
            print("=" * 50)
            
            model_results = []
            
            # Test normal images
            print(f"\nTesting {len(normal_images)} NORMAL images...")
            for i, image_path in enumerate(normal_images, 1):
                print(f"  [{i:2d}/{len(normal_images)}] {image_path.name}", end="")
                
                result = self.test_single_image(image_path, "normal", model_name)
                model_results.append(result)
                
                status = "✅" if result['correct'] else "❌"
                print(f" -> {result['predicted']} ({result['confidence']:.3f}) {status}")
            
            # Test pneumonia images
            print(f"\nTesting {len(pneumonia_images)} PNEUMONIA images...")
            for i, image_path in enumerate(pneumonia_images, 1):
                print(f"  [{i:2d}/{len(pneumonia_images)}] {image_path.name}", end="")
                
                result = self.test_single_image(image_path, "pneumonia", model_name)
                model_results.append(result)
                
                status = "✅" if result['correct'] else "❌"
                print(f" -> {result['predicted']} ({result['confidence']:.3f}) {status}")
            
            # Calculate performance metrics for this model
            performance = self.calculate_performance_metrics(model_results, model_name)
            model_performance[model_name] = performance
            
            # Print model summary
            self.print_model_summary(performance, model_name)
            
            all_results.extend(model_results)
        
        # Overall comparison
        print("\n" + "=" * 60)
        print("📊 OVERALL PERFORMANCE COMPARISON")
        print("=" * 60)
        
        for model_name, perf in model_performance.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Overall Accuracy: {perf['accuracy']:.1%}")
            print(f"  Normal Accuracy:  {perf['normal_accuracy']:.1%} ({perf['normal_correct']}/{perf['normal_total']})")
            print(f"  Pneumonia Accuracy: {perf['pneumonia_accuracy']:.1%} ({perf['pneumonia_correct']}/{perf['pneumonia_total']})")
            print(f"  Avg Confidence: {perf['avg_confidence']:.3f}")
            print(f"  False Positive Rate: {perf['false_positive_rate']:.1%}")
            print(f"  False Negative Rate: {perf['false_negative_rate']:.1%}")
        
        # Find best model
        best_model = max(model_performance.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model[0].upper()}")
        print(f"   Accuracy: {best_model[1]['accuracy']:.1%}")
        
        # Save detailed results
        self.save_results(all_results, model_performance)
        
        # Medical assessment
        self.medical_assessment(model_performance)
        
        return all_results, model_performance
    
    def calculate_performance_metrics(self, results, model_name):
        """Calculate detailed performance metrics"""
        
        # Filter out errors
        valid_results = [r for r in results if r['predicted'] != 'error']
        
        if not valid_results:
            return {
                'accuracy': 0.0,
                'normal_accuracy': 0.0,
                'pneumonia_accuracy': 0.0,
                'avg_confidence': 0.0,
                'false_positive_rate': 0.0,
                'false_negative_rate': 0.0,
                'total_tests': 0,
                'errors': len(results)
            }
        
        # Overall metrics
        total_correct = sum(1 for r in valid_results if r['correct'])
        total_tests = len(valid_results)
        accuracy = total_correct / total_tests if total_tests > 0 else 0
        
        # Class-specific metrics
        normal_results = [r for r in valid_results if r['expected'] == 'normal']
        pneumonia_results = [r for r in valid_results if r['expected'] == 'pneumonia']
        
        normal_correct = sum(1 for r in normal_results if r['correct'])
        pneumonia_correct = sum(1 for r in pneumonia_results if r['correct'])
        
        normal_accuracy = normal_correct / len(normal_results) if normal_results else 0
        pneumonia_accuracy = pneumonia_correct / len(pneumonia_results) if pneumonia_results else 0
        
        # False positive/negative rates
        false_positives = sum(1 for r in normal_results if r['predicted'] == 'pneumonia')
        false_negatives = sum(1 for r in pneumonia_results if r['predicted'] == 'normal')
        
        false_positive_rate = false_positives / len(normal_results) if normal_results else 0
        false_negative_rate = false_negatives / len(pneumonia_results) if pneumonia_results else 0
        
        # Confidence metrics
        confidences = [r['confidence'] for r in valid_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'model': model_name,
            'accuracy': accuracy,
            'normal_accuracy': normal_accuracy,
            'pneumonia_accuracy': pneumonia_accuracy,
            'normal_correct': normal_correct,
            'normal_total': len(normal_results),
            'pneumonia_correct': pneumonia_correct,
            'pneumonia_total': len(pneumonia_results),
            'avg_confidence': avg_confidence,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_tests': total_tests,
            'total_correct': total_correct,
            'errors': len(results) - len(valid_results)
        }
    
    def print_model_summary(self, performance, model_name):
        """Print performance summary for a model"""
        print(f"\n📈 {model_name.upper()} PERFORMANCE SUMMARY:")
        print("-" * 40)
        print(f"Overall Accuracy: {performance['accuracy']:.1%}")
        print(f"Normal Images: {performance['normal_correct']}/{performance['normal_total']} ({performance['normal_accuracy']:.1%})")
        print(f"Pneumonia Images: {performance['pneumonia_correct']}/{performance['pneumonia_total']} ({performance['pneumonia_accuracy']:.1%})")
        print(f"Average Confidence: {performance['avg_confidence']:.3f}")
        print(f"False Positives: {performance['false_positives']} ({performance['false_positive_rate']:.1%})")
        print(f"False Negatives: {performance['false_negatives']} ({performance['false_negative_rate']:.1%})")
        if performance['errors'] > 0:
            print(f"Errors: {performance['errors']}")
    
    def medical_assessment(self, model_performance):
        """Provide medical assessment of the models"""
        print("\n" + "=" * 60)
        print("🏥 MEDICAL ASSESSMENT")
        print("=" * 60)
        
        print("\nClinical Requirements:")
        print("- Overall Accuracy ≥ 90%: Required for clinical use")
        print("- False Positive Rate ≤ 10%: Avoid unnecessary treatment")
        print("- False Negative Rate ≤ 5%: Critical - missing pneumonia is dangerous")
        print("- Average Confidence ≥ 80%: Reliable decision support")
        
        for model_name, perf in model_performance.items():
            print(f"\n{model_name.upper()} Clinical Assessment:")
            
            # Check clinical requirements
            requirements = [
                (perf['accuracy'] >= 0.90, f"Overall Accuracy: {perf['accuracy']:.1%}"),
                (perf['false_positive_rate'] <= 0.10, f"False Positive Rate: {perf['false_positive_rate']:.1%}"),
                (perf['false_negative_rate'] <= 0.05, f"False Negative Rate: {perf['false_negative_rate']:.1%}"),
                (perf['avg_confidence'] >= 0.80, f"Average Confidence: {perf['avg_confidence']:.1%}")
            ]
            
            passed_requirements = 0
            for passed, description in requirements:
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {status} {description}")
                if passed:
                    passed_requirements += 1
            
            # Overall clinical verdict
            if passed_requirements == 4:
                verdict = "✅ READY FOR CLINICAL USE"
            elif passed_requirements >= 3:
                verdict = "⚠️  NEEDS IMPROVEMENT BEFORE CLINICAL USE"
            else:
                verdict = "❌ NOT SUITABLE FOR CLINICAL USE"
            
            print(f"  {verdict} ({passed_requirements}/4 requirements met)")
    
    def save_results(self, all_results, model_performance):
        """Save detailed results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"real_dataset_test_results_{timestamp}.json"
        
        data = {
            'timestamp': timestamp,
            'dataset_path': str(self.dataset_path),
            'model_performance': model_performance,
            'detailed_results': all_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_file}")

def main():
    """Main testing function"""
    try:
        # Initialize tester
        tester = RealDatasetTester()
        
        # Run comprehensive test
        results, performance = tester.run_comprehensive_test(
            sample_size=30,  # Test 30 images per class
            models_to_test=["torchxrayvision", "efficientnet_b4"]
        )
        
        print("\n🎉 Real dataset testing completed!")
        
    except FileNotFoundError as e:
        print(f"❌ Dataset not found: {e}")
        print("Please ensure the chest X-ray dataset is at: data_tests/chest_xray/")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
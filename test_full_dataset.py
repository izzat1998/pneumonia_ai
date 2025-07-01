#!/usr/bin/env python3
"""
Comprehensive test on the full real chest X-ray dataset
Tests the high-performance ViT model on larger sample sizes
"""

import os
import sys
import django
from pathlib import Path
import random
import json
from datetime import datetime
import time

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pneumonia_detection.settings')
django.setup()

from ai_engine.medical_predictor import medical_predictor

class FullDatasetTester:
    """Test on the complete real chest X-ray dataset"""
    
    def __init__(self, dataset_path="/home/izzat/PycharmProjects/pneumonia_ai/data_tests/chest_xray"):
        self.dataset_path = Path(dataset_path)
        self.test_path = self.dataset_path / "test"
        self.normal_path = self.test_path / "NORMAL"
        self.pneumonia_path = self.test_path / "PNEUMONIA"
        
        print(f"Dataset path: {self.dataset_path}")
        print(f"Test path: {self.test_path}")
        
        # Count available images
        self.normal_count = len(list(self.normal_path.glob("*.jpeg"))) if self.normal_path.exists() else 0
        self.pneumonia_count = len(list(self.pneumonia_path.glob("*.jpeg"))) if self.pneumonia_path.exists() else 0
        
        print(f"Available normal images: {self.normal_count}")
        print(f"Available pneumonia images: {self.pneumonia_count}")
        print(f"Total test images: {self.normal_count + self.pneumonia_count}")
    
    def run_comprehensive_test(self, sample_sizes=[20, 50, 100], models_to_test=None):
        """Run tests with different sample sizes"""
        
        if models_to_test is None:
            models_to_test = ["huggingface_vit"]  # Focus on the high-performance model
        
        print("\n" + "=" * 80)
        print("🏥 COMPREHENSIVE REAL DATASET TESTING")
        print("=" * 80)
        print(f"Models to test: {models_to_test}")
        print(f"Sample sizes: {sample_sizes}")
        
        all_results = {}
        
        for sample_size in sample_sizes:
            print(f"\n{'='*60}")
            print(f"📊 TESTING WITH SAMPLE SIZE: {sample_size}")
            print(f"{'='*60}")
            
            # Adjust sample size based on available data
            max_normal = min(sample_size, self.normal_count)
            max_pneumonia = min(sample_size, self.pneumonia_count)
            
            print(f"Testing {max_normal} normal + {max_pneumonia} pneumonia images")
            
            # Get random samples
            normal_images = self.get_random_sample(self.normal_path, max_normal)
            pneumonia_images = self.get_random_sample(self.pneumonia_path, max_pneumonia)
            
            for model_name in models_to_test:
                print(f"\n🤖 Testing Model: {model_name.upper()}")
                print("-" * 50)
                
                results = self.test_model_on_sample(
                    normal_images, 
                    pneumonia_images, 
                    model_name,
                    sample_size
                )
                
                all_results[f"{model_name}_sample_{sample_size}"] = results
                self.print_detailed_results(results, model_name, sample_size)
        
        # Summary comparison
        self.print_summary_comparison(all_results)
        
        # Save results
        self.save_comprehensive_results(all_results)
        
        return all_results
    
    def get_random_sample(self, path, size):
        """Get random sample of images from path"""
        if not path.exists():
            return []
        
        all_files = list(path.glob("*.jpeg"))
        return random.sample(all_files, min(size, len(all_files)))
    
    def test_model_on_sample(self, normal_images, pneumonia_images, model_name, sample_size):
        """Test model on a specific sample"""
        
        results = {
            'model': model_name,
            'sample_size': sample_size,
            'normal_count': len(normal_images),
            'pneumonia_count': len(pneumonia_images),
            'predictions': [],
            'start_time': time.time()
        }
        
        total_images = len(normal_images) + len(pneumonia_images)
        processed = 0
        
        # Test normal images
        print(f"\nTesting {len(normal_images)} NORMAL images...")
        for i, image_path in enumerate(normal_images, 1):
            processed += 1
            
            if i % 10 == 0 or i <= 5:  # Show progress for first 5 and every 10th
                print(f"  [{i:3d}/{len(normal_images)}] {image_path.name[:40]:40}", end="")
            
            try:
                result = medical_predictor.predict_from_file(
                    str(image_path),
                    model_name=model_name,
                    enhance_confidence=True,
                    use_ensemble=False
                )
                
                prediction_result = {
                    'image_path': str(image_path),
                    'image_name': image_path.name,
                    'expected': 'normal',
                    'predicted': result['prediction_class'],
                    'confidence': result['confidence_score'],
                    'correct': result['prediction_class'] == 'normal',
                    'class_probabilities': result.get('class_probabilities', {}),
                    'model_version': result.get('model_version', 'unknown')
                }
                
                results['predictions'].append(prediction_result)
                
                if i % 10 == 0 or i <= 5:
                    status = "✅" if prediction_result['correct'] else "❌"
                    print(f" -> {prediction_result['predicted']:9} ({prediction_result['confidence']:.3f}) {status}")
                
            except Exception as e:
                error_result = {
                    'image_path': str(image_path),
                    'image_name': image_path.name,
                    'expected': 'normal',
                    'predicted': 'error',
                    'confidence': 0.0,
                    'correct': False,
                    'error': str(e)
                }
                results['predictions'].append(error_result)
                
                if i % 10 == 0 or i <= 5:
                    print(f" -> ERROR: {str(e)[:30]}")
        
        # Test pneumonia images
        print(f"\nTesting {len(pneumonia_images)} PNEUMONIA images...")
        for i, image_path in enumerate(pneumonia_images, 1):
            processed += 1
            
            if i % 10 == 0 or i <= 5:
                print(f"  [{i:3d}/{len(pneumonia_images)}] {image_path.name[:40]:40}", end="")
            
            try:
                result = medical_predictor.predict_from_file(
                    str(image_path),
                    model_name=model_name,
                    enhance_confidence=True,
                    use_ensemble=False
                )
                
                prediction_result = {
                    'image_path': str(image_path),
                    'image_name': image_path.name,
                    'expected': 'pneumonia',
                    'predicted': result['prediction_class'],
                    'confidence': result['confidence_score'],
                    'correct': result['prediction_class'] == 'pneumonia',
                    'class_probabilities': result.get('class_probabilities', {}),
                    'model_version': result.get('model_version', 'unknown')
                }
                
                results['predictions'].append(prediction_result)
                
                if i % 10 == 0 or i <= 5:
                    status = "✅" if prediction_result['correct'] else "❌"
                    print(f" -> {prediction_result['predicted']:9} ({prediction_result['confidence']:.3f}) {status}")
                
            except Exception as e:
                error_result = {
                    'image_path': str(image_path),
                    'image_name': image_path.name,
                    'expected': 'pneumonia',
                    'predicted': 'error',
                    'confidence': 0.0,
                    'correct': False,
                    'error': str(e)
                }
                results['predictions'].append(error_result)
                
                if i % 10 == 0 or i <= 5:
                    print(f" -> ERROR: {str(e)[:30]}")
        
        results['end_time'] = time.time()
        results['duration'] = results['end_time'] - results['start_time']
        
        # Calculate metrics
        results.update(self.calculate_metrics(results['predictions']))
        
        return results
    
    def calculate_metrics(self, predictions):
        """Calculate comprehensive performance metrics"""
        
        valid_predictions = [p for p in predictions if p['predicted'] != 'error']
        
        if not valid_predictions:
            return {
                'accuracy': 0.0,
                'normal_accuracy': 0.0,
                'pneumonia_accuracy': 0.0,
                'false_positive_rate': 0.0,
                'false_negative_rate': 0.0,
                'avg_confidence': 0.0,
                'total_errors': len(predictions)
            }
        
        # Overall metrics
        total_correct = sum(1 for p in valid_predictions if p['correct'])
        accuracy = total_correct / len(valid_predictions)
        
        # Class-specific metrics
        normal_preds = [p for p in valid_predictions if p['expected'] == 'normal']
        pneumonia_preds = [p for p in valid_predictions if p['expected'] == 'pneumonia']
        
        normal_correct = sum(1 for p in normal_preds if p['correct'])
        pneumonia_correct = sum(1 for p in pneumonia_preds if p['correct'])
        
        normal_accuracy = normal_correct / len(normal_preds) if normal_preds else 0
        pneumonia_accuracy = pneumonia_correct / len(pneumonia_preds) if pneumonia_preds else 0
        
        # Error rates
        false_positives = sum(1 for p in normal_preds if p['predicted'] == 'pneumonia')
        false_negatives = sum(1 for p in pneumonia_preds if p['predicted'] == 'normal')
        
        false_positive_rate = false_positives / len(normal_preds) if normal_preds else 0
        false_negative_rate = false_negatives / len(pneumonia_preds) if pneumonia_preds else 0
        
        # Confidence metrics
        confidences = [p['confidence'] for p in valid_predictions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'accuracy': accuracy,
            'normal_accuracy': normal_accuracy,
            'pneumonia_accuracy': pneumonia_accuracy,
            'normal_correct': normal_correct,
            'normal_total': len(normal_preds),
            'pneumonia_correct': pneumonia_correct,
            'pneumonia_total': len(pneumonia_preds),
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'avg_confidence': avg_confidence,
            'total_correct': total_correct,
            'total_tested': len(valid_predictions),
            'total_errors': len(predictions) - len(valid_predictions)
        }
    
    def print_detailed_results(self, results, model_name, sample_size):
        """Print detailed results for a test run"""
        
        print(f"\n📈 DETAILED RESULTS - {model_name.upper()} (Sample: {sample_size})")
        print("-" * 60)
        print(f"Overall Accuracy:     {results['accuracy']:.1%} ({results['total_correct']}/{results['total_tested']})")
        print(f"Normal Accuracy:      {results['normal_accuracy']:.1%} ({results['normal_correct']}/{results['normal_total']})")
        print(f"Pneumonia Accuracy:   {results['pneumonia_accuracy']:.1%} ({results['pneumonia_correct']}/{results['pneumonia_total']})")
        print(f"Average Confidence:   {results['avg_confidence']:.3f}")
        print(f"False Positive Rate:  {results['false_positive_rate']:.1%}")
        print(f"False Negative Rate:  {results['false_negative_rate']:.1%}")
        print(f"Processing Time:      {results['duration']:.1f} seconds")
        
        if results['total_errors'] > 0:
            print(f"Errors:               {results['total_errors']}")
        
        # Medical assessment
        clinical_pass = 0
        requirements = [
            (results['accuracy'] >= 0.90, "Overall Accuracy ≥ 90%"),
            (results['false_positive_rate'] <= 0.10, "False Positive Rate ≤ 10%"),
            (results['false_negative_rate'] <= 0.05, "False Negative Rate ≤ 5%")
        ]
        
        print(f"\nMedical Requirements:")
        for passed, req in requirements:
            status = "✅" if passed else "❌"
            print(f"  {status} {req}")
            if passed:
                clinical_pass += 1
        
        if clinical_pass == 3:
            verdict = "✅ READY FOR CLINICAL USE"
        elif clinical_pass >= 2:
            verdict = "⚠️  CLOSE TO CLINICAL READINESS"
        else:
            verdict = "❌ NEEDS IMPROVEMENT"
        
        print(f"  {verdict} ({clinical_pass}/3 requirements met)")
    
    def print_summary_comparison(self, all_results):
        """Print comparison across different sample sizes"""
        
        print(f"\n{'='*80}")
        print("📊 SUMMARY COMPARISON ACROSS SAMPLE SIZES")
        print(f"{'='*80}")
        
        print(f"{'Sample Size':<12} {'Accuracy':<10} {'Normal Acc':<12} {'Pneumonia Acc':<14} {'FP Rate':<9} {'FN Rate':<9} {'Clinical'}")
        print("-" * 80)
        
        for key, results in all_results.items():
            sample_size = results['sample_size']
            clinical_pass = sum([
                results['accuracy'] >= 0.90,
                results['false_positive_rate'] <= 0.10,
                results['false_negative_rate'] <= 0.05
            ])
            clinical_status = "✅ READY" if clinical_pass == 3 else f"⚠️  {clinical_pass}/3"
            
            print(f"{sample_size:<12} {results['accuracy']:<10.1%} {results['normal_accuracy']:<12.1%} "
                  f"{results['pneumonia_accuracy']:<14.1%} {results['false_positive_rate']:<9.1%} "
                  f"{results['false_negative_rate']:<9.1%} {clinical_status}")
        
        # Best performing sample size
        best_result = max(all_results.values(), key=lambda x: x['accuracy'])
        print(f"\n🏆 BEST PERFORMANCE: Sample size {best_result['sample_size']} with {best_result['accuracy']:.1%} accuracy")
    
    def save_comprehensive_results(self, all_results):
        """Save comprehensive results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_dataset_test_{timestamp}.json"
        
        # Convert to JSON-serializable format
        json_results = {}
        for key, results in all_results.items():
            json_results[key] = {k: v for k, v in results.items() if k != 'predictions'}
            json_results[key]['prediction_count'] = len(results['predictions'])
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive results saved to: {filename}")

def main():
    """Main testing function"""
    try:
        tester = FullDatasetTester()
        
        if tester.normal_count == 0 and tester.pneumonia_count == 0:
            print("❌ No dataset found!")
            return
        
        # Test with increasing sample sizes
        sample_sizes = [20, 50]  # Start with manageable sizes
        
        # Add larger sizes if we have enough data
        if tester.normal_count >= 100 and tester.pneumonia_count >= 100:
            sample_sizes.append(100)
        
        if tester.normal_count >= 200 and tester.pneumonia_count >= 200:
            sample_sizes.append(200)
        
        print(f"Will test with sample sizes: {sample_sizes}")
        
        results = tester.run_comprehensive_test(
            sample_sizes=sample_sizes,
            models_to_test=["huggingface_vit"]
        )
        
        print("\n🎉 Comprehensive dataset testing completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
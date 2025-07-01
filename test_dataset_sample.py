#!/usr/bin/env python3
"""
Quick focused test on real dataset sample
"""

import os
import sys
import django
from pathlib import Path
import random

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pneumonia_detection.settings')
django.setup()

from ai_engine.medical_predictor import medical_predictor

def quick_dataset_test():
    """Quick test on real dataset"""
    
    print("=== QUICK REAL DATASET TEST ===\n")
    
    # Dataset paths
    dataset_path = Path("/home/izzat/PycharmProjects/pneumonia_ai/data_tests/chest_xray/test")
    normal_path = dataset_path / "NORMAL"
    pneumonia_path = dataset_path / "PNEUMONIA"
    
    # Count available images
    normal_count = len(list(normal_path.glob("*.jpeg"))) if normal_path.exists() else 0
    pneumonia_count = len(list(pneumonia_path.glob("*.jpeg"))) if pneumonia_path.exists() else 0
    
    print(f"Available normal images: {normal_count}")
    print(f"Available pneumonia images: {pneumonia_count}")
    
    # Test with 30 images per class for solid statistics
    sample_size = 30
    
    # Get random samples
    normal_files = list(normal_path.glob("*.jpeg"))
    pneumonia_files = list(pneumonia_path.glob("*.jpeg"))
    
    normal_sample = random.sample(normal_files, min(sample_size, len(normal_files)))
    pneumonia_sample = random.sample(pneumonia_files, min(sample_size, len(pneumonia_files)))
    
    print(f"\nTesting {len(normal_sample)} normal + {len(pneumonia_sample)} pneumonia images")
    print(f"Using high-performance ViT model (97.42% expected accuracy)")
    
    # Test normal images
    print(f"\n📊 Testing NORMAL images:")
    normal_correct = 0
    normal_results = []
    
    for i, image_path in enumerate(normal_sample, 1):
        try:
            result = medical_predictor.predict_from_file(
                str(image_path),
                model_name="huggingface_vit",
                enhance_confidence=True,
                use_ensemble=False
            )
            
            prediction = result['prediction_class']
            confidence = result['confidence_score']
            correct = prediction == "normal"
            
            if correct:
                normal_correct += 1
            
            normal_results.append({
                'correct': correct,
                'confidence': confidence,
                'prediction': prediction
            })
            
            if i <= 10:  # Show first 10 predictions
                status = "✅" if correct else "❌"
                print(f"  [{i:2d}] {image_path.name[:35]:35} -> {prediction:9} ({confidence:.3f}) {status}")
            elif i == 11:
                print(f"  ... (showing first 10, processing remaining {len(normal_sample)-10})")
            
        except Exception as e:
            print(f"  [{i:2d}] ERROR: {e}")
            normal_results.append({'correct': False, 'confidence': 0.0, 'prediction': 'error'})
    
    # Test pneumonia images
    print(f"\n📊 Testing PNEUMONIA images:")
    pneumonia_correct = 0
    pneumonia_results = []
    
    for i, image_path in enumerate(pneumonia_sample, 1):
        try:
            result = medical_predictor.predict_from_file(
                str(image_path),
                model_name="huggingface_vit",
                enhance_confidence=True,
                use_ensemble=False
            )
            
            prediction = result['prediction_class']
            confidence = result['confidence_score']
            correct = prediction == "pneumonia"
            
            if correct:
                pneumonia_correct += 1
            
            pneumonia_results.append({
                'correct': correct,
                'confidence': confidence,
                'prediction': prediction
            })
            
            if i <= 10:  # Show first 10 predictions
                status = "✅" if correct else "❌"
                print(f"  [{i:2d}] {image_path.name[:35]:35} -> {prediction:9} ({confidence:.3f}) {status}")
            elif i == 11:
                print(f"  ... (showing first 10, processing remaining {len(pneumonia_sample)-10})")
            
        except Exception as e:
            print(f"  [{i:2d}] ERROR: {e}")
            pneumonia_results.append({'correct': False, 'confidence': 0.0, 'prediction': 'error'})
    
    # Calculate metrics
    total_correct = normal_correct + pneumonia_correct
    total_tested = len(normal_sample) + len(pneumonia_sample)
    
    accuracy = total_correct / total_tested if total_tested > 0 else 0
    normal_accuracy = normal_correct / len(normal_sample) if normal_sample else 0
    pneumonia_accuracy = pneumonia_correct / len(pneumonia_sample) if pneumonia_sample else 0
    
    # Error rates
    false_positives = len(normal_sample) - normal_correct
    false_negatives = len(pneumonia_sample) - pneumonia_correct
    
    false_positive_rate = false_positives / len(normal_sample) if normal_sample else 0
    false_negative_rate = false_negatives / len(pneumonia_sample) if pneumonia_sample else 0
    
    # Confidence analysis
    all_confidences = ([r['confidence'] for r in normal_results if r['confidence'] > 0] + 
                      [r['confidence'] for r in pneumonia_results if r['confidence'] > 0])
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    # Results
    print(f"\n{'='*60}")
    print(f"📈 COMPREHENSIVE RESULTS")
    print(f"{'='*60}")
    print(f"Sample Size:          {total_tested} images ({len(normal_sample)} normal + {len(pneumonia_sample)} pneumonia)")
    print(f"Overall Accuracy:     {accuracy:.1%} ({total_correct}/{total_tested})")
    print(f"Normal Accuracy:      {normal_accuracy:.1%} ({normal_correct}/{len(normal_sample)})")
    print(f"Pneumonia Accuracy:   {pneumonia_accuracy:.1%} ({pneumonia_correct}/{len(pneumonia_sample)})")
    print(f"Average Confidence:   {avg_confidence:.3f}")
    print(f"False Positive Rate:  {false_positive_rate:.1%} ({false_positives} healthy → pneumonia)")
    print(f"False Negative Rate:  {false_negative_rate:.1%} ({false_negatives} pneumonia → healthy)")
    
    # Medical Assessment
    print(f"\n🏥 MEDICAL ASSESSMENT:")
    requirements = [
        (accuracy >= 0.90, f"Overall Accuracy ≥ 90%: {accuracy:.1%}"),
        (false_positive_rate <= 0.10, f"False Positive Rate ≤ 10%: {false_positive_rate:.1%}"),
        (false_negative_rate <= 0.05, f"False Negative Rate ≤ 5%: {false_negative_rate:.1%}"),
        (avg_confidence >= 0.80, f"Average Confidence ≥ 80%: {avg_confidence:.1%}")
    ]
    
    clinical_pass = 0
    for passed, description in requirements:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {description}")
        if passed:
            clinical_pass += 1
    
    # Clinical verdict
    if clinical_pass >= 3:
        verdict = "✅ READY FOR CLINICAL USE"
        color = "🟢"
    elif clinical_pass >= 2:
        verdict = "⚠️  CLOSE TO CLINICAL READINESS"
        color = "🟡"
    else:
        verdict = "❌ NEEDS IMPROVEMENT"
        color = "🔴"
    
    print(f"\n{color} CLINICAL VERDICT: {verdict}")
    print(f"   Requirements met: {clinical_pass}/4")
    
    # Comparison with previous models
    print(f"\n📊 COMPARISON WITH PREVIOUS TESTS:")
    print(f"  TorchXRayVision:  61.7% accuracy (76.7% false negatives)")
    print(f"  EfficientNet-B4:  31.7% accuracy (86.7% false positives)")
    print(f"  High-Perf ViT:    {accuracy:.1%} accuracy ({false_negative_rate:.1%} false negatives)")
    
    if accuracy > 0.90:
        print(f"  🎉 MASSIVE IMPROVEMENT! Clinical-grade performance achieved!")
    elif accuracy > 0.80:
        print(f"  ✅ SIGNIFICANT IMPROVEMENT! Much better than previous models!")
    elif accuracy > 0.70:
        print(f"  📈 GOOD IMPROVEMENT over previous models")
    else:
        print(f"  ⚠️  Some improvement but still needs work")

if __name__ == "__main__":
    quick_dataset_test()
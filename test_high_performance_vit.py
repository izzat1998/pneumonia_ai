#!/usr/bin/env python3
"""
Test the high-performance Hugging Face ViT model (97.42% accuracy)
"""

import os
import sys
import django
from pathlib import Path
import random

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pneumonia_detection.settings')
django.setup()

from ai_engine.medical_model_loader import medical_model_manager
from ai_engine.medical_predictor import medical_predictor

def test_high_performance_vit():
    """Test the new high-performance ViT model"""
    
    print("=== TESTING HIGH-PERFORMANCE VIT MODEL (97.42% ACCURACY) ===\n")
    
    # Test model loading
    print("1. Loading high-performance ViT model...")
    try:
        model_info = medical_model_manager.get_medical_model_info("huggingface_vit")
        print(f"   Model: {model_info['name']}")
        print(f"   Expected Accuracy: {model_info.get('accuracy', 'Unknown')}")
        print(f"   Input Size: {model_info['input_size']}")
        print(f"   Device: {model_info['device']}")
        print("   ✅ Model configuration loaded")
    except Exception as e:
        print(f"   ❌ Error loading model info: {e}")
        return
    
    # Test on real dataset sample
    dataset_path = Path("/home/izzat/PycharmProjects/pneumonia_ai/data_tests/chest_xray/test")
    normal_path = dataset_path / "NORMAL"
    pneumonia_path = dataset_path / "PNEUMONIA"
    
    if not dataset_path.exists():
        print("❌ Real dataset not found, testing with synthetic images only")
        return test_synthetic_only()
    
    print(f"\n2. Testing on real chest X-ray dataset...")
    print(f"   Dataset path: {dataset_path}")
    
    # Get sample images (smaller sample for quick test)
    sample_size = 10
    
    normal_images = []
    pneumonia_images = []
    
    if normal_path.exists():
        normal_files = list(normal_path.glob("*.jpeg"))
        normal_images = random.sample(normal_files, min(sample_size, len(normal_files)))
    
    if pneumonia_path.exists():
        pneumonia_files = list(pneumonia_path.glob("*.jpeg"))
        pneumonia_images = random.sample(pneumonia_files, min(sample_size, len(pneumonia_files)))
    
    print(f"   Testing {len(normal_images)} normal + {len(pneumonia_images)} pneumonia images")
    
    # Test normal images
    print(f"\n   Testing NORMAL images:")
    normal_correct = 0
    for i, image_path in enumerate(normal_images, 1):
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
            
            status = "✅" if correct else "❌"
            print(f"     [{i:2d}] {image_path.name[:30]:30} -> {prediction:9} ({confidence:.3f}) {status}")
            
        except Exception as e:
            print(f"     [{i:2d}] {image_path.name[:30]:30} -> ERROR: {e}")
    
    # Test pneumonia images
    print(f"\n   Testing PNEUMONIA images:")
    pneumonia_correct = 0
    for i, image_path in enumerate(pneumonia_images, 1):
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
            
            status = "✅" if correct else "❌"
            print(f"     [{i:2d}] {image_path.name[:30]:30} -> {prediction:9} ({confidence:.3f}) {status}")
            
        except Exception as e:
            print(f"     [{i:2d}] {image_path.name[:30]:30} -> ERROR: {e}")
    
    # Calculate performance
    total_correct = normal_correct + pneumonia_correct
    total_images = len(normal_images) + len(pneumonia_images)
    accuracy = total_correct / total_images if total_images > 0 else 0
    
    normal_accuracy = normal_correct / len(normal_images) if normal_images else 0
    pneumonia_accuracy = pneumonia_correct / len(pneumonia_images) if pneumonia_images else 0
    
    print(f"\n3. PERFORMANCE RESULTS:")
    print(f"   Overall Accuracy: {accuracy:.1%} ({total_correct}/{total_images})")
    print(f"   Normal Accuracy:  {normal_accuracy:.1%} ({normal_correct}/{len(normal_images)})")
    print(f"   Pneumonia Accuracy: {pneumonia_accuracy:.1%} ({pneumonia_correct}/{len(pneumonia_images)})")
    
    # Compare with expected performance
    expected_accuracy = 0.9742
    print(f"\n4. COMPARISON WITH EXPECTED:")
    print(f"   Expected Accuracy: {expected_accuracy:.1%}")
    print(f"   Actual Accuracy:   {accuracy:.1%}")
    
    if accuracy >= 0.90:
        print(f"   🎉 EXCELLENT - Meets clinical requirements!")
    elif accuracy >= 0.80:
        print(f"   ✅ GOOD - Significant improvement!")
    elif accuracy >= 0.70:
        print(f"   ⚠️  MODERATE - Better than before but needs improvement")
    else:
        print(f"   ❌ POOR - Still needs work")
    
    # Medical assessment
    false_positive_rate = (len(normal_images) - normal_correct) / len(normal_images) if normal_images else 0
    false_negative_rate = (len(pneumonia_images) - pneumonia_correct) / len(pneumonia_images) if pneumonia_images else 0
    
    print(f"\n5. MEDICAL ASSESSMENT:")
    print(f"   False Positive Rate: {false_positive_rate:.1%}")
    print(f"   False Negative Rate: {false_negative_rate:.1%}")
    
    clinical_requirements = [
        (accuracy >= 0.90, f"Overall Accuracy ≥ 90%: {accuracy:.1%}"),
        (false_positive_rate <= 0.10, f"False Positive Rate ≤ 10%: {false_positive_rate:.1%}"),
        (false_negative_rate <= 0.05, f"False Negative Rate ≤ 5%: {false_negative_rate:.1%}"),
    ]
    
    passed_requirements = 0
    for passed, description in clinical_requirements:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {description}")
        if passed:
            passed_requirements += 1
    
    if passed_requirements == 3:
        verdict = "✅ READY FOR CLINICAL USE"
    elif passed_requirements >= 2:
        verdict = "⚠️  CLOSE TO CLINICAL READINESS"
    else:
        verdict = "❌ NEEDS MORE IMPROVEMENT"
    
    print(f"   {verdict} ({passed_requirements}/3 requirements met)")

def test_synthetic_only():
    """Fallback test with synthetic images"""
    print("Testing with synthetic images only...")
    
    from PIL import Image
    
    # Create test images
    dark_image = Image.new('RGB', (224, 224), (30, 30, 30))
    bright_image = Image.new('RGB', (224, 224), (200, 200, 200))
    
    test_cases = [
        ("Dark Image (Normal-like)", dark_image, "normal"),
        ("Bright Image (Pneumonia-like)", bright_image, "pneumonia")
    ]
    
    for name, image, expected in test_cases:
        temp_path = f"/tmp/test_{name.lower().replace(' ', '_')}.png"
        image.save(temp_path)
        
        try:
            result = medical_predictor.predict_from_file(
                temp_path,
                model_name="huggingface_vit",
                enhance_confidence=True
            )
            
            prediction = result['prediction_class']
            confidence = result['confidence_score']
            correct = prediction == expected
            
            status = "✅" if correct else "❌"
            print(f"   {name}: {prediction} ({confidence:.3f}) {status}")
            
            os.remove(temp_path)
            
        except Exception as e:
            print(f"   {name}: ERROR - {e}")

if __name__ == "__main__":
    test_high_performance_vit()
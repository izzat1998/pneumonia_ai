#!/usr/bin/env python3
"""
Comprehensive test suite for the improved medical AI system
Tests the fixes for false positives and validates medical-grade accuracy
"""

import os
import sys
import django
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pneumonia_detection.settings')
django.setup()

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from ai_engine.medical_model_loader import medical_model_manager
from ai_engine.medical_predictor import medical_predictor

def create_healthy_chest_xray():
    """Create a synthetic healthy chest X-ray pattern"""
    # Create base dark background (air/lungs)
    img = Image.new('RGB', (512, 512), (25, 25, 25))
    draw = ImageDraw.Draw(img)
    
    # Add rib structure (horizontal bright lines)
    for y in range(80, 400, 35):
        # Curved ribs
        for x in range(50, 450, 2):
            curve_offset = int(10 * np.sin((x - 50) * np.pi / 400))
            rib_y = y + curve_offset
            if 50 <= x <= 450 and 50 <= rib_y <= 450:
                draw.point((x, rib_y), fill=(85, 85, 85))
    
    # Add lung boundaries
    # Left lung outline
    for y in range(100, 350):
        draw.point((80, y), fill=(100, 100, 100))
        draw.point((220, y), fill=(100, 100, 100))
    
    # Right lung outline  
    for y in range(100, 350):
        draw.point((290, y), fill=(100, 100, 100))
        draw.point((430, y), fill=(100, 100, 100))
    
    # Add heart shadow (left side)
    for y in range(200, 350):
        for x in range(150, 220):
            if (x - 185)**2 + (y - 275)**2 < 2000:
                draw.point((x, y), fill=(65, 65, 65))
    
    # Add spine shadow (center)
    for y in range(50, 450):
        for x in range(250, 270):
            draw.point((x, y), fill=(120, 120, 120))
    
    # Apply slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    return img

def create_pneumonia_chest_xray():
    """Create a synthetic pneumonia chest X-ray pattern"""
    # Start with healthy pattern
    img = create_healthy_chest_xray()
    draw = ImageDraw.Draw(img)
    
    # Add consolidation in right lower lobe
    for y in range(300, 400):
        for x in range(320, 420):
            # Patchy consolidation pattern
            if np.random.random() < 0.7:
                draw.point((x, y), fill=(150, 150, 150))
    
    # Add some infiltrates in left lung
    for y in range(250, 320):
        for x in range(120, 200):
            if np.random.random() < 0.4:
                draw.point((x, y), fill=(130, 130, 130))
    
    # Apply blur for realistic appearance
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    return img

def create_normal_non_xray():
    """Create a normal non-medical image that should NOT be detected as pneumonia"""
    # Create a landscape-like image
    img = Image.new('RGB', (512, 512), (135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)
    
    # Add some clouds
    for i in range(5):
        x = np.random.randint(50, 400)
        y = np.random.randint(50, 200)
        for j in range(20):
            cx = x + np.random.randint(-30, 30)
            cy = y + np.random.randint(-15, 15)
            draw.ellipse((cx-10, cy-5, cx+10, cy+5), fill=(255, 255, 255))
    
    # Add ground
    for y in range(400, 512):
        for x in range(512):
            draw.point((x, y), fill=(34, 139, 34))  # Forest green
    
    return img

def test_model_predictions():
    """Test the improved model with various test cases"""
    print("=== TESTING IMPROVED MEDICAL AI SYSTEM ===\n")
    
    # Test cases with different models
    test_cases = [
        ("Synthetic Healthy X-ray", create_healthy_chest_xray(), "normal"),
        ("Synthetic Pneumonia X-ray", create_pneumonia_chest_xray(), "pneumonia"), 
        ("Non-medical Image", create_normal_non_xray(), "normal"),
        ("Dark Test Image", Image.new('RGB', (224, 224), (30, 30, 30)), "normal"),
        ("Bright Test Image", Image.new('RGB', (224, 224), (200, 200, 200)), "normal")
    ]
    
    # Test with multiple models
    models_to_test = ["torchxrayvision", "efficientnet_b4"]
    
    results = []
    
    for model_name in models_to_test:
        print(f"\n🤖 TESTING MODEL: {model_name.upper()}")
        print("=" * 60)
        
        for name, image, expected in test_cases:
            print(f"\nTesting: {name} with {model_name}")
            print(f"Expected: {expected}")
            
            try:
                # Save temp image for testing
                temp_path = f"/tmp/test_{name.lower().replace(' ', '_')}.png"
                image.save(temp_path)
                
                # Test with specific model
                result = medical_predictor.predict_from_file(
                    temp_path,
                    model_name=model_name,
                    enhance_confidence=True,
                    use_ensemble=False
                )
                
                prediction = result['prediction_class']
                confidence = result['confidence_score']
                
                # Determine if prediction is correct
                correct = prediction == expected
                
                print(f"Prediction: {prediction}")
                print(f"Confidence: {confidence:.3f}")
                print(f"Model: {result.get('model_version', 'unknown')}")
                print(f"Correct: {'✅' if correct else '❌'}")
                
                # Show medical indicators if available
                if 'medical_indicators' in result:
                    indicators = result['medical_indicators']
                    print(f"Medical indicators:")
                    for indicator, value in indicators.items():
                        print(f"  {indicator}: {value:.3f}")
                
                results.append({
                    'name': f"{name} ({model_name})",
                    'model': model_name,
                    'expected': expected,
                    'predicted': prediction,
                    'confidence': confidence,
                    'correct': correct
                })
                
                # Clean up
                os.remove(temp_path)
                
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'name': f"{name} ({model_name})",
                    'model': model_name,
                    'expected': expected,
                    'predicted': 'error',
                    'confidence': 0.0,
                    'correct': False
                })
            
            print("-" * 40)
    
    # Summary
    print("\n=== RESULTS SUMMARY ===")
    total_tests = len(results)
    correct_predictions = sum(1 for r in results if r['correct'])
    accuracy = correct_predictions / total_tests if total_tests > 0 else 0
    
    print(f"Total tests: {total_tests}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.1%}")
    
    # Detailed results
    print("\nDetailed Results:")
    for result in results:
        status = "✅ PASS" if result['correct'] else "❌ FAIL"
        print(f"{status} {result['name']}: {result['predicted']} (conf: {result['confidence']:.3f})")
    
    # Check for improvements
    print("\n=== IMPROVEMENT ANALYSIS ===")
    
    # Check if healthy images are correctly identified
    healthy_results = [r for r in results if r['expected'] == 'normal']
    healthy_accuracy = sum(1 for r in healthy_results if r['correct']) / len(healthy_results) if healthy_results else 0
    print(f"Healthy image accuracy: {healthy_accuracy:.1%}")
    
    # Check confidence distribution
    confidences = [r['confidence'] for r in results if r['predicted'] != 'error']
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
    
    # Success criteria
    print("\n=== SUCCESS CRITERIA ===")
    criteria = [
        (accuracy >= 0.8, f"Overall accuracy ≥ 80%: {accuracy:.1%}"),
        (healthy_accuracy >= 0.8, f"Healthy image accuracy ≥ 80%: {healthy_accuracy:.1%}"),
        (avg_confidence >= 0.6, f"Average confidence ≥ 60%: {avg_confidence:.3f}"),
    ]
    
    all_passed = True
    for passed, description in criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {description}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL CRITERIA PASSED! Medical AI improvements are successful!")
    else:
        print("\n⚠️  Some criteria failed. Further improvements needed.")
    
    return results

if __name__ == "__main__":
    test_model_predictions()
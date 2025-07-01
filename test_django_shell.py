#!/usr/bin/env python3
"""
Test script to run in Django shell to verify the fix
"""

import numpy as np
from PIL import Image
import os
from pathlib import Path

def run_test():
    """Run the test within Django environment"""
    
    print("=== TESTING 65% CONFIDENCE FIX IN DJANGO ===\n")
    
    # Import within Django context
    from ai_engine.medical_predictor import medical_predictor
    from ai_engine.medical_model_loader import medical_model_manager
    
    print("1. Medical System Status:")
    print(f"   Medical predictor: {medical_predictor is not None}")
    print(f"   Medical model manager: {medical_model_manager is not None}")
    
    if medical_model_manager:
        print(f"   Device: {medical_model_manager.device}")
        print(f"   Available models: {list(medical_model_manager.model_configs.keys())}")
    
    print()
    
    # Create test images
    print("2. Creating Test Images:")
    
    # Very dark (healthy-like)
    dark_img = Image.new('RGB', (224, 224), (30, 30, 30))
    
    # Bright (pneumonia-like)
    bright_img = Image.new('RGB', (224, 224), (180, 180, 180))
    
    # Structured pneumonia-like
    pneumonia_img = Image.new('RGB', (224, 224), (50, 50, 50))
    pn_array = np.array(pneumonia_img)
    pn_array[100:160, 140:190] = [200, 200, 200]  # Bright opacity
    pneumonia_img = Image.fromarray(pn_array)
    
    test_images = {
        'dark_healthy': dark_img,
        'bright_opacity': bright_img, 
        'structured_pneumonia': pneumonia_img
    }
    
    print("3. Testing Predictions:")
    
    results = []
    
    for name, image in test_images.items():
        temp_path = f"/tmp/django_test_{name}.png"
        image.save(temp_path)
        
        try:
            # Test medical predictor
            result = medical_predictor.predict_from_file(
                temp_path,
                model_name='resnet50',  # Should map to torchxrayvision
                enhance_confidence=True
            )
            
            print(f"\n   {name}:")
            print(f"     Prediction: {result['prediction_class']}")
            print(f"     Confidence: {result['confidence_score']:.4f}")
            print(f"     Medical grade: {result.get('medical_grade', False)}")
            print(f"     Model: {result.get('model_version', 'unknown')}")
            
            results.append({
                'name': name,
                'confidence': result['confidence_score'],
                'prediction': result['prediction_class']
            })
            
        except Exception as e:
            print(f"     ❌ Error: {e}")
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    print(f"\n4. Analysis:")
    
    if results:
        confidences = [r['confidence'] for r in results]
        print(f"   Confidence range: {min(confidences):.4f} - {max(confidences):.4f}")
        print(f"   Standard deviation: {np.std(confidences):.4f}")
        
        # Check if still stuck at 65%
        if all(0.63 <= c <= 0.67 for c in confidences):
            print(f"   ❌ STILL STUCK AT ~65% - Need more debugging")
        else:
            print(f"   ✅ CONFIDENCE VARIES - Fix is working!")
        
        # Show results
        for r in results:
            print(f"   {r['name']}: {r['prediction']} ({r['confidence']:.3f})")
    else:
        print(f"   ❌ No successful predictions")
    
    return results

if __name__ == "__main__":
    run_test()
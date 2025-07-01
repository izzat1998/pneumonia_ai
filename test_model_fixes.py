#!/usr/bin/env python3
"""
Quick test of model improvements
"""

import os
import sys

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pneumonia_detection.settings')
import django
django.setup()

from ai_engine.medical_predictor import medical_predictor
from ai_engine.medical_model_loader import medical_model_manager

def test_model_loading():
    """Test if models can be loaded"""
    print("Testing model loading...")
    
    models_to_test = ['huggingface_vit', 'torchxrayvision', 'efficientnet_b4']
    
    for model_name in models_to_test:
        try:
            print(f"\nLoading {model_name}...")
            model = medical_model_manager.load_medical_model(model_name)
            info = medical_model_manager.get_medical_model_info(model_name)
            print(f"✓ {model_name} loaded successfully")
            print(f"  Name: {info['name']}")
            print(f"  Accuracy: {info.get('accuracy', 'Not specified')}")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {str(e)}")

def test_prediction():
    """Test prediction with a sample image"""
    print("\n\nTesting predictions...")
    
    # Create a dummy test image
    from PIL import Image
    import numpy as np
    
    # Create a grayscale image that looks like a chest X-ray
    img_array = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
    # Add some structure to make it more realistic
    img_array[50:174, 50:174] = np.random.randint(100, 150, (124, 124), dtype=np.uint8)
    
    img = Image.fromarray(img_array).convert('RGB')
    img.save('test_xray.jpg')
    
    models = ['huggingface_vit', 'torchxrayvision']
    
    for model_name in models:
        try:
            print(f"\n{model_name}:")
            result = medical_predictor.predict_from_file(
                image_file='test_xray.jpg',
                model_name=model_name,
                enhance_confidence=False,
                use_ensemble=False
            )
            
            print(f"  Prediction: {result['prediction_class']}")
            print(f"  Confidence: {result['confidence_score']:.2%}")
            print(f"  Model version: {result.get('model_version', 'Unknown')}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    # Test ensemble
    try:
        print(f"\nEnsemble prediction:")
        result = medical_predictor.predict_from_file(
            image_file='test_xray.jpg',
            enhance_confidence=True,
            use_ensemble=True
        )
        
        print(f"  Prediction: {result['prediction_class']}")
        print(f"  Confidence: {result['confidence_score']:.2%}")
        if 'individual_predictions' in result:
            print("  Individual predictions:")
            for model, pred, conf in result['individual_predictions']:
                print(f"    {model}: {pred} ({conf:.2%})")
                
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    # Clean up
    os.remove('test_xray.jpg')

def test_threshold_changes():
    """Test if TorchXRayVision thresholds are working better"""
    print("\n\nTesting TorchXRayVision threshold improvements...")
    
    # Create test images with different intensities
    from PIL import Image
    import numpy as np
    
    test_cases = [
        ("dark_xray.jpg", np.random.randint(20, 80, (224, 224), dtype=np.uint8)),
        ("normal_xray.jpg", np.random.randint(80, 150, (224, 224), dtype=np.uint8)),
        ("bright_xray.jpg", np.random.randint(150, 220, (224, 224), dtype=np.uint8))
    ]
    
    for filename, img_array in test_cases:
        # Add some lung-like structure
        h, w = img_array.shape
        center_h, center_w = h // 2, w // 2
        
        # Left lung area
        img_array[center_h-40:center_h+40, center_w-60:center_w-20] = img_array[center_h-40:center_h+40, center_w-60:center_w-20] + 30
        # Right lung area  
        img_array[center_h-40:center_h+40, center_w+20:center_w+60] = img_array[center_h-40:center_h+40, center_w+20:center_w+60] + 30
        
        img = Image.fromarray(img_array).convert('RGB')
        img.save(filename)
        
        try:
            result = medical_predictor.predict_from_file(
                image_file=filename,
                model_name='torchxrayvision',
                enhance_confidence=False
            )
            
            print(f"\n{filename}: {result['prediction_class']} ({result['confidence_score']:.2%})")
            
            if 'pathology_predictions' in result:
                pneumonia_score = result['pathology_predictions'].get('Pneumonia', 0)
                print(f"  Raw pneumonia score: {pneumonia_score:.3f}")
                if 'composite_pneumonia_score' in result:
                    print(f"  Composite score: {result['composite_pneumonia_score']:.3f}")
                
        except Exception as e:
            print(f"{filename}: Error - {str(e)}")
        
        os.remove(filename)

def main():
    print("="*60)
    print("Testing Pneumonia AI Model Improvements")
    print("="*60)
    
    test_model_loading()
    test_prediction()
    test_threshold_changes()
    
    print("\n" + "="*60)
    print("Test completed!")

if __name__ == "__main__":
    main()
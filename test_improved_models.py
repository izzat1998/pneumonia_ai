#!/usr/bin/env python3
"""
Test script for improved pneumonia detection models
Tests all the improvements:
1. Fixed TorchXRayVision thresholds
2. High-performance ViT as default
3. Enhanced preprocessing
4. Ensemble voting
5. Visualization capabilities
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pneumonia_detection.settings')
import django
django.setup()

from ai_engine.medical_predictor import medical_predictor
from ai_engine.medical_model_loader import medical_model_manager
from PIL import Image
import torch

def test_single_model(image_path, model_name):
    """Test a single model"""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print('='*60)
    
    try:
        result = medical_predictor.predict_from_file(
            image_file=image_path,
            model_name=model_name,
            enhance_confidence=True,
            use_ensemble=False,
            generate_visualization=True
        )
        
        print(f"Prediction: {result['prediction_class']}")
        print(f"Confidence: {result['confidence_score']:.2%}")
        print(f"Processing time: {result.get('total_processing_time', 0):.2f}s")
        
        if 'pathology_predictions' in result:
            print("\nPathology scores:")
            for pathology, score in sorted(result['pathology_predictions'].items(), 
                                         key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {pathology}: {score:.3f}")
        
        if 'visualization' in result and 'explanation' in result['visualization']:
            print(f"\nExplanation: {result['visualization']['explanation']}")
        
        return result
        
    except Exception as e:
        print(f"Error testing {model_name}: {str(e)}")
        return None

def test_ensemble(image_path):
    """Test ensemble prediction"""
    print(f"\n{'='*60}")
    print("Testing ENSEMBLE prediction")
    print('='*60)
    
    try:
        result = medical_predictor.predict_from_file(
            image_file=image_path,
            enhance_confidence=True,
            use_ensemble=True,
            generate_visualization=False
        )
        
        print(f"Ensemble Prediction: {result['prediction_class']}")
        print(f"Ensemble Confidence: {result['confidence_score']:.2%}")
        
        if 'individual_predictions' in result:
            print("\nIndividual model predictions:")
            for model, pred, conf in result['individual_predictions']:
                print(f"  {model}: {pred} ({conf:.2%})")
        
        return result
        
    except Exception as e:
        print(f"Error in ensemble prediction: {str(e)}")
        return None

def test_dataset_performance():
    """Test on multiple images from the dataset"""
    test_dir = Path("data_tests")
    
    if not test_dir.exists():
        print(f"Test directory {test_dir} not found")
        return
    
    # Find test images
    pneumonia_dir = test_dir / "Pneumonia"
    normal_dir = test_dir / "Normal"
    
    test_images = []
    
    if pneumonia_dir.exists():
        pneumonia_images = list(pneumonia_dir.glob("*.jpeg")) + list(pneumonia_dir.glob("*.jpg"))
        test_images.extend([(img, "pneumonia") for img in pneumonia_images[:5]])
    
    if normal_dir.exists():
        normal_images = list(normal_dir.glob("*.jpeg")) + list(normal_dir.glob("*.jpg"))
        test_images.extend([(img, "normal") for img in normal_images[:5]])
    
    if not test_images:
        print("No test images found")
        return
    
    # Test each model
    models = ['huggingface_vit', 'torchxrayvision', 'efficientnet_b4']
    results = {model: {'correct': 0, 'total': 0, 'predictions': []} for model in models}
    results['ensemble'] = {'correct': 0, 'total': 0, 'predictions': []}
    
    print(f"\nTesting on {len(test_images)} images...")
    
    for img_path, true_label in test_images:
        print(f"\nTesting: {img_path.name} (True: {true_label})")
        
        # Test individual models
        for model_name in models:
            result = test_single_model(str(img_path), model_name)
            if result:
                predicted = result['prediction_class']
                confidence = result['confidence_score']
                is_correct = predicted == true_label
                
                results[model_name]['total'] += 1
                if is_correct:
                    results[model_name]['correct'] += 1
                
                results[model_name]['predictions'].append({
                    'image': img_path.name,
                    'true': true_label,
                    'predicted': predicted,
                    'confidence': confidence,
                    'correct': is_correct
                })
        
        # Test ensemble
        ensemble_result = test_ensemble(str(img_path))
        if ensemble_result:
            predicted = ensemble_result['prediction_class']
            confidence = ensemble_result['confidence_score']
            is_correct = predicted == true_label
            
            results['ensemble']['total'] += 1
            if is_correct:
                results['ensemble']['correct'] += 1
            
            results['ensemble']['predictions'].append({
                'image': img_path.name,
                'true': true_label,
                'predicted': predicted,
                'confidence': confidence,
                'correct': is_correct
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print('='*60)
    
    for model_name, model_results in results.items():
        if model_results['total'] > 0:
            accuracy = model_results['correct'] / model_results['total']
            print(f"\n{model_name}:")
            print(f"  Accuracy: {accuracy:.2%} ({model_results['correct']}/{model_results['total']})")
            
            # Calculate per-class accuracy
            pneumonia_correct = sum(1 for p in model_results['predictions'] 
                                   if p['true'] == 'pneumonia' and p['correct'])
            pneumonia_total = sum(1 for p in model_results['predictions'] 
                                 if p['true'] == 'pneumonia')
            normal_correct = sum(1 for p in model_results['predictions'] 
                               if p['true'] == 'normal' and p['correct'])
            normal_total = sum(1 for p in model_results['predictions'] 
                             if p['true'] == 'normal')
            
            if pneumonia_total > 0:
                print(f"  Pneumonia accuracy: {pneumonia_correct/pneumonia_total:.2%} ({pneumonia_correct}/{pneumonia_total})")
            if normal_total > 0:
                print(f"  Normal accuracy: {normal_correct/normal_total:.2%} ({normal_correct}/{normal_total})")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"improved_model_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")

def main():
    """Main test function"""
    print("Testing Improved Pneumonia Detection Models")
    print("==========================================")
    
    # Test single image first
    test_images = [
        "data_tests/Pneumonia/Pneumonia (1010).jpeg",
        "data_tests/Normal/Normal (1317).jpeg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting single image: {img_path}")
            
            # Test default model (should be huggingface_vit now)
            print("\nTesting DEFAULT model:")
            result = medical_predictor.predict_from_file(
                image_file=img_path,
                enhance_confidence=True,
                generate_visualization=True
            )
            print(f"Default model used: {result.get('model_version', 'Unknown')}")
            print(f"Prediction: {result['prediction_class']} ({result['confidence_score']:.2%})")
            
            break
    
    # Run comprehensive dataset test
    print("\n" + "="*60)
    print("COMPREHENSIVE DATASET TEST")
    print("="*60)
    test_dataset_performance()

if __name__ == "__main__":
    main()
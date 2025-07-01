#!/usr/bin/env python3
"""
Test script to verify the normalization fix resolves the 65% confidence issue
"""

import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def test_normalization_fix():
    """Test the corrected normalization"""
    print("=== TESTING NORMALIZATION FIX ===\n")
    
    try:
        import torchxrayvision as xrv
        from ai_engine.medical_model_loader import medical_model_manager
        
        print("1. Loading TorchXRayVision model with corrected preprocessing...")
        
        # Create test images with different characteristics
        test_images = {
            "very_dark": Image.new('RGB', (224, 224), (10, 10, 10)),      # Almost black (healthy-like)
            "dark": Image.new('RGB', (224, 224), (50, 50, 50)),          # Dark gray (healthy-like)
            "medium": Image.new('RGB', (224, 224), (128, 128, 128)),     # Medium gray
            "bright": Image.new('RGB', (224, 224), (200, 200, 200)),     # Bright (pneumonia-like opacity)
            "very_bright": Image.new('RGB', (224, 224), (240, 240, 240)) # Very bright
        }
        
        print("2. Testing with corrected preprocessing:")
        
        # Test corrected preprocessing
        processor = medical_model_manager.get_medical_processor("torchxrayvision")
        model = medical_model_manager.load_medical_model("torchxrayvision")
        
        results = {}
        
        for name, image in test_images.items():
            # Preprocess with corrected pipeline
            tensor = processor(image).unsqueeze(0)
            
            print(f"\n   {name.upper()} image:")
            print(f"     Input tensor range: [{tensor.min():.1f}, {tensor.max():.1f}]")
            print(f"     Input tensor mean: {tensor.mean():.1f}")
            
            # Check if in expected DICOM range
            if tensor.min() >= -1024 and tensor.max() <= 1024:
                print(f"     ✅ Input is in correct DICOM range [-1024, 1024]")
            else:
                print(f"     ❌ Input is NOT in DICOM range!")
            
            # Get model prediction
            with torch.no_grad():
                model.eval()
                raw_output = model(tensor)
                probs = torch.sigmoid(raw_output)
                
                pneumonia_prob = probs[0, 8].item()  # Index 8 is pneumonia
                normal_prob = 1.0 - pneumonia_prob
                
                prediction = "pneumonia" if pneumonia_prob > 0.5 else "normal"
                confidence = max(pneumonia_prob, normal_prob)
                
                results[name] = {
                    'pneumonia_prob': pneumonia_prob,
                    'confidence': confidence,
                    'prediction': prediction
                }
                
                print(f"     Pneumonia probability: {pneumonia_prob:.4f}")
                print(f"     Prediction: {prediction}")
                print(f"     Confidence: {confidence:.4f}")
        
        print(f"\n3. Results Analysis:")
        
        # Check for variation in outputs
        pneumonia_probs = [r['pneumonia_prob'] for r in results.values()]
        confidences = [r['confidence'] for r in results.values()]
        
        pneumonia_std = np.std(pneumonia_probs)
        confidence_std = np.std(confidences)
        
        print(f"   Pneumonia probability variation (std): {pneumonia_std:.4f}")
        print(f"   Confidence variation (std): {confidence_std:.4f}")
        
        if pneumonia_std < 0.05:  # Less than 5% variation
            print(f"   ⚠️  WARNING: Still very low variation in pneumonia probabilities!")
            print(f"   The model might still have calibration issues.")
        else:
            print(f"   ✅ Good variation in pneumonia probabilities - fix is working!")
        
        # Check if any clearly different patterns emerge
        very_dark_conf = results['very_dark']['confidence']
        very_bright_conf = results['very_bright']['confidence']
        
        print(f"\n   Very dark image confidence: {very_dark_conf:.4f}")
        print(f"   Very bright image confidence: {very_bright_conf:.4f}")
        print(f"   Difference: {abs(very_dark_conf - very_bright_conf):.4f}")
        
        if abs(very_dark_conf - very_bright_conf) > 0.1:  # 10% difference
            print(f"   ✅ Significant difference detected - normalization fix is working!")
        else:
            print(f"   ⚠️  Still small difference - may need additional fixes")
        
        print(f"\n4. Detailed Results:")
        for name, result in results.items():
            print(f"   {name}: {result['prediction']} ({result['confidence']:.3f})")
        
        # Test if fix resolved the "consistent 65%" issue
        if all(0.63 <= r['confidence'] <= 0.67 for r in results.values()):
            print(f"\n   ❌ ISSUE PERSISTS: All confidences still around 65%")
            print(f"   Additional debugging needed - model might have other issues")
        else:
            print(f"\n   ✅ SUCCESS: Confidence values now vary based on input!")
        
        return results
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return None

def test_with_real_xray_patterns():
    """Test with more realistic X-ray patterns"""
    print("\n=== TESTING WITH REALISTIC X-RAY PATTERNS ===\n")
    
    try:
        from ai_engine.medical_model_loader import medical_model_manager
        
        # Create more realistic test patterns
        def create_healthy_xray():
            """Create a pattern resembling healthy lungs"""
            img = np.full((224, 224, 3), 40, dtype=np.uint8)  # Dark background
            
            # Add lung outlines (brighter)
            img[50:180, 30:100] = 80   # Left lung
            img[50:180, 130:200] = 80  # Right lung
            
            # Add rib shadows (darker lines)
            for y in range(60, 170, 15):
                img[y:y+2, 30:200] = 20
            
            return Image.fromarray(img)
        
        def create_pneumonia_xray():
            """Create a pattern resembling pneumonia opacity"""
            img = np.full((224, 224, 3), 40, dtype=np.uint8)  # Dark background
            
            # Add lung outlines
            img[50:180, 30:100] = 80   # Left lung
            img[50:180, 130:200] = 80  # Right lung
            
            # Add pneumonia-like opacity (brighter patch)
            img[100:160, 140:190] = 180  # Bright opacity in right lower lobe
            img[90:170, 130:200] = 120   # Less intense opacity around it
            
            # Add rib shadows
            for y in range(60, 170, 15):
                img[y:y+2, 30:200] = 20
                
            return Image.fromarray(img)
        
        healthy_pattern = create_healthy_xray()
        pneumonia_pattern = create_pneumonia_xray()
        
        patterns = {
            "healthy_pattern": healthy_pattern,
            "pneumonia_pattern": pneumonia_pattern
        }
        
        processor = medical_model_manager.get_medical_processor("torchxrayvision")
        model = medical_model_manager.load_medical_model("torchxrayvision")
        
        print("Testing with realistic X-ray patterns:")
        
        for name, image in patterns.items():
            tensor = processor(image).unsqueeze(0)
            
            with torch.no_grad():
                model.eval()
                raw_output = model(tensor)
                probs = torch.sigmoid(raw_output)
                
                pneumonia_prob = probs[0, 8].item()
                prediction = "pneumonia" if pneumonia_prob > 0.5 else "normal"
                confidence = max(pneumonia_prob, 1.0 - pneumonia_prob)
                
                print(f"\n   {name.upper()}:")
                print(f"     Prediction: {prediction}")
                print(f"     Pneumonia probability: {pneumonia_prob:.4f}")
                print(f"     Confidence: {confidence:.4f}")
                
                # Expected behavior
                if name == "healthy_pattern" and prediction == "normal":
                    print(f"     ✅ Correctly identified as normal")
                elif name == "pneumonia_pattern" and prediction == "pneumonia":
                    print(f"     ✅ Correctly identified as pneumonia")
                else:
                    print(f"     ⚠️  Unexpected prediction")
        
    except Exception as e:
        print(f"Error during realistic pattern testing: {e}")

if __name__ == "__main__":
    results = test_normalization_fix()
    test_with_real_xray_patterns()
    
    print("\n=== SUMMARY ===")
    print("If normalization fix worked, you should see:")
    print("1. Input tensors in range [-1024, 1024]")
    print("2. Variation in confidence scores (not all ~65%)")  
    print("3. Different predictions for different image types")
    print("4. No normalization warnings from TorchXRayVision")
#!/usr/bin/env python3
"""
Test preprocessing pipeline on sample images to verify dataset integrity
"""

from medical_model_loader import MedicalModelManager
from PIL import Image
import os
import numpy as np

# Initialize model manager
manager = MedicalModelManager()

# Test preprocessing on sample images
test_dir = '/home/izzat/PycharmProjects/pneumonia_ai/data_tests/chest_xray/test'
sample_images = []

# Get 2 normal and 2 pneumonia samples
normal_dir = os.path.join(test_dir, 'NORMAL')
pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')

normal_files = [f for f in os.listdir(normal_dir) if f.endswith('.jpeg')][:2]
pneumonia_files = [f for f in os.listdir(pneumonia_dir) if f.endswith('.jpeg')][:2]

for f in normal_files:
    sample_images.append((os.path.join(normal_dir, f), 'normal', f))
for f in pneumonia_files:
    sample_images.append((os.path.join(pneumonia_dir, f), 'pneumonia', f))

print('=== PREPROCESSING VALIDATION ===')
for filepath, true_label, filename in sample_images:
    try:
        with Image.open(filepath) as img:
            # Test both model preprocessors
            for model_name in ['huggingface_vit', 'torchxrayvision']:
                try:
                    processor = manager.get_medical_processor(model_name)
                    
                    # Convert to RGB for processing
                    if img.mode != 'RGB':
                        rgb_img = img.convert('RGB')
                    else:
                        rgb_img = img
                    
                    if model_name == 'huggingface_vit':
                        processed = processor(images=rgb_img, return_tensors='pt')
                        tensor_shape = processed['pixel_values'].shape
                    else:
                        tensor = processor(rgb_img).unsqueeze(0)
                        tensor_shape = tensor.shape
                    
                    print(f'{filename} ({true_label}) -> {model_name}: {tensor_shape} - OK')
                except Exception as e:
                    print(f'{filename} ({true_label}) -> {model_name}: ERROR - {e}')
    except Exception as e:
        print(f'{filename}: IMAGE_LOAD_ERROR - {e}')

print('\n=== MANUAL PREDICTION TEST ===')
# Test actual predictions on these samples
try:
    from medical_predictor import medical_predictor
    if medical_predictor:
        for filepath, true_label, filename in sample_images[:2]:  # Test first 2 images
            try:
                result = medical_predictor.predict_from_file(
                    image_file=filepath,
                    model_name='huggingface_vit',
                    enhance_confidence=False,
                    use_ensemble=False,
                    generate_visualization=False
                )
                
                pred_class = result.get('prediction_class', 'unknown')
                confidence = result.get('confidence_score', 0.0)
                class_probs = result.get('class_probabilities', {})
                
                print(f'{filename} (true: {true_label}) -> pred: {pred_class}, conf: {confidence:.3f}')
                print(f'  Class probabilities: {class_probs}')
                
            except Exception as e:
                print(f'{filename}: PREDICTION_ERROR - {e}')
    else:
        print('Medical predictor not available')
except Exception as e:
    print(f'Prediction test failed: {e}')
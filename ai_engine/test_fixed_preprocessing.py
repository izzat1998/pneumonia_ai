#!/usr/bin/env python3
"""
Test preprocessing with properly loaded models
"""

from medical_model_loader import MedicalModelManager
from PIL import Image
import os

manager = MedicalModelManager()

# Load the model first to initialize processors
model = manager.load_medical_model('huggingface_vit')
print('Model and processor loaded')

# Test preprocessing
test_dir = '/home/izzat/PycharmProjects/pneumonia_ai/data_tests/chest_xray/test'
sample_file = os.path.join(test_dir, 'NORMAL', 'IM-0007-0001.jpeg')

with Image.open(sample_file) as img:
    print(f'Original image: {img.size} {img.mode}')
    
    # Convert to RGB for processing
    if img.mode != 'RGB':
        rgb_img = img.convert('RGB')
    else:
        rgb_img = img
    
    # Get processor and test
    processor = manager.get_medical_processor('huggingface_vit')
    print(f'Processor type: {type(processor)}')
    
    # Process image
    processed = processor(images=rgb_img, return_tensors='pt')
    tensor_shape = processed['pixel_values'].shape
    print(f'Processed tensor shape: {tensor_shape}')
    print('Preprocessing successful!')

# Test actual prediction
print('\n=== TESTING ACTUAL PREDICTION ===')
try:
    result = manager.predict_medical(processed['pixel_values'], 'huggingface_vit')
    print('Prediction result:', result)
except Exception as e:
    print(f'Prediction error: {e}')
from rest_framework import serializers
from .models import AnalysisResult
from django.core.files.uploadedfile import InMemoryUploadedFile

class ImageUploadSerializer(serializers.Serializer):
    """Serializer for image upload and analysis"""
    
    image = serializers.ImageField(
        max_length=None,
        allow_empty_file=False,
        use_url=False,
        help_text="Image file (JPEG, PNG, BMP, TIFF)"
    )
    model_name = serializers.CharField(
        max_length=50,
        default="huggingface_vit",
        required=False,
        help_text="AI model to use for prediction"
    )
    enhance_image = serializers.BooleanField(
        default=True,
        required=False,
        help_text="Apply medical image enhancement"
    )
    use_ensemble = serializers.BooleanField(
        default=False,
        required=False,
        help_text="Use ensemble of multiple models for higher accuracy"
    )
    generate_visualization = serializers.BooleanField(
        default=False,
        required=False,
        help_text="Generate AI visualization heatmap"
    )
    
    def validate_image(self, value):
        """Validate uploaded image"""
        if not isinstance(value, InMemoryUploadedFile):
            raise serializers.ValidationError("Invalid image file")
        
        # Check file size (10MB limit)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("Image file too large (max 10MB)")
        
        # Check file format
        allowed_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        file_ext = value.name.lower().split('.')[-1]
        if f'.{file_ext}' not in allowed_formats:
            raise serializers.ValidationError(f"Unsupported format. Allowed: {', '.join(allowed_formats)}")
        
        return value


class AnalysisResultSerializer(serializers.ModelSerializer):
    """Serializer for analysis results"""
    
    confidence_percentage = serializers.ReadOnlyField()
    is_pneumonia_detected = serializers.ReadOnlyField()
    
    class Meta:
        model = AnalysisResult
        fields = [
            'id',
            'session_key',
            'original_filename',
            'file_size',
            'image_width',
            'image_height',
            'prediction_class',
            'confidence_score',
            'confidence_percentage',
            'is_pneumonia_detected',
            'model_version',
            'processing_time',
            'raw_probabilities',
            'status',
            'error_message',
            'created_at',
            'updated_at'
        ]
        read_only_fields = [
            'id',
            'session_key',
            'file_size',
            'image_width',
            'image_height',
            'prediction_class',
            'confidence_score',
            'model_version',
            'processing_time',
            'raw_probabilities',
            'status',
            'error_message',
            'created_at',
            'updated_at'
        ]


class AnalysisListSerializer(serializers.ModelSerializer):
    """Simplified serializer for analysis list"""
    
    confidence_percentage = serializers.ReadOnlyField()
    is_pneumonia_detected = serializers.ReadOnlyField()
    
    class Meta:
        model = AnalysisResult
        fields = [
            'id',
            'original_filename',
            'prediction_class',
            'confidence_percentage',
            'is_pneumonia_detected',
            'model_version',
            'status',
            'created_at'
        ]


class BatchUploadSerializer(serializers.Serializer):
    """Serializer for batch image upload"""
    
    images = serializers.ListField(
        child=serializers.ImageField(max_length=None, allow_empty_file=False, use_url=False),
        max_length=10,  # Max 10 images per batch
        min_length=1,
        help_text="List of image files (max 10)"
    )
    model_name = serializers.CharField(
        max_length=50,
        default="resnet50",
        required=False
    )
    enhance_image = serializers.BooleanField(
        default=True,
        required=False
    )
    
    def validate_images(self, value):
        """Validate all images in batch"""
        total_size = 0
        allowed_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        for image in value:
            # Check individual file size
            if image.size > 10 * 1024 * 1024:
                raise serializers.ValidationError(f"Image {image.name} too large (max 10MB)")
            
            total_size += image.size
            
            # Check format
            file_ext = image.name.lower().split('.')[-1]
            if f'.{file_ext}' not in allowed_formats:
                raise serializers.ValidationError(f"Unsupported format in {image.name}")
        
        # Check total batch size (50MB limit)
        if total_size > 50 * 1024 * 1024:
            raise serializers.ValidationError("Total batch size too large (max 50MB)")
        
        return value


class ModelInfoSerializer(serializers.Serializer):
    """Serializer for model information"""
    
    name = serializers.CharField()
    version = serializers.CharField()
    source = serializers.CharField() 
    classes = serializers.ListField(child=serializers.CharField())
    input_size = serializers.ListField(child=serializers.IntegerField())
    device = serializers.CharField()
    loaded = serializers.BooleanField()
    compile_enabled = serializers.BooleanField()
    parameters = serializers.IntegerField(required=False)
    trainable_parameters = serializers.IntegerField(required=False)


class SystemStatsSerializer(serializers.Serializer):
    """Serializer for system statistics"""
    
    total_analyses = serializers.IntegerField()
    pneumonia_detected = serializers.IntegerField()
    normal_detected = serializers.IntegerField()
    avg_confidence = serializers.FloatField()
    avg_processing_time = serializers.FloatField()
    recent_analyses = serializers.IntegerField()
    active_sessions = serializers.IntegerField()
    
    # Model statistics
    models_loaded = serializers.IntegerField()
    total_predictions = serializers.IntegerField()
    high_confidence_predictions = serializers.IntegerField()
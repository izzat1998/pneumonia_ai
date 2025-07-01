from django.db import models
from django.contrib.sessions.models import Session
import uuid
from pathlib import Path
from django.utils import timezone
from datetime import timedelta


class AnalysisResultManager(models.Manager):
    """Custom manager with optimized queries"""
    
    def for_session(self, session_key):
        """Get results for a specific session"""
        return self.filter(session_key=session_key, status='completed')
    
    def recent(self, hours=24):
        """Get recent results within specified hours"""
        cutoff = timezone.now() - timedelta(hours=hours)
        return self.filter(created_at__gte=cutoff, status='completed')
    
    def statistics(self):
        """Get aggregated statistics"""
        return self.aggregate(
            total=models.Count('id'),
            pneumonia_detected=models.Count('id', filter=models.Q(prediction_class='pneumonia')),
            avg_confidence=models.Avg('confidence_score'),
            avg_processing_time=models.Avg('processing_time')
        )
    
    def cleanup_old_files(self, days=7):
        """Clean up old analysis files"""
        cutoff = timezone.now() - timedelta(days=days)
        old_results = self.filter(created_at__lt=cutoff)
        
        for result in old_results:
            result.delete_file()
            
        return old_results.delete()


class AnalysisResult(models.Model):
    """Main model for storing analysis results using Django 5.2 features"""
    
    # Core fields
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session_key = models.CharField(max_length=40, db_index=True)
    
    # Image information
    image = models.ImageField(upload_to='uploads/%Y/%m/%d/', max_length=255)
    original_filename = models.CharField(max_length=255)
    file_size = models.PositiveIntegerField()  # bytes
    image_width = models.PositiveIntegerField()
    image_height = models.PositiveIntegerField()
    
    # Analysis results
    prediction_class = models.CharField(
        max_length=20,
        choices=[('normal', 'Normal'), ('pneumonia', 'Pneumonia')]
    )
    confidence_score = models.FloatField()  # 0.0 to 1.0
    model_version = models.CharField(max_length=50, default='resnet50-v1.0')
    processing_time = models.FloatField()  # seconds
    
    # Additional prediction data
    raw_probabilities = models.JSONField(default=dict, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    
    # Status tracking
    status = models.CharField(
        max_length=20,
        choices=[
            ('processing', 'Processing'),
            ('completed', 'Completed'),
            ('failed', 'Failed'),
            ('deleted', 'Deleted')
        ],
        default='processing'
    )
    error_message = models.TextField(blank=True)
    
    # Custom manager
    objects = AnalysisResultManager()
    
    class Meta:
        db_table = 'analysis_results'
        indexes = [
            models.Index(fields=['session_key', '-created_at']),
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['prediction_class', 'confidence_score']),
        ]
        ordering = ['-created_at']
        
    def __str__(self):
        return f"Analysis {self.id} - {self.prediction_class} ({self.confidence_percentage}%)"
    
    @property
    def confidence_percentage(self):
        return round(self.confidence_score * 100, 1)
    
    @property
    def is_pneumonia_detected(self):
        return self.prediction_class == 'pneumonia'
    
    def delete_file(self):
        """Clean up associated image file"""
        if self.image and Path(self.image.path).exists():
            Path(self.image.path).unlink()


class ModelMetrics(models.Model):
    """Track model performance metrics"""
    
    model_name = models.CharField(max_length=100)
    version = models.CharField(max_length=50)
    
    # Performance metrics
    total_predictions = models.PositiveIntegerField(default=0)
    avg_processing_time = models.FloatField(default=0.0)
    avg_confidence_score = models.FloatField(default=0.0)
    
    # Distribution stats
    normal_predictions = models.PositiveIntegerField(default=0)
    pneumonia_predictions = models.PositiveIntegerField(default=0)
    high_confidence_predictions = models.PositiveIntegerField(default=0)  # >80%
    
    # Timestamps
    first_prediction = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'model_metrics'
        unique_together = ['model_name', 'version']
        
    def __str__(self):
        return f"{self.model_name} v{self.version} - {self.total_predictions} predictions"
    
    def update_metrics(self, result):
        """Update metrics with new analysis result"""
        self.total_predictions += 1
        
        # Update averages
        total = self.total_predictions
        self.avg_processing_time = ((self.avg_processing_time * (total - 1)) + result.processing_time) / total
        self.avg_confidence_score = ((self.avg_confidence_score * (total - 1)) + result.confidence_score) / total
        
        # Update distributions
        if result.prediction_class == 'normal':
            self.normal_predictions += 1
        else:
            self.pneumonia_predictions += 1
            
        if result.confidence_score > 0.8:
            self.high_confidence_predictions += 1
            
        self.save()


class SystemLog(models.Model):
    """System monitoring and error logging"""
    
    LEVEL_CHOICES = [
        ('DEBUG', 'Debug'),
        ('INFO', 'Info'),
        ('WARNING', 'Warning'),
        ('ERROR', 'Error'),
        ('CRITICAL', 'Critical'),
    ]
    
    level = models.CharField(max_length=10, choices=LEVEL_CHOICES)
    message = models.TextField()
    category = models.CharField(max_length=50)  # 'model', 'upload', 'analysis', etc.
    
    # Context information
    session_key = models.CharField(max_length=40, blank=True)
    analysis_id = models.UUIDField(null=True, blank=True)
    
    # Request context
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    
    # Additional data
    extra_data = models.JSONField(default=dict, blank=True)
    
    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'system_logs'
        indexes = [
            models.Index(fields=['level', '-created_at']),
            models.Index(fields=['category', '-created_at']),
            models.Index(fields=['session_key', '-created_at']),
        ]
        ordering = ['-created_at']
        
    def __str__(self):
        return f"[{self.level}] {self.category}: {self.message[:50]}..."

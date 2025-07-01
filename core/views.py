from rest_framework import status, viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import action
from django.shortcuts import get_object_or_404, render
from django.db.models import Count, Avg, Q
from django.contrib.sessions.models import Session
from django.utils import timezone
from datetime import timedelta
import logging

from .models import AnalysisResult, ModelMetrics, SystemLog
from .serializers import (
    ImageUploadSerializer, 
    AnalysisResultSerializer,
    AnalysisListSerializer,
    BatchUploadSerializer,
    ModelInfoSerializer,
    SystemStatsSerializer
)
from ai_engine import predictor, model_manager
# Import medical predictor for enhanced medical-grade predictions
try:
    from ai_engine.medical_predictor import medical_predictor
    MEDICAL_MODE_AVAILABLE = medical_predictor is not None
except ImportError:
    MEDICAL_MODE_AVAILABLE = False
    medical_predictor = None

logger = logging.getLogger(__name__)

class AnalysisAPIView(APIView):
    """Main API endpoint for image analysis"""
    
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        """Upload and analyze image"""
        try:
            # Validate input
            serializer = ImageUploadSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    {'error': 'Invalid input', 'details': serializer.errors},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get session key
            session_key = request.session.session_key
            if not session_key:
                request.session.create()
                session_key = request.session.session_key
            
            # Extract validated data
            image = serializer.validated_data['image']
            model_name = serializer.validated_data['model_name']
            enhance_image = serializer.validated_data['enhance_image']
            
            # Create analysis record with safe defaults
            analysis = AnalysisResult.objects.create(
                session_key=session_key,
                image=image,
                original_filename=image.name,
                file_size=image.size,
                image_width=0,  # Will be updated by predictor
                image_height=0,  # Will be updated by predictor
                prediction_class='normal',  # Default value, will be updated
                confidence_score=0.0,  # Default value, will be updated
                processing_time=0.0,  # Default value, will be updated
                status='processing',
                ip_address=self._get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', '')
            )
            
            try:
                # Get additional parameters from request
                use_ensemble = serializer.validated_data.get('use_ensemble', False)
                generate_visualization = serializer.validated_data.get('generate_visualization', False)
                
                # Perform prediction using medical-grade model if available
                if MEDICAL_MODE_AVAILABLE:
                    logger.info("Using medical-grade prediction with enhanced models")
                    prediction_result = medical_predictor.predict_from_file(
                        image_file=image,
                        model_name=model_name,
                        session_key=session_key,
                        request_meta=request.META,
                        enhance_confidence=True,  # Enable confidence enhancement
                        use_ensemble=use_ensemble,
                        generate_visualization=generate_visualization
                    )
                else:
                    logger.warning("Medical predictor not available, falling back to standard predictor")
                    prediction_result = predictor.predict_from_file(
                        image_file=image,
                        model_name=model_name,
                        session_key=session_key,
                        request_meta=request.META
                    )
                
                # Update analysis record with results (with null safety and validation)
                analysis.image_width = max(0, prediction_result.get('image_width', 0))
                analysis.image_height = max(0, prediction_result.get('image_height', 0))
                
                # Validate prediction class
                prediction_class = prediction_result.get('prediction_class', 'normal')
                if prediction_class not in ['normal', 'pneumonia']:
                    prediction_class = 'normal'
                analysis.prediction_class = prediction_class
                
                # Validate confidence score (must be between 0.0 and 1.0)
                confidence_score = prediction_result.get('confidence_score', 0.0)
                if confidence_score is None or not isinstance(confidence_score, (int, float)):
                    confidence_score = 0.0
                analysis.confidence_score = max(0.0, min(1.0, float(confidence_score)))
                
                analysis.model_version = prediction_result.get('model_version', 'unknown-v1.0') or 'unknown-v1.0'
                
                # Validate processing time
                processing_time = prediction_result.get('processing_time', 0.0)
                if processing_time is None or not isinstance(processing_time, (int, float)):
                    processing_time = 0.0
                analysis.processing_time = max(0.0, float(processing_time))
                
                analysis.raw_probabilities = prediction_result.get('class_probabilities', {})
                analysis.status = 'completed'
                analysis.save()
                
                # Return success response
                response_serializer = AnalysisResultSerializer(analysis)
                return Response(
                    {
                        'success': True,
                        'analysis': response_serializer.data,
                        'message': f'Analysis completed: {analysis.prediction_class} detected with {analysis.confidence_score:.1%} confidence'
                    },
                    status=status.HTTP_200_OK
                )
                
            except Exception as e:
                # Update analysis record with error (with safe defaults)
                try:
                    analysis.status = 'failed'
                    analysis.error_message = str(e)
                    # Ensure required fields have valid values before saving
                    if not hasattr(analysis, 'confidence_score') or analysis.confidence_score is None:
                        analysis.confidence_score = 0.0
                    if not hasattr(analysis, 'processing_time') or analysis.processing_time is None:
                        analysis.processing_time = 0.0
                    if not hasattr(analysis, 'prediction_class') or not analysis.prediction_class:
                        analysis.prediction_class = 'normal'
                    analysis.save()
                except Exception as save_error:
                    logger.error(f"Failed to save error status for analysis {analysis.id}: {str(save_error)}")
                
                logger.error(f"Analysis failed for {analysis.id}: {str(e)}")
                
                return Response(
                    {
                        'error': 'Analysis failed',
                        'message': str(e),
                        'analysis_id': str(analysis.id)
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            return Response(
                {'error': 'Internal server error', 'message': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _get_client_ip(self, request):
        """Get client IP address"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


class AnalysisViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for analysis results"""
    
    serializer_class = AnalysisResultSerializer
    
    def get_queryset(self):
        """Get analyses for current session"""
        session_key = self.request.session.session_key
        if not session_key:
            return AnalysisResult.objects.none()
        
        return AnalysisResult.objects.filter(
            session_key=session_key
        ).order_by('-created_at')
    
    def list(self, request):
        """List all analyses for current session"""
        queryset = self.get_queryset()
        serializer = AnalysisListSerializer(queryset, many=True)
        
        return Response({
            'count': queryset.count(),
            'results': serializer.data
        })
    
    def retrieve(self, request, pk=None):
        """Get specific analysis result"""
        try:
            analysis = get_object_or_404(self.get_queryset(), pk=pk)
            serializer = AnalysisResultSerializer(analysis)
            return Response(serializer.data)
        except Exception as e:
            return Response(
                {'error': 'Analysis not found', 'message': str(e)},
                status=status.HTTP_404_NOT_FOUND
            )
    
    @action(detail=False, methods=['delete'])
    def clear_session(self, request):
        """Clear all analyses for current session"""
        session_key = request.session.session_key
        if not session_key:
            return Response({'message': 'No session found'})
        
        # Delete files and records
        analyses = self.get_queryset()
        deleted_count = 0
        
        for analysis in analyses:
            try:
                analysis.delete_file()
                analysis.delete()
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete analysis {analysis.id}: {str(e)}")
        
        return Response({
            'message': f'Cleared {deleted_count} analyses from session',
            'deleted_count': deleted_count
        })


class ModelInfoAPIView(APIView):
    """API endpoint for model information"""
    
    def get(self, request):
        """Get information about available models"""
        try:
            model_name = request.query_params.get('model', 'resnet50')
            
            # Get model info
            model_info = model_manager.get_model_info(model_name)
            
            # Get model metrics from database
            try:
                metrics = ModelMetrics.objects.get(
                    model_name=model_info['name'],
                    version=model_info['version']
                )
                model_info.update({
                    'total_predictions': metrics.total_predictions,
                    'avg_processing_time': metrics.avg_processing_time,
                    'avg_confidence_score': metrics.avg_confidence_score,
                    'normal_predictions': metrics.normal_predictions,
                    'pneumonia_predictions': metrics.pneumonia_predictions,
                    'high_confidence_predictions': metrics.high_confidence_predictions
                })
            except ModelMetrics.DoesNotExist:
                model_info.update({
                    'total_predictions': 0,
                    'avg_processing_time': 0.0,
                    'avg_confidence_score': 0.0,
                    'normal_predictions': 0,
                    'pneumonia_predictions': 0,
                    'high_confidence_predictions': 0
                })
            
            serializer = ModelInfoSerializer(model_info)
            return Response(serializer.data)
            
        except Exception as e:
            logger.error(f"Model info error: {str(e)}")
            return Response(
                {'error': 'Failed to get model information', 'message': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SystemStatsAPIView(APIView):
    """API endpoint for system statistics"""
    
    def get(self, request):
        """Get system statistics"""
        try:
            # Get analysis statistics
            total_analyses = AnalysisResult.objects.filter(status='completed').count()
            pneumonia_detected = AnalysisResult.objects.filter(
                status='completed', 
                prediction_class='pneumonia'
            ).count()
            normal_detected = total_analyses - pneumonia_detected
            
            # Get averages
            avg_stats = AnalysisResult.objects.filter(status='completed').aggregate(
                avg_confidence=Avg('confidence_score'),
                avg_processing_time=Avg('processing_time')
            )
            
            # Get recent analyses (last 24 hours)
            recent_cutoff = timezone.now() - timedelta(hours=24)
            recent_analyses = AnalysisResult.objects.filter(
                status='completed',
                created_at__gte=recent_cutoff
            ).count()
            
            # Get active sessions (last 24 hours)
            active_sessions = AnalysisResult.objects.filter(
                created_at__gte=recent_cutoff
            ).values('session_key').distinct().count()
            
            # Get model statistics
            model_metrics = ModelMetrics.objects.aggregate(
                total_predictions=Count('total_predictions'),
                high_confidence_predictions=Count('high_confidence_predictions')
            )
            
            stats_data = {
                'total_analyses': total_analyses,
                'pneumonia_detected': pneumonia_detected,
                'normal_detected': normal_detected,
                'avg_confidence': avg_stats['avg_confidence'] or 0.0,
                'avg_processing_time': avg_stats['avg_processing_time'] or 0.0,
                'recent_analyses': recent_analyses,
                'active_sessions': active_sessions,
                'models_loaded': len(model_manager.models),
                'total_predictions': model_metrics['total_predictions'] or 0,
                'high_confidence_predictions': model_metrics['high_confidence_predictions'] or 0
            }
            
            serializer = SystemStatsSerializer(stats_data)
            return Response(serializer.data)
            
        except Exception as e:
            logger.error(f"System stats error: {str(e)}")
            return Response(
                {'error': 'Failed to get system statistics', 'message': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HomeView(APIView):
    """Render the main application page"""
    
    def get(self, request):
        """Render the home page"""
        return render(request, 'index.html')

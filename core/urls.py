from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create router for ViewSets
router = DefaultRouter()
router.register('analyses', views.AnalysisViewSet, basename='analysis')

# Define URL patterns
urlpatterns = [
    # Home page
    path('', views.HomeView.as_view(), name='home'),
    
    # API endpoints
    path('api/analyze/', views.AnalysisAPIView.as_view(), name='api-analyze'),
    path('api/model-info/', views.ModelInfoAPIView.as_view(), name='api-model-info'),
    path('api/stats/', views.SystemStatsAPIView.as_view(), name='api-stats'),
    
    # ViewSet routes (includes list, retrieve, clear_session)
    path('api/', include(router.urls)),
]
"""
Controllers Package

This package contains controller classes that coordinate data flow between models and views
in the MVC (Model-View-Controller) architecture of the Eye Movement Analysis application.

Controllers handle:
1. User input processing from views
2. Data manipulation through models
3. Updating views with new data
4. Coordination of application flow
5. Managing the application's business logic
"""

from .base_controller import BaseController
from .data_processing_controller import DataProcessingController
from .visualization_controller import VisualizationController
from .feature_controller import FeatureController

__all__ = [
    'BaseController',
    'DataProcessingController',
    'VisualizationController', 
    'FeatureController'
]
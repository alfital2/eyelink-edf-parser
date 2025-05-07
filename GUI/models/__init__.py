"""
Models Package

This package contains model classes that represent data entities and business logic
in the MVC (Model-View-Controller) architecture of the Eye Movement Analysis application.

Models are responsible for:
1. Data representation and encapsulation
2. Business logic and calculations
3. State management
4. Notifying controllers of data changes
"""

from .eye_tracking_data import EyeTrackingData
from .feature_model import FeatureModel
from .roi_model import ROIModel

__all__ = [
    'EyeTrackingData',
    'FeatureModel',
    'ROIModel'
]
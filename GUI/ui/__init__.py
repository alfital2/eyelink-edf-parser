# UI Components Package

from .main_window import EyeMovementAnalysisGUI
from .theme_manager import ThemeManager
from .feature_table_manager import FeatureTableManager
from .animated_roi_scanpath import AnimatedROIScanpathWidget

__all__ = [
    'EyeMovementAnalysisGUI',
    'ThemeManager',
    'FeatureTableManager',
    'AnimatedROIScanpathWidget',
]
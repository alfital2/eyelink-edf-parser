"""
Settings Manager Module

Provides a centralized way to manage application settings using QSettings.
"""

from PyQt5.QtCore import QSettings


class SettingsManager:
    """
    Manager for application settings using QSettings.
    Provides methods to load and save settings with default values.
    """
    
    def __init__(self, organization="ASD_Analysis", application="EyeMovementAnalysis"):
        """Initialize the settings manager with organization and application names."""
        self.settings = QSettings(organization, application)
        
    def get_screen_dimensions(self):
        """Get the screen dimensions from settings or return defaults."""
        screen_width = int(self.settings.value("screen_width", 1280))
        screen_height = int(self.settings.value("screen_height", 1024))
        return screen_width, screen_height
        
    def save_screen_dimensions(self, width, height, aspect_ratio_index=-1):
        """Save screen dimensions and optional aspect ratio index."""
        self.settings.setValue("screen_width", width)
        self.settings.setValue("screen_height", height)
        if aspect_ratio_index >= 0:
            self.settings.setValue("aspect_ratio_index", aspect_ratio_index)
        self.settings.sync()
        
    def get_aspect_ratio_index(self):
        """Get the saved aspect ratio index or return default."""
        return int(self.settings.value("aspect_ratio_index", -1))
        
    def save_setting(self, key, value):
        """Save a generic setting by key and value."""
        self.settings.setValue(key, value)
        self.settings.sync()
        
    def get_setting(self, key, default=None):
        """Get a generic setting by key with optional default value."""
        return self.settings.value(key, default)
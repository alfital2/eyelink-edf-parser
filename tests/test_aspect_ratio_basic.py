"""
Basic tests for the aspect ratio feature in the Eye Movement Analysis application.
These tests verify that the aspect ratio options are correctly defined in the GUI.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Ensure QApplication is created before any QWidgets
from PyQt5.QtWidgets import QApplication
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

# Import GUI modules
from GUI.gui import EyeMovementAnalysisGUI
from GUI.utils.settings_manager import SettingsManager


class TestAspectRatioBasic(unittest.TestCase):
    """Basic tests for the aspect ratio feature."""
    
    def setUp(self):
        """Set up test with minimal mocking."""
        # Create mock for settings manager
        self.settings_patcher = patch('GUI.utils.settings_manager.QSettings')
        self.mock_settings = self.settings_patcher.start()
        
        # Patch the MovieEyeTrackingVisualizer only for this test
        # Using patch in a more localized way that won't affect other tests
        self.visualizer_patcher = patch('GUI.visualization.eyelink_visualizer.MovieEyeTrackingVisualizer')
        self.mock_visualizer = self.visualizer_patcher.start()
        
        # Create minimal GUI
        self.gui = None
        try:
            with patch('GUI.gui.SettingsManager') as mock_manager_class:
                # Create a mock manager that returns standard values
                manager = MagicMock()
                manager.get_screen_dimensions.return_value = (1280, 1024)
                manager.get_aspect_ratio_index.return_value = 0
                mock_manager_class.return_value = manager
                
                # Create the GUI with our mocked components
                self.gui = EyeMovementAnalysisGUI()
        except Exception as e:
            self.fail(f"Failed to create GUI: {e}")
    
    def tearDown(self):
        """Clean up after tests."""
        if self.gui:
            self.gui.close()
        self.settings_patcher.stop()
        self.visualizer_patcher.stop()  # Stop the visualizer patch
    
    def test_aspect_ratio_combo_exists(self):
        """Test that the aspect ratio combo box exists."""
        self.assertTrue(hasattr(self.gui, 'aspect_ratio_combo'))
    
    def test_aspect_ratio_options_exist(self):
        """Test that aspect ratio combo box has the expected options."""
        # The aspect ratio combo box should have 6 items
        self.assertEqual(self.gui.aspect_ratio_combo.count(), 6)
        
        # Check that the expected options are present
        expected_options = [
            "1280 x 1024 (5:4)",
            "1920 x 1080 (16:9)",
            "1366 x 768 (16:9)",
            "1440 x 900 (16:10)",
            "1024 x 768 (4:3)",
            "3840 x 2160 (4K 16:9)"
        ]
        
        # Check each option text
        for i, expected in enumerate(expected_options):
            self.assertEqual(self.gui.aspect_ratio_combo.itemText(i), expected)
            
    def test_aspect_ratio_data_values(self):
        """Test that aspect ratio combo box items have the correct data values."""
        # Check data for each option
        expected_data = [
            (1280, 1024),  # 5:4
            (1920, 1080),  # 16:9
            (1366, 768),   # 16:9
            (1440, 900),   # 16:10
            (1024, 768),   # 4:3
            (3840, 2160)   # 4K 16:9
        ]
        
        # Verify all data values
        for i, expected in enumerate(expected_data):
            actual = self.gui.aspect_ratio_combo.itemData(i)
            self.assertEqual(actual, expected)
    
    def test_settings_manager_creation(self):
        """Test that SettingsManager can be created."""
        # Patch QSettings for this test
        with patch('GUI.utils.settings_manager.QSettings'):
            # Create an instance of SettingsManager
            settings_mgr = SettingsManager()
            
            # Verify it has the methods we need
            self.assertTrue(hasattr(settings_mgr, 'get_screen_dimensions'))
            self.assertTrue(hasattr(settings_mgr, 'save_screen_dimensions'))
            self.assertTrue(hasattr(settings_mgr, 'get_aspect_ratio_index'))


# Simple test runner
if __name__ == '__main__':
    unittest.main()
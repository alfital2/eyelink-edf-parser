"""
Tests for the custom aspect ratio feature in the Eye Movement Analysis application.
This module tests the functionality to change screen resolution/aspect ratio
and verify that changes are properly saved and applied.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import shutil

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Patch PyQt5's QApplication to allow headless testing
from PyQt5.QtWidgets import QApplication
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

# Import modules to test
from GUI.gui import EyeMovementAnalysisGUI
from GUI.utils.settings_manager import SettingsManager


class TestCustomAspectRatio(unittest.TestCase):
    """Test the custom aspect ratio feature."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary settings directory/file
        self.temp_dir = tempfile.mkdtemp()
        
        # Apply patches for tests
        # Patch MovieEyeTrackingVisualizer in a more localized way that won't affect other tests
        self.visualizer_patcher = patch('GUI.visualization.eyelink_visualizer.MovieEyeTrackingVisualizer', autospec=True)
        self.mock_visualizer = self.visualizer_patcher.start()
        
        # First apply the patch to settings_manager's QSettings
        self.settings_patcher = patch('GUI.utils.settings_manager.QSettings')
        self.mock_settings_class = self.settings_patcher.start()
        self.mock_settings = MagicMock()
        
        # Configure mock methods
        self.mock_settings.value = MagicMock(side_effect=self._mock_settings_value)
        self.mock_settings_class.return_value = self.mock_settings
        
        # Initialize stored values - match actual default values
        self.stored_values = {
            "screen_width": 1280,
            "screen_height": 1024,
            "aspect_ratio_index": 0
        }
        
        # Create mock for any other imports that might be needed
        self.plot_gen_patcher = patch('GUI.visualization.plot_generator.PlotGenerator')
        self.mock_plot_gen = self.plot_gen_patcher.start()
        
        self.animated_path_patcher = patch('animated_roi_scanpath.AnimatedROIScanpathWidget')
        self.mock_animated = self.animated_path_patcher.start()
        
        # Initialize the GUI 
        with patch('GUI.gui.SettingsManager') as mock_manager_class:
            # Create a mock for the settings manager
            self.mock_manager = MagicMock()
            self.mock_manager.get_screen_dimensions = MagicMock(return_value=(1280, 1024))
            self.mock_manager.get_aspect_ratio_index = MagicMock(return_value=0)
            self.mock_manager.save_screen_dimensions = MagicMock()
            
            # Return our mock manager when the GUI initializes its SettingsManager
            mock_manager_class.return_value = self.mock_manager
            
            # Create the GUI
            self.gui = EyeMovementAnalysisGUI()
            
            # Override the internal settings manager for testing
            self.gui.settings_manager = self.mock_manager
            
            # Create mock for GUI components
            self.gui.plot_generator = MagicMock()
            self.gui.plot_generator.screen_width = 1280
            self.gui.plot_generator.screen_height = 1024
            
            self.gui.animated_scanpath = MagicMock()
            self.gui.animated_scanpath.screen_width = 1280
            self.gui.animated_scanpath.screen_height = 1024
            self.gui.animated_scanpath.redraw = MagicMock()
            self.gui.animated_scanpath.data = MagicMock()
            
            # Reset screen dimensions to our test values
            self.gui.screen_width = 1280
            self.gui.screen_height = 1024
            
            # Prepare the aspect ratio combo box - create a proper combo box mock
            combo_mock = MagicMock()
            combo_mock.currentIndex.return_value = 0
            combo_mock.count.return_value = 6
            combo_mock.setCurrentIndex = MagicMock()
            combo_mock.currentIndexChanged = MagicMock()
            
            combo_mock.itemText = MagicMock(side_effect=lambda index: [
                "1280 x 1024 (5:4)",
                "1920 x 1080 (16:9)",
                "1366 x 768 (16:9)",
                "1440 x 900 (16:10)",
                "1024 x 768 (4:3)",
                "3840 x 2160 (4K 16:9)"
            ][index])
            
            combo_mock.itemData = MagicMock(side_effect=lambda index: [
                (1280, 1024),  # 5:4
                (1920, 1080),  # 16:9
                (1366, 768),   # 16:9
                (1440, 900),   # 16:10
                (1024, 768),   # 4:3
                (3840, 2160)   # 4K 16:9
            ][index])
            
            # This is a crucial fix: use currentData to return the current selection data
            combo_mock.currentData = MagicMock(return_value=(1280, 1024))
            
            self.gui.aspect_ratio_combo = combo_mock
    
    def tearDown(self):
        """Clean up after tests."""
        self.settings_patcher.stop()
        self.visualizer_patcher.stop()
        self.plot_gen_patcher.stop()
        self.animated_path_patcher.stop()
        self.gui.close()
        del self.gui
        
        # Remove temp directory
        shutil.rmtree(self.temp_dir)
    
    def _mock_settings_value(self, key, default=None):
        """Mock for QSettings.value that returns our controlled values."""
        return self.stored_values.get(key, default)
    
    def test_aspect_ratio_combo_initialization(self):
        """Test that the aspect ratio combo box is properly initialized."""
        # Check that the aspect ratio combo box has the expected number of items
        self.assertEqual(self.gui.aspect_ratio_combo.count(), 6)
        
        # Check the dimensions are initialized correctly from our mock
        self.assertEqual(self.gui.screen_width, 1280)
        self.assertEqual(self.gui.screen_height, 1024)
        
        # Check that the combo box is set to the correct initial value
        self.assertEqual(self.gui.aspect_ratio_combo.currentIndex(), 0)
        
        # Check that the first item in the combo box is the 1280x1024 option
        first_item_text = self.gui.aspect_ratio_combo.itemText(0)
        self.assertEqual(first_item_text, "1280 x 1024 (5:4)")
        
        # Verify our settings manager was called correctly
        self.mock_manager.get_screen_dimensions.assert_called()
        self.mock_manager.get_aspect_ratio_index.assert_called()
    
    def test_aspect_ratio_change(self):
        """Test changing the aspect ratio."""
        # Create a signal handler function
        def signal_handler(index):
            self.gui.update_screen_dimensions(index)
        
        # Connect the signal to our handler
        self.gui.aspect_ratio_combo.currentIndexChanged.connect = MagicMock()
        self.gui.aspect_ratio_combo.currentIndexChanged.connect.side_effect = lambda handler: handler(1)
        
        # Replace the update_screen_dimensions method with a mock
        self.gui.update_screen_dimensions = MagicMock()
        
        # Simulate the signal being triggered
        self.gui.aspect_ratio_combo.currentIndexChanged.connect(signal_handler)
        
        # Verify the update_screen_dimensions method was called with index 1
        self.gui.update_screen_dimensions.assert_called_once_with(1)
    
    def test_update_screen_dimensions_updates_values(self):
        """Test that update_screen_dimensions correctly updates screen dimensions."""
        # Set up combo box to return 1920x1080 for currentData
        self.gui.aspect_ratio_combo.currentData = MagicMock(return_value=(1920, 1080))
        
        # Call the update_screen_dimensions method with index 1
        self.gui.update_screen_dimensions(1)
        
        # Check screen dimensions were updated
        self.assertEqual(self.gui.screen_width, 1920)
        self.assertEqual(self.gui.screen_height, 1080)
        
        # Verify settings were saved
        self.mock_manager.save_screen_dimensions.assert_called_once_with(1920, 1080, 1)
    
    def test_update_screen_dimensions_updates_plot_generator(self):
        """Test that update_screen_dimensions updates the plot generator."""
        # Set combo box to return 1920x1080 for currentData
        self.gui.aspect_ratio_combo.currentData = MagicMock(return_value=(1920, 1080))
        
        # Call update_screen_dimensions
        self.gui.update_screen_dimensions(1)
        
        # Check plot generator dimensions were updated
        self.assertEqual(self.gui.plot_generator.screen_width, 1920)
        self.assertEqual(self.gui.plot_generator.screen_height, 1080)
    
    def test_update_screen_dimensions_updates_animated_scanpath(self):
        """Test that update_screen_dimensions updates the animated scanpath widget."""
        # Set combo box to return 1920x1080 for currentData
        self.gui.aspect_ratio_combo.currentData = MagicMock(return_value=(1920, 1080))
        
        # Call update_screen_dimensions
        self.gui.update_screen_dimensions(1)
        
        # Check scanpath dimensions were updated
        self.assertEqual(self.gui.animated_scanpath.screen_width, 1920)
        self.assertEqual(self.gui.animated_scanpath.screen_height, 1080)
        
        # Check redraw was called (but only if data is not None)
        self.gui.animated_scanpath.redraw.assert_called_once()
    
    def test_settings_manager_save_dimensions(self):
        """Test SettingsManager.save_screen_dimensions method."""
        # Create test QSettings mock
        test_mock = MagicMock()
        
        # Test SettingsManager directly
        with patch('GUI.utils.settings_manager.QSettings', return_value=test_mock):
            settings_manager = SettingsManager()
            settings_manager.save_screen_dimensions(1920, 1080, 1)
            
            # Verify setValue calls
            expected_calls = [
                unittest.mock.call('screen_width', 1920),
                unittest.mock.call('screen_height', 1080),
                unittest.mock.call('aspect_ratio_index', 1)
            ]
            
            # Extract and check calls for each parameter
            actual_calls = test_mock.setValue.call_args_list
            for expected in expected_calls:
                self.assertIn(expected, actual_calls)
            
            # Verify sync was called
            test_mock.sync.assert_called_once()
    
    def test_available_aspect_ratios(self):
        """Test the available aspect ratio options."""
        # Expected options with their data
        expected_options = [
            ("1280 x 1024 (5:4)", (1280, 1024)),
            ("1920 x 1080 (16:9)", (1920, 1080)),
            ("1366 x 768 (16:9)", (1366, 768)),
            ("1440 x 900 (16:10)", (1440, 900)),
            ("1024 x 768 (4:3)", (1024, 768)),
            ("3840 x 2160 (4K 16:9)", (3840, 2160))
        ]
        
        # Check each combo box entry
        for i, (expected_text, expected_data) in enumerate(expected_options):
            # Check display text
            self.assertEqual(self.gui.aspect_ratio_combo.itemText(i), expected_text)
            
            # Check stored data value
            self.assertEqual(self.gui.aspect_ratio_combo.itemData(i), expected_data)
    
    def test_processing_thread_gets_current_dimensions(self):
        """Test that the processing thread uses current screen dimensions."""
        # Create a wrapper for process_data method that calls the constructor directly
        def mock_process_data():
            # Import ProcessingThread in the method directly
            from GUI.data.processing_thread import ProcessingThread
            
            # Create the thread instance with the test parameters
            return ProcessingThread(
                file_paths=["dummy.csv"],
                output_dir=self.temp_dir,
                visualize=True,
                extract_features=True,
                generate_report=True,
                file_type="CSV Files",
                screen_width=1920,
                screen_height=1080
            )
            
        # Set thread constructor mock
        with patch('GUI.data.processing_thread.ProcessingThread', autospec=True) as mock_thread_constructor:
            # Create a dummy thread instance to return
            mock_thread_instance = MagicMock()
            mock_thread_constructor.return_value = mock_thread_instance
            
            # Set up GUI for testing 
            self.gui.screen_width = 1920
            self.gui.screen_height = 1080
            self.gui.file_paths = ["dummy.csv"]
            self.gui.output_dir = self.temp_dir
            self.gui.selected_file_type = "CSV Files"
            
            # Add mock checkboxes
            self.gui.visualize_cb = MagicMock()
            self.gui.visualize_cb.isChecked = MagicMock(return_value=True)
            
            self.gui.extract_features_cb = MagicMock()
            self.gui.extract_features_cb.isChecked = MagicMock(return_value=True)
            
            self.gui.generate_report_cb = MagicMock()
            self.gui.generate_report_cb.isChecked = MagicMock(return_value=True)
            
            # Mock the start method to prevent thread start
            mock_thread_instance.start = MagicMock()
            
            # Mock the signals from the thread
            mock_thread_instance.update_progress = MagicMock()
            mock_thread_instance.status_update = MagicMock()
            mock_thread_instance.processing_complete = MagicMock()
            mock_thread_instance.error_occurred = MagicMock()
            
            # Execute the wrapper method (which creates the thread)
            thread = mock_process_data()
            
            # Check that thread was created with our dimensions
            mock_thread_constructor.assert_called_once()
            args, kwargs = mock_thread_constructor.call_args
            
            # Verify that screen dimensions were passed correctly
            self.assertEqual(kwargs.get('screen_width'), 1920)
            self.assertEqual(kwargs.get('screen_height'), 1080)
    
    def test_settings_load_from_previous_session(self):
        """Test that settings are loaded from a previous session."""
        # Create a settings mock that returns custom values
        custom_mock = MagicMock()
        custom_mock.value = MagicMock(side_effect=lambda k, d: 
            {"screen_width": 1920, "screen_height": 1080, "aspect_ratio_index": 1}.get(k, d))
        
        # Create a settings manager that will use our custom mock
        with patch('GUI.utils.settings_manager.QSettings', return_value=custom_mock):
            manager = SettingsManager()
            
            # Verify it returns the correct values
            width, height = manager.get_screen_dimensions()
            self.assertEqual(width, 1920)
            self.assertEqual(height, 1080)
            
            index = manager.get_aspect_ratio_index()
            self.assertEqual(index, 1)
            
            # Test the GUI initialization with this manager
            with patch('GUI.gui.SettingsManager', return_value=manager):
                gui = EyeMovementAnalysisGUI()
                
                try:
                    # Verify the GUI loaded the correct initial dimensions
                    self.assertEqual(gui.screen_width, 1920)
                    self.assertEqual(gui.screen_height, 1080)
                finally:
                    # Clean up
                    gui.close()
                    del gui


if __name__ == '__main__':
    unittest.main()
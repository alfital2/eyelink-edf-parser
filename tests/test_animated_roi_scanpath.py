import unittest
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animated_roi_scanpath import AnimatedROIScanpathWidget
from roi_manager import ROIManager
from PyQt5.QtWidgets import QApplication

# Create a QApplication instance for testing
app = QApplication.instance()
if app is None:
    app = QApplication([])

TEST_ROI_FILE = "test_data/test_roi.json"


class TestAnimatedROIScanpath(unittest.TestCase):
    """
    Test suite for the Animated ROI Scanpath module.
    Tests validate the functionality of the animated scanpath with ROI visualization.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Create sample eye tracking data with normalized coordinates
        cls.eye_data = pd.DataFrame({
            'timestamp': list(range(1000, 1100, 10)),  # 10 samples
            'frame_number': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'x_left': [256, 260, 270, 280, 290, 300, 310, 320, 330, 340],
            'y_left': [200, 205, 210, 215, 220, 225, 230, 235, 240, 245],
            'x_right': [830, 835, 840, 845, 850, 855, 860, 865, 870, 875],
            'y_right': [600, 605, 610, 615, 620, 625, 630, 635, 640, 645],
            'pupil_left': [1200, 1210, 1220, 1230, 1240, 1250, 1260, 1270, 1280, 1290],
            'pupil_right': [1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190],
            'is_fixation_left': [True, True, False, False, True, True, False, False, True, True],
            'is_fixation_right': [True, True, False, False, True, True, False, False, True, True],
        })

        # Define screen dimensions for normalization
        cls.screen_width = 1280
        cls.screen_height = 1024

        # Add normalized coordinates
        cls.eye_data['x_left_norm'] = cls.eye_data['x_left'] / cls.screen_width
        cls.eye_data['y_left_norm'] = cls.eye_data['y_left'] / cls.screen_height
        cls.eye_data['x_right_norm'] = cls.eye_data['x_right'] / cls.screen_width
        cls.eye_data['y_right_norm'] = cls.eye_data['y_right'] / cls.screen_height

        # Create a temporary directory for test outputs
        cls.temp_dir = tempfile.TemporaryDirectory()

        # Get the full path to the test ROI file
        cls.roi_file_path = os.path.join(os.path.dirname(__file__), TEST_ROI_FILE)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove temporary directory and files
        cls.temp_dir.cleanup()

    def create_widget_with_data(self):
        """Helper to create a widget with test data loaded."""
        # Create the widget
        widget = AnimatedROIScanpathWidget()

        # Store the data
        widget.data = self.eye_data.copy()
        widget.movie_name = "Test Movie"
        widget.screen_width = self.screen_width
        widget.screen_height = self.screen_height

        # Calculate relative time for display
        widget.data['time_sec'] = (widget.data['timestamp'] - widget.data['timestamp'].iloc[0]) / 1000.0
        widget.total_duration = widget.data['time_sec'].iloc[-1]

        # Load the ROI data
        widget.roi_manager.load_roi_file(self.roi_file_path)

        return widget

    def test_roi_widget_initialization(self):
        """Test initialization of the animated ROI scanpath widget."""
        # Create the widget
        widget = AnimatedROIScanpathWidget()

        # Verify default settings
        self.assertTrue(widget.show_rois, "ROI display should be enabled by default")
        self.assertTrue(widget.highlight_active_roi, "ROI highlighting should be enabled by default")
        self.assertTrue(widget.show_roi_labels, "ROI labels should be enabled by default")
        self.assertTrue(widget.show_left_eye, "Left eye display should be enabled by default")
        self.assertTrue(widget.show_right_eye, "Right eye display should be enabled by default")

        # Verify ROI manager is initialized
        self.assertIsNotNone(widget.roi_manager, "ROI manager should be initialized")

    def test_roi_toggle_functions(self):
        """Test the toggle functions for ROI display options."""
        # Create the widget with data
        widget = self.create_widget_with_data()

        # Test toggling ROI display
        widget.toggle_roi_display(False)
        self.assertFalse(widget.show_rois, "ROI display should be disabled")

        widget.toggle_roi_display(True)
        self.assertTrue(widget.show_rois, "ROI display should be enabled")

        # Test toggling ROI highlighting
        widget.toggle_roi_highlight(False)
        self.assertFalse(widget.highlight_active_roi, "ROI highlighting should be disabled")

        widget.toggle_roi_highlight(True)
        self.assertTrue(widget.highlight_active_roi, "ROI highlighting should be enabled")

        # Test toggling ROI labels
        widget.toggle_roi_labels(False)
        self.assertFalse(widget.show_roi_labels, "ROI labels should be disabled")

        widget.toggle_roi_labels(True)
        self.assertTrue(widget.show_roi_labels, "ROI labels should be enabled")

    def test_roi_detection(self):
        """Test ROI detection at gaze points."""
        # Create the widget with data
        widget = self.create_widget_with_data()

        # Set current frame to one that has ROI data
        widget.current_frame = 0  # Frame 1 in the data

        # Create test points that should match ROIs in the test data
        test_points = [
            {'frame': 1, 'x': 0.25, 'y': 0.25, 'expected_label': 'Face'},
            {'frame': 1, 'x': 0.65, 'y': 0.65, 'expected_label': 'Hand'},
            {'frame': 5, 'x': 0.45, 'y': 0.45, 'expected_label': 'Torso'}
        ]

        # Test ROI detection
        for point in test_points:
            frame = point['frame']
            x = point['x']
            y = point['y']
            expected_label = point['expected_label']

            roi = widget.roi_manager.find_roi_at_gaze(frame, x, y)
            self.assertIsNotNone(roi, f"Should find ROI at point ({x}, {y}) in frame {frame}")
            self.assertEqual(roi['label'], expected_label,
                             f"ROI at point ({x}, {y}) in frame {frame} should be {expected_label}")


if __name__ == '__main__':
    unittest.main()

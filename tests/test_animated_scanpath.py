import unittest
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animated_scanpath import create_animated_scanpath
from PyQt5.QtWidgets import QApplication

# Create a QApplication instance for testing
try:
    app = QApplication.instance()
    if app is None:
        # If running in CI, pass the offscreen platform argument
        if 'CI' in os.environ or 'GITHUB_ACTIONS' in os.environ:
            app = QApplication(['', '-platform', 'offscreen'])
        else:
            app = QApplication([])
except Exception as e:
    print(f"Warning: Could not initialize QApplication: {e}")
    print("Tests requiring GUI will be skipped")


class TestAnimatedScanpath(unittest.TestCase):
    """
    Test suite for the Animated Scanpath module.
    Tests validate the functionality of the animated scanpath visualization.
    """
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Create sample eye tracking data
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
        
        # Create a temporary directory for test outputs
        cls.temp_dir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove temporary directory and files
        cls.temp_dir.cleanup()
        
    def test_create_animated_scanpath(self):
        """Test creating an animated scanpath widget."""
        # Skip test if QApplication initialization failed
        if not QApplication.instance():
            self.skipTest("QApplication not available, skipping GUI test")
        
        # Create the widget
        try:
            widget = create_animated_scanpath(
                data=self.eye_data,
                movie_name="Test Movie",
                screen_width=1280,
                screen_height=1024
            )
            
            # Check that the widget was created
            self.assertIsNotNone(widget, "Widget should not be None")
            
            # Verify widget properties
            self.assertEqual(widget.movie_name, "Test Movie", "Movie name incorrect")
            self.assertEqual(widget.screen_width, 1280, "Screen width incorrect")
            self.assertEqual(widget.screen_height, 1024, "Screen height incorrect")
            
            # Verify data loading
            self.assertEqual(len(widget.data), 10, "Data should have 10 rows")
            self.assertIn('time_sec', widget.data.columns, "time_sec column missing")
            
            # Verify that total_duration is calculated correctly
            self.assertAlmostEqual(widget.total_duration, 0.09, 1, "Total duration incorrect")
            
        except Exception as e:
            self.fail(f"create_animated_scanpath raised exception: {str(e)}")

    def test_multiple_movie_loading(self):
        """Test loading multiple movies into the widget."""
        # Skip test if QApplication initialization failed
        if not QApplication.instance():
            self.skipTest("QApplication not available, skipping GUI test")
            
        # Create the widget with first movie
        widget = create_animated_scanpath(
            data=self.eye_data,
            movie_name="Movie 1",
            screen_width=1280,
            screen_height=1024
        )
        
        # Create a second movie dataset
        movie2_data = self.eye_data.copy()
        movie2_data['x_left'] += 100  # Offset x coordinates
        movie2_data['y_left'] += 100  # Offset y coordinates
        
        # Load the second movie
        result = widget.load_data(
            data=movie2_data,
            movie_name="Movie 2",
            screen_width=1280,
            screen_height=1024
        )
        
        # Verify the movie was loaded
        self.assertTrue(result, "Second movie should load successfully")
        self.assertIn("Movie 1", widget.loaded_movies, "Movie 1 should be in loaded_movies")
        self.assertIn("Movie 2", widget.loaded_movies, "Movie 2 should be in loaded_movies")
        
        # Check that both movies have data
        self.assertEqual(len(widget.loaded_movies["Movie 1"]["data"]), 10, "Movie 1 data incorrect")
        self.assertEqual(len(widget.loaded_movies["Movie 2"]["data"]), 10, "Movie 2 data incorrect")
        
    def test_error_handling_with_missing_columns(self):
        """Test error handling when loading data with missing columns."""
        # Skip test if QApplication initialization failed
        if not QApplication.instance():
            self.skipTest("QApplication not available, skipping GUI test")
            
        # Create data with missing columns
        bad_data = pd.DataFrame({
            'timestamp': list(range(1000, 1100, 10)),
            # Missing x_left, y_left, etc.
        })
        
        # Create the widget
        widget = create_animated_scanpath(
            data=self.eye_data,
            movie_name="Good Movie",
            screen_width=1280,
            screen_height=1024
        )
        
        # Try to load bad data
        result = widget.load_data(
            data=bad_data,
            movie_name="Bad Movie",
            screen_width=1280,
            screen_height=1024
        )
        
        # Verify the result
        self.assertFalse(result, "Loading data with missing columns should fail")
        self.assertNotIn("Bad Movie", widget.loaded_movies, "Bad Movie should not be in loaded_movies")

    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes."""
        # Skip test if QApplication initialization failed
        if not QApplication.instance():
            self.skipTest("QApplication not available, skipping GUI test")
            
        # Create an empty dataframe
        empty_data = pd.DataFrame()
        
        # Create the widget
        widget = create_animated_scanpath(
            data=self.eye_data,
            movie_name="Good Movie",
            screen_width=1280,
            screen_height=1024
        )
        
        # Try to load empty data
        result = widget.load_data(
            data=empty_data,
            movie_name="Empty Movie",
            screen_width=1280,
            screen_height=1024
        )
        
        # Verify the result
        self.assertFalse(result, "Loading empty data should fail")
        self.assertNotIn("Empty Movie", widget.loaded_movies, "Empty Movie should not be in loaded_movies")


if __name__ == '__main__':
    unittest.main()
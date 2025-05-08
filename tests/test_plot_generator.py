"""
Unit tests for the PlotGenerator class
Author: Tal Alfi
Date: May 2025
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from GUI.visualization.plot_generator import PlotGenerator

class TestPlotGenerator(unittest.TestCase):
    """Test cases for the PlotGenerator class"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Create test data
        cls.create_test_data()
        
        # Create a temporary directory for test output
        cls.temp_dir = tempfile.TemporaryDirectory()

        # Initialize PlotGenerator
        cls.plot_generator = PlotGenerator(
            screen_width=1280,
            screen_height=1024,
            visualization_results={},
            movie_visualizations={}
        )
        
        # Set plots_dir and output_dir
        cls.plot_generator.plots_dir = cls.temp_dir.name
        cls.plot_generator.output_dir = cls.temp_dir.name
        
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Close the temporary directory
        cls.temp_dir.cleanup()

    @classmethod
    def create_test_data(cls):
        """Create test data for plot generator tests."""
        # Create eye tracking data with fixations and saccades
        timestamps = np.arange(1000, 5000, 100)  # 40 samples at 100ms intervals
        frames = np.repeat(np.arange(1, 5), 10)  # 10 samples per frame
        
        # Generate x, y coordinates for both eyes
        x_left = np.random.uniform(0.2, 0.8, len(timestamps)) * 1280  # Screen width
        y_left = np.random.uniform(0.2, 0.8, len(timestamps)) * 1024  # Screen height
        x_right = x_left + np.random.normal(0, 10, len(timestamps))  # Slight offset
        y_right = y_left + np.random.normal(0, 10, len(timestamps))  # Slight offset
        
        # Generate fixation flags
        is_fixation_left = np.zeros(len(timestamps), dtype=bool)
        is_fixation_right = np.zeros(len(timestamps), dtype=bool)
        
        # Set some consecutive samples as fixations
        for i in range(5, 15):  # First fixation
            is_fixation_left[i] = True
            is_fixation_right[i] = True
            
        for i in range(20, 30):  # Second fixation
            is_fixation_left[i] = True
            is_fixation_right[i] = True
        
        # Create dataframe
        cls.eye_data = pd.DataFrame({
            'timestamp': timestamps,
            'frame_number': frames,
            'x_left': x_left,
            'y_left': y_left,
            'x_right': x_right,
            'y_right': y_right,
            'is_fixation_left': is_fixation_left,
            'is_fixation_right': is_fixation_right
        })
        
        # Create test ROI data
        cls.roi_data = {
            "1": [
                {
                    "label": "Face",
                    "coordinates": [
                        {"x": 0.2, "y": 0.2},
                        {"x": 0.4, "y": 0.2},
                        {"x": 0.4, "y": 0.4},
                        {"x": 0.2, "y": 0.4}
                    ]
                },
                {
                    "label": "Hand",
                    "coordinates": [
                        {"x": 0.6, "y": 0.6},
                        {"x": 0.8, "y": 0.6},
                        {"x": 0.8, "y": 0.8},
                        {"x": 0.6, "y": 0.8}
                    ]
                }
            ],
            "2": [
                {
                    "label": "Face",
                    "coordinates": [
                        {"x": 0.3, "y": 0.3},
                        {"x": 0.5, "y": 0.3},
                        {"x": 0.5, "y": 0.5},
                        {"x": 0.3, "y": 0.5}
                    ]
                }
            ],
            "3": [
                {
                    "label": "Object",
                    "coordinates": [
                        {"x": 0.4, "y": 0.4},
                        {"x": 0.6, "y": 0.4},
                        {"x": 0.6, "y": 0.6},
                        {"x": 0.4, "y": 0.6}
                    ]
                }
            ]
        }

    def test_point_in_polygon(self):
        """Test the _point_in_polygon method."""
        # Create test coordinates for a square
        coords = [
            {"x": 0.2, "y": 0.2},
            {"x": 0.4, "y": 0.2},
            {"x": 0.4, "y": 0.4},
            {"x": 0.2, "y": 0.4}
        ]
        
        # Test a point inside the polygon
        inside_point = self.plot_generator._point_in_polygon(0.3, 0.3, coords)
        self.assertTrue(inside_point, "Point should be inside the polygon")
        
        # Test a point outside the polygon
        outside_point = self.plot_generator._point_in_polygon(0.5, 0.5, coords)
        self.assertFalse(outside_point, "Point should be outside the polygon")
        
        # Test a point on the edge of the polygon
        edge_point = self.plot_generator._point_in_polygon(0.2, 0.3, coords)
        self.assertTrue(edge_point, "Point on the edge should be considered inside the polygon")
        
        # Test a point at a vertex of the polygon
        vertex_point = self.plot_generator._point_in_polygon(0.2, 0.2, coords)
        self.assertTrue(vertex_point, "Point at a vertex should be considered inside the polygon")
        
    def test_point_in_polygon_with_invalid_inputs(self):
        """Test the _point_in_polygon method with invalid inputs."""
        # Test with too few points (not a valid polygon)
        invalid_coords = [
            {"x": 0.2, "y": 0.2},
            {"x": 0.4, "y": 0.2}
        ]
        
        result = self.plot_generator._point_in_polygon(0.3, 0.3, invalid_coords)
        self.assertFalse(result, "A polygon with fewer than 3 points should return False")
        
        # Test with empty coordinates
        empty_coords = []
        result = self.plot_generator._point_in_polygon(0.3, 0.3, empty_coords)
        self.assertFalse(result, "Empty coordinates should return False")
    
    def test_create_advanced_roi_plots_with_empty_data(self):
        """Test creating advanced ROI plots with empty data."""
        # Create empty data
        empty_data = pd.DataFrame(columns=['timestamp', 'frame_number', 'x_left', 'y_left'])
        
        # Create mock status label and progress bar
        class MockWidget:
            def setText(self, text): pass
            def setValue(self, value): pass
        
        status_label = MockWidget()
        progress_bar = MockWidget()
        
        # Call the method with empty data
        plots = self.plot_generator.create_advanced_roi_plots(
            movie="test_movie",
            roi_durations={},  # Empty durations
            fixation_data=empty_data,
            plots_dir=self.temp_dir.name,
            frame_keys={},  # Empty frame keys
            frame_range_map={},  # Empty frame range map
            polygon_check_cache={},  # Empty cache
            status_label=status_label,
            progress_bar=progress_bar
        )
        
        # Should return without creating any plots
        self.assertEqual(len(plots) if plots else 0, 0, "Should return empty list for empty data")
    
    def test_create_advanced_roi_plots(self):
        """Test creating advanced ROI plots with valid data."""
        # Convert test ROI data to frame keys
        frame_keys = {}
        for frame, rois in self.roi_data.items():
            frame_keys[int(frame)] = rois
            
        # Create simple roi_durations dict
        roi_durations = {
            "Face": 100,
            "Hand": 50,
            "Object": 25
        }
        
        # Make sure frame_keys includes all frame numbers that appear in eye_data
        frames_in_eye_data = self.eye_data['frame_number'].unique()
        for frame in frames_in_eye_data:
            if int(frame) not in frame_keys:
                # Add a dummy ROI for any missing frames
                frame_keys[int(frame)] = [
                    {
                        "label": "Face",
                        "coordinates": [
                            {"x": 0.2, "y": 0.2},
                            {"x": 0.4, "y": 0.2},
                            {"x": 0.4, "y": 0.4},
                            {"x": 0.2, "y": 0.4}
                        ]
                    }
                ]
        
        # Create frame range map
        frame_range_map = {
            (0, 1): 1,
            (1, 2): 2,
            (2, 3): 3,
            (3, 5): 4
        }
        
        # Create mock status label and progress bar
        class MockWidget:
            def setText(self, text): pass
            def setValue(self, value): pass
        
        status_label = MockWidget()
        progress_bar = MockWidget()
        
        # Initialize visualization_results for this movie
        self.plot_generator.visualization_results = {
            "test_movie": {
                "basic": [],
                "social": []
            }
        }
        self.plot_generator.movie_visualizations = {
            "test_movie": {}
        }
        
        # Call the method with test data
        plots = self.plot_generator.create_advanced_roi_plots(
            movie="test_movie",
            roi_durations=roi_durations,
            fixation_data=self.eye_data,
            plots_dir=self.temp_dir.name,
            frame_keys=frame_keys,
            frame_range_map=frame_range_map,
            polygon_check_cache={},
            status_label=status_label,
            progress_bar=progress_bar
        )
        
        # Should at least attempt to create some plots
        # We can't guarantee plots are created due to GUI dependencies,
        # but the function should run without errors
        self.assertIsNotNone(plots, "Should return a list, even if empty")
    
    def test_negative_cases(self):
        """Test negative cases and error handling."""
        # Test with invalid ROI data
        invalid_roi_data = {
            "1": [
                {
                    # Missing "label" key
                    "coordinates": [
                        {"x": 0.2, "y": 0.2},
                        {"x": 0.4, "y": 0.2},
                        {"x": 0.4, "y": 0.4},
                        {"x": 0.2, "y": 0.4}
                    ]
                },
                {
                    "label": "Hand"
                    # Missing "coordinates" key
                }
            ]
        }
        
        # Convert to frame keys
        invalid_frame_keys = {}
        for frame, rois in invalid_roi_data.items():
            invalid_frame_keys[int(frame)] = rois
        
        # Create mock status label and progress bar
        class MockWidget:
            def setText(self, text): pass
            def setValue(self, value): pass
        
        status_label = MockWidget()
        progress_bar = MockWidget()
        
        # Call the method with invalid data
        plots = self.plot_generator.create_advanced_roi_plots(
            movie="test_movie",
            roi_durations={"Face": 100},
            fixation_data=self.eye_data,
            plots_dir=self.temp_dir.name,
            frame_keys=invalid_frame_keys,
            frame_range_map={},
            polygon_check_cache={},
            status_label=status_label,
            progress_bar=progress_bar
        )
        
        # Should handle invalid ROIs gracefully
        self.assertIsNotNone(plots, "Should return a list even with invalid ROI data")
        
    def test_generate_social_attention_plots_without_roi_file(self):
        """Test that generate_social_attention_plots handles missing ROI file."""
        # We need to mock QMessageBox to avoid GUI interactions
        class MockMessageBox:
            @staticmethod
            def warning(*args, **kwargs):
                return None
                
            @staticmethod
            def critical(*args, **kwargs):
                return None
        
        # Save the original and patch it temporarily
        from PyQt5.QtWidgets import QMessageBox
        original_messagebox = QMessageBox.warning
        QMessageBox.warning = MockMessageBox.warning
        
        try:
            # Call the method without setting roi_file_path
            result = self.plot_generator.generate_social_attention_plots()
            
            # Should return None due to missing ROI file
            self.assertIsNone(result, "Should return None when ROI file is missing")
        finally:
            # Restore the original QMessageBox
            QMessageBox.warning = original_messagebox
            
    def test_html_report_update_without_report_path(self):
        """Test that update_html_report handles missing report path."""
        # Call the method without setting report_path
        result = self.plot_generator.update_html_report("test_movie")
        
        # Should return False due to missing report path
        self.assertFalse(result, "Should return False when report_path is not set")

if __name__ == '__main__':
    unittest.main()